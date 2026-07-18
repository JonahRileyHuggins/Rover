"""Build and run a hybrid BNGsim + StochMod engine (plain Python orchestrator)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rover.modules.bngsim_module import BngsimModule, build_reaction_kernel
from rover.modules.stochmod_module import StochModModule
from rover.species_index import (
    SpeciesIndex,
    build_species_index,
    exchange_counts,
    local_to_global,
)
from rover.units import sbml_initial_nM

logger = logging.getLogger("rover")


def _ensure_rover_logging() -> None:
    if not logging.getLogger("rover").handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


@dataclass
class HybridEngine:
    """Shared nM state + both simulator modules (SingleCell-shaped)."""

    counts: np.ndarray
    index: SpeciesIndex
    bng: BngsimModule
    stoch: StochModModule
    codegen_active: bool


def build_hybrid_engine(
    deterministic_sbml: str | Path,
    stochastic_sbml: str | Path,
    *,
    dt: float = 1.0,
    initial_counts: np.ndarray | None = None,
    bngsim_kwargs: dict[str, Any] | None = None,
) -> HybridEngine:
    """Load BNGsim + StochMod once and wire them to a shared nanomolar vector.

    ``dt`` is recorded only for logging; each :func:`run_steps` call passes its
    own coupling interval into ``advance_from``.
    """
    _ensure_rover_logging()
    t_build0 = time.perf_counter()

    deterministic_sbml = Path(deterministic_sbml)
    stochastic_sbml = Path(stochastic_sbml)
    sim_kwargs = {
        "codegen": True,
        "jacobian": "analytical",
        "rtol": 1e-4,
        "atol": 1e-6,
        "max_steps": 100_000_000,
    }
    if bngsim_kwargs:
        sim_kwargs.update(bngsim_kwargs)

    index = build_species_index(
        stochastic_sbml,
        deterministic_sbml,
        deterministic_sbml=deterministic_sbml,
        stochastic_sbml=stochastic_sbml,
    )

    from stochmod import StochasticModule

    t0 = time.perf_counter()
    stoch_raw = StochasticModule(stochastic_sbml)
    logger.info(
        "StochMod load: %.3fs (%d species)",
        time.perf_counter() - t0,
        len(stoch_raw.species_names),
    )

    t0 = time.perf_counter()
    kernel, codegen_active = build_reaction_kernel(
        deterministic_sbml,
        method="ode",
        simulator_kwargs=sim_kwargs,
    )
    logger.info(
        "BNGsim load: %.3fs (codegen=%s, %d species)",
        time.perf_counter() - t0,
        codegen_active,
        len(kernel.state_names),
    )

    if initial_counts is None:
        counts = np.zeros(index.n_species, dtype=np.float64)
        bng_names = list(kernel.state_names)
        bng_nM = np.asarray(kernel.get_state(), dtype=np.float64)
        for i, name in enumerate(bng_names):
            counts[index.name_to_index[name]] = float(bng_nM[i])

        stoch_nM = sbml_initial_nM(stochastic_sbml)
        # Stoch-only: take stochastic SBML nM ICs. Overlap already seeded from
        # BNGsim; warn if stochastic SBML disagrees.
        mismatch = 0
        for name in stoch_raw.species_names:
            gi = index.name_to_index[name]
            sm = float(stoch_nM.get(name, 0.0))
            if index.stochastic_only_mask[gi]:
                counts[gi] = sm
            elif index.overlap_mask[gi]:
                bng_v = float(counts[gi])
                scale = max(abs(bng_v), abs(sm), 1e-12)
                if abs(bng_v - sm) > 1e-3 * scale + 1e-12:
                    mismatch += 1
        if mismatch:
            logger.warning(
                "Overlap IC mismatch on %d species (BNGsim vs StochMod SBML); "
                "using BNGsim nM values for overlap",
                mismatch,
            )
        initial_counts = counts
        logger.info(
            "Membership: %d det-only / %d stoch-only / %d overlap (N=%d); "
            "ownership: %d stoch-owned / %d det-owned; shared currency=nM",
            int(index.deterministic_only_mask.sum()),
            int(index.stochastic_only_mask.sum()),
            int(index.overlap_mask.sum()),
            index.n_species,
            int(index.stoch_owned_mask.sum()),
            int(index.det_owned_mask.sum()),
        )
    else:
        initial_counts = np.asarray(initial_counts, dtype=np.float64)
        if initial_counts.shape != (index.n_species,):
            raise ValueError(
                f"initial_counts shape {initial_counts.shape} != ({index.n_species},)"
            )

    det_indices = local_to_global(list(kernel.state_names), index).tolist()
    stoch_indices = local_to_global(stoch_raw.species_names, index).tolist()

    bng = BngsimModule(
        kernel=kernel,
        local_indices=det_indices,
        n_species=index.n_species,
        codegen_active=codegen_active,
        stoch_owned_global=index.stoch_owned_mask,
    )
    stoch = StochModModule(
        module=stoch_raw,
        sbml_path=stochastic_sbml,
        local_indices=stoch_indices,
        n_species=index.n_species,
        companion_deterministic_sbml=deterministic_sbml,
        det_owned_global=index.det_owned_mask,
    )

    engine = HybridEngine(
        counts=np.asarray(initial_counts, dtype=np.float64).copy(),
        index=index,
        bng=bng,
        stoch=stoch,
        codegen_active=codegen_active,
    )
    logger.info(
        "Hybrid engine ready in %.3fs (N=%d, dt=%g, codegen=%s)",
        time.perf_counter() - t_build0,
        index.n_species,
        dt,
        codegen_active,
    )
    return engine


def run_steps(
    engine: HybridEngine,
    *,
    t_end: float,
    dt: float,
    progress_every: int | None = None,
    trajectory=None,
    t0_abs: float = 0.0,
    counts: np.ndarray | None = None,
) -> np.ndarray:
    """Advance both modules from the same pre-step state, then exchange.

    Per coupling step (SingleCell-shaped)::

        s0 = counts
        s_stoch = stoch.advance_from(s0, dt)
        s_bng   = bng.advance_from(s0, dt)
        counts  = exchange(s0, s_stoch, s_bng)   # exclusive copy + overlap deltas

    Absolute times written to ``trajectory`` are ``t0_abs + step * dt``.
    """
    _ensure_rover_logging()
    if counts is None:
        counts = np.asarray(engine.counts, dtype=np.float64).copy()
    else:
        counts = np.asarray(counts, dtype=np.float64)

    n_steps = int(np.floor(float(t_end) / float(dt)))
    if n_steps < 1:
        raise ValueError(f"t_end={t_end} / dt={dt} yields no steps")

    if progress_every is None:
        progress_every = max(1, n_steps // 10)

    logger.info(
        "Coupled run: t_end=%g, dt=%g, steps=%d (codegen=%s)",
        t_end,
        dt,
        n_steps,
        engine.codegen_active,
    )

    if trajectory is not None:
        if trajectory.n_points != n_steps + 1:
            raise ValueError(
                f"trajectory.n_points={trajectory.n_points} != n_steps+1={n_steps + 1}"
            )
        trajectory.record(0, float(t0_abs), counts)

    wall0 = time.perf_counter()
    t_bng = 0.0
    t_stoch = 0.0
    t_rover = 0.0
    for step in range(1, n_steps + 1):
        s0 = counts
        t_a = time.perf_counter()
        local_stoch = engine.stoch.advance_from(s0, dt)
        t_b = time.perf_counter()
        local_bng = engine.bng.advance_from(s0, dt)
        t_c = time.perf_counter()

        # Module-reported integrate vs bridge; residual wall time is orchestrator.
        t_stoch += float(getattr(engine.stoch, "last_integrate_s", t_b - t_a))
        t_bng += float(getattr(engine.bng, "last_integrate_s", t_c - t_b))
        t_rover += float(getattr(engine.stoch, "last_bridge_s", 0.0))
        t_rover += float(getattr(engine.bng, "last_bridge_s", 0.0))

        counts = exchange_counts(
            s0,
            stoch_local=local_stoch,
            stoch_indices=engine.stoch.local_indices,
            bng_local=local_bng,
            bng_indices=engine.bng.local_indices,
            index=engine.index,
        )

        if trajectory is not None:
            trajectory.record(step, float(t0_abs + step * dt), counts)

        t_d = time.perf_counter()
        t_rover += t_d - t_c

        if step % progress_every == 0 or step == n_steps:
            elapsed = time.perf_counter() - wall0
            logger.info(
                "  step %d/%d (t=%.4g)  elapsed=%.2fs  "
                "stoch=%.2fs  bng=%.2fs  rover=%.2fs",
                step,
                n_steps,
                t0_abs + step * dt,
                elapsed,
                t_stoch,
                t_bng,
                t_rover,
            )

    if trajectory is not None:
        trajectory.flush()

    engine.counts = counts
    total = time.perf_counter() - wall0
    logger.info(
        "Coupled run done in %.3fs (stoch=%.1f%% bng=%.1f%% rover=%.1f%%)",
        total,
        100.0 * t_stoch / total if total else 0.0,
        100.0 * t_bng / total if total else 0.0,
        100.0 * t_rover / total if total else 0.0,
    )
    return counts


def run_hybrid(
    deterministic_sbml: str | Path,
    stochastic_sbml: str | Path,
    *,
    t_end: float,
    dt: float = 1.0,
    initial_counts: np.ndarray | None = None,
    bngsim_kwargs: dict[str, Any] | None = None,
    results_path: str | Path | None = None,
) -> tuple[np.ndarray, SpeciesIndex]:
    """One-shot hybrid run (builds a :class:`~rover.simulator.HybridSimulator`).

    Returns ``(trajectory, index)`` where trajectory has shape
    ``(n_steps + 1, n_species)``.
    """
    from rover.simulator import HybridSimulator

    sim = HybridSimulator(
        deterministic_sbml,
        stochastic_sbml,
        dt=dt,
        initial_counts=initial_counts,
        bngsim_kwargs=bngsim_kwargs,
    )
    traj = sim.run(t_end=t_end, results_path=results_path)
    return traj, sim.index

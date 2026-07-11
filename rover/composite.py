"""Build and run a hybrid BNGsim + StochMod process-bigraph composite."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from process_bigraph import Composite, allocate_core, register_types

from rover.processes.bngsim_process import BngsimProcess, build_reaction_kernel
from rover.processes.stochmod_process import StochModProcess
from rover.species_index import SpeciesIndex, build_species_index, local_to_global
from rover.units import counts_from_bngsim_storage, stochmod_to_molecule_scales

logger = logging.getLogger("rover")


def _ensure_rover_logging() -> None:
    if not logging.getLogger("rover").handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


def make_core(extra_top: dict[str, Any] | None = None):
    """Allocate a process-bigraph core with Rover process classes in scope."""
    top = {
        "BngsimProcess": BngsimProcess,
        "StochModProcess": StochModProcess,
    }
    if extra_top:
        top.update(extra_top)
    return register_types(allocate_core(top=top))


def _get_process_instance(composite: Composite, name: str):
    node = composite.state[name]
    if isinstance(node, dict) and "instance" in node:
        return node["instance"]
    raise KeyError(f"No process instance at '{name}'")


def build_hybrid_composite(
    deterministic_sbml: str | Path,
    stochastic_sbml: str | Path,
    *,
    dt: float = 1.0,
    initial_counts: np.ndarray | None = None,
    core=None,
    bngsim_kwargs: dict[str, Any] | None = None,
) -> tuple[Composite, SpeciesIndex]:
    """Wire BNGsim + StochMod processes to a shared molecule-count array store.

    Models are loaded **once** and passed into the process configs so Composite
    realization does not rebuild StochMod / BNGsim a second (or third) time.

    Returns
    -------
    composite, species_index
    """
    _ensure_rover_logging()
    t_build0 = time.perf_counter()

    deterministic_sbml = Path(deterministic_sbml)
    stochastic_sbml = Path(stochastic_sbml)
    sim_kwargs = {
        "codegen": True,
        "jacobian": "fd",
        "rtol": 1e-4,
        "atol": 1e-6,
        "max_steps": 100_000,
    }
    if bngsim_kwargs:
        sim_kwargs.update(bngsim_kwargs)

    index = build_species_index(
        stochastic_sbml,
        deterministic_sbml,
        ownership_sbml=stochastic_sbml,
        deterministic_sbml=deterministic_sbml,
        stochastic_sbml=stochastic_sbml,
    )

    # --- single load of each simulator ---
    from stochmod import StochasticModule

    t0 = time.perf_counter()
    stoch_module = StochasticModule(stochastic_sbml)
    logger.info("StochMod load: %.3fs (%d species)", time.perf_counter() - t0, len(stoch_module.species_names))

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

    # Initial counts: BNGsim ICs for all det species, then StochMod amounts
    # overwrite for stoch species. BNGsim-only species must not stay at 0.
    if initial_counts is None:
        import bngsim

        counts = np.zeros(index.n_species, dtype=np.float64)
        uc = bngsim.UnitConverter.from_model(kernel.model)
        det_counts = counts_from_bngsim_storage(kernel.get_state(), uc)
        for i, name in enumerate(kernel.state_names):
            counts[index.name_to_index[name]] = float(det_counts[i])

        stoch_state = stoch_module.get_state()
        stoch_scales = stochmod_to_molecule_scales(stochastic_sbml)
        if stoch_scales.shape[0] != len(stoch_module.species_names):
            raise ValueError("StochMod unit scales do not match species_names")
        stoch_molecules = np.asarray(stoch_state, dtype=np.float64) * stoch_scales
        for i, name in enumerate(stoch_module.species_names):
            counts[index.name_to_index[name]] = float(stoch_molecules[i])
        initial_counts = counts
        logger.info(
            "Ownership: %d deterministic / %d stochastic writers (N=%d)",
            int(index.deterministic_mask.sum()),
            int(index.stochastic_mask.sum()),
            index.n_species,
        )
    else:
        initial_counts = np.asarray(initial_counts, dtype=np.float64)
        if initial_counts.shape != (index.n_species,):
            raise ValueError(
                f"initial_counts shape {initial_counts.shape} != ({index.n_species},)"
            )

    det_indices = local_to_global(list(kernel.state_names), index).tolist()
    stoch_indices = local_to_global(stoch_module.species_names, index).tolist()

    if core is None:
        core = make_core()

    state = {
        "counts": np.asarray(initial_counts, dtype=np.float64).copy(),
        "bngsim": {
            "_type": "process",
            "address": "local:!rover.processes.bngsim_process.BngsimProcess",
            "config": {
                "kernel": kernel,
                "local_indices": det_indices,
                "ownership_mask": index.deterministic_mask.tolist(),
                "n_species": index.n_species,
                "method": "ode",
                "simulator_kwargs": sim_kwargs,
                "time_step": float(dt),
            },
            "interval": float(dt),
            "inputs": {"counts": ["counts"]},
            "outputs": {"counts": ["counts"]},
        },
        "stochmod": {
            "_type": "process",
            "address": "local:!rover.processes.stochmod_process.StochModProcess",
            "config": {
                "sbml_path": str(stochastic_sbml.resolve()),
                "module": stoch_module,
                "local_indices": stoch_indices,
                "ownership_mask": index.stochastic_mask.tolist(),
                "n_species": index.n_species,
                "time_step": float(dt),
            },
            "interval": float(dt),
            "inputs": {"counts": ["counts"]},
            "outputs": {"counts": ["counts"]},
        },
    }

    composite = Composite({"state": state}, core=core)
    logger.info(
        "Hybrid composite ready in %.3fs (N=%d, dt=%g, codegen=%s)",
        time.perf_counter() - t_build0,
        index.n_species,
        dt,
        codegen_active,
    )
    return composite, index


def run_operator_split(
    composite: Composite,
    *,
    t_end: float,
    dt: float,
    progress_every: int | None = None,
    trajectory=None,
    t0_abs: float = 0.0,
) -> np.ndarray:
    """Tight operator-split loop — bypasses per-step process-bigraph scheduling.

    Live state is a 1-D ``counts`` vector in RAM. If ``trajectory`` is given
    (a :class:`~rover.trajectory.TrajectoryStore`), each coupling step writes
    one row into that pre-sized buffer (memory or memmap) — the kernels never
    touch disk themselves.
    """
    _ensure_rover_logging()
    bng = _get_process_instance(composite, "bngsim")
    stoch = _get_process_instance(composite, "stochmod")
    counts = np.asarray(composite.state["counts"], dtype=np.float64).copy()

    n_steps = int(np.floor(float(t_end) / float(dt)))
    if n_steps < 1:
        raise ValueError(f"t_end={t_end} / dt={dt} yields no steps")

    if progress_every is None:
        progress_every = max(1, n_steps // 10)

    logger.info(
        "Operator-split run: t_end=%g, dt=%g, steps=%d (codegen=%s)",
        t_end,
        dt,
        n_steps,
        getattr(bng, "codegen_active", "?"),
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
    for step in range(1, n_steps + 1):
        t_a = time.perf_counter()
        stoch.apply_inplace(counts, dt)
        t_b = time.perf_counter()
        bng.apply_inplace(counts, dt)
        t_c = time.perf_counter()
        t_stoch += t_b - t_a
        t_bng += t_c - t_b

        if trajectory is not None:
            trajectory.record(step, float(t0_abs + step * dt), counts)

        if step % progress_every == 0 or step == n_steps:
            elapsed = time.perf_counter() - wall0
            logger.info(
                "  step %d/%d (t=%.4g)  elapsed=%.2fs  "
                "stoch=%.2fs  bng=%.2fs  us/step=%.1f",
                step,
                n_steps,
                t0_abs + step * dt,
                elapsed,
                t_stoch,
                t_bng,
                1e6 * elapsed / step,
            )

    if trajectory is not None:
        trajectory.flush()

    composite.state["counts"] = counts
    total = time.perf_counter() - wall0
    logger.info(
        "Operator-split done in %.3fs (%.1f us/step; stoch=%.1f%% bng=%.1f%%)",
        total,
        1e6 * total / n_steps,
        100.0 * t_stoch / total if total else 0.0,
        100.0 * t_bng / total if total else 0.0,
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
    engine: str = "split",
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
    traj = sim.run(t_end=t_end, engine=engine, results_path=results_path)
    return traj, sim.index

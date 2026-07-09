"""Build and run a hybrid BNGsim + StochMod process-bigraph composite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from process_bigraph import Composite, allocate_core, register_types

from rover.species_index import SpeciesIndex, build_species_index, local_to_global
from rover.units import counts_from_bngsim_storage


def make_core(extra_top: dict[str, Any] | None = None):
    """Allocate a process-bigraph core with Rover process classes in scope."""
    from rover.processes.bngsim_process import BngsimProcess
    from rover.processes.stochmod_process import StochModProcess

    top = {
        "BngsimProcess": BngsimProcess,
        "StochModProcess": StochModProcess,
    }
    if extra_top:
        top.update(extra_top)
    return register_types(allocate_core(top=top))


def _initial_counts_from_models(
    index: SpeciesIndex,
    deterministic_sbml: str | Path,
    stochastic_sbml: str | Path,
) -> np.ndarray:
    """Seed the shared count store from both SBMLs.

    Stochastic amounts are preferred for stochastic-owned species.
    Deterministic nM ICs (via BNGsim) seed deterministic-owned species.
    """
    import bngsim
    from stochmod import StochasticModule

    counts = np.zeros(index.n_species, dtype=np.float64)

    # Stochastic amounts (molecule counts)
    stoch = StochasticModule(stochastic_sbml)
    stoch_names = stoch.species_names
    stoch_state = stoch.get_state()
    for i, name in enumerate(stoch_names):
        gi = index.name_to_index[name]
        if index.stochastic_mask[gi]:
            counts[gi] = float(stoch_state[i])

    # Deterministic concentrations → counts for protein species
    model = bngsim.Model.from_sbml(str(deterministic_sbml))
    kernel = bngsim.ReactionKernel(model, method="ode")
    uc = bngsim.UnitConverter.from_model(model)
    det_counts = counts_from_bngsim_storage(kernel.get_state(), uc)
    for i, name in enumerate(kernel.state_names):
        gi = index.name_to_index[name]
        if index.deterministic_mask[gi]:
            counts[gi] = float(det_counts[i])

    return counts


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

    Returns
    -------
    composite, species_index
    """
    deterministic_sbml = Path(deterministic_sbml)
    stochastic_sbml = Path(stochastic_sbml)

    index = build_species_index(
        stochastic_sbml,
        deterministic_sbml,
        ownership_sbml=stochastic_sbml,
    )

    if initial_counts is None:
        initial_counts = _initial_counts_from_models(
            index, deterministic_sbml, stochastic_sbml
        )
    else:
        initial_counts = np.asarray(initial_counts, dtype=np.float64)
        if initial_counts.shape != (index.n_species,):
            raise ValueError(
                f"initial_counts shape {initial_counts.shape} != ({index.n_species},)"
            )

    # Local index maps (kernel / module species order → global)
    import bngsim
    from stochmod import StochasticModule

    det_model = bngsim.Model.from_sbml(str(deterministic_sbml))
    det_names = list(bngsim.ReactionKernel(det_model, method="ode").state_names)
    stoch_names = StochasticModule(stochastic_sbml).species_names

    det_indices = local_to_global(det_names, index).tolist()
    stoch_indices = local_to_global(stoch_names, index).tolist()

    if core is None:
        core = make_core()

    state = {
        "counts": np.asarray(initial_counts, dtype=np.float64).copy(),
        "bngsim": {
            "_type": "process",
            "address": "local:!rover.processes.bngsim_process.BngsimProcess",
            "config": {
                "sbml_path": str(deterministic_sbml.resolve()),
                "local_indices": det_indices,
                "ownership_mask": index.deterministic_mask.tolist(),
                "n_species": index.n_species,
                "method": "ode",
                "simulator_kwargs": dict(bngsim_kwargs or {}),
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
    return composite, index


def run_hybrid(
    deterministic_sbml: str | Path,
    stochastic_sbml: str | Path,
    *,
    t_end: float,
    dt: float = 1.0,
    initial_counts: np.ndarray | None = None,
    bngsim_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, SpeciesIndex]:
    """Run the hybrid composite to ``t_end`` and return final counts + index."""
    composite, index = build_hybrid_composite(
        deterministic_sbml,
        stochastic_sbml,
        dt=dt,
        initial_counts=initial_counts,
        bngsim_kwargs=bngsim_kwargs,
    )
    composite.run(float(t_end))
    return np.asarray(composite.state["counts"], dtype=np.float64), index

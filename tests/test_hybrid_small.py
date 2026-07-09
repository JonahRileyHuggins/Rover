"""Hybrid BNGsim + StochMod integration on the small test SBMLs."""

from pathlib import Path

import bngsim
import numpy as np
import pytest
from stochmod import StochasticModule

from rover.composite import build_hybrid_composite, run_hybrid
from rover.processes.stochmod_process import StochModProcess
from rover.species_index import build_species_index, local_to_global

DATA = Path(__file__).resolve().parent / "data"
DET = DATA / "deterministic-interactions.xml"
STOCH = DATA / "stochastic-gene-expression.xml"


def test_stochmod_standalone_advance():
    mod = StochasticModule(STOCH)
    s0 = mod.get_state()
    s1 = mod.advance(10.0)
    assert s1.shape == s0.shape
    # run() still works and resets from initial_state
    traj = mod.run(0.0, 5.0, 1.0)
    assert traj.ndim == 2
    assert traj.shape[1] == len(mod.species_names)


def test_bngsim_standalone_advance():
    model = bngsim.Model.from_sbml(str(DET))
    kernel = bngsim.ReactionKernel(model, method="ode")
    s0 = kernel.get_state()
    s1 = kernel.advance(1.0)
    assert s1.shape == s0.shape


def test_process_update_returns_delta_array():
    from rover.composite import make_core

    index = build_species_index(STOCH, DET, ownership_sbml=STOCH)
    stoch_names = StochasticModule(STOCH).species_names
    idxs = local_to_global(stoch_names, index).tolist()
    core = make_core()

    proc = StochModProcess(
        {
            "sbml_path": str(STOCH.resolve()),
            "local_indices": idxs,
            "ownership_mask": index.stochastic_mask.tolist(),
            "n_species": index.n_species,
        },
        core=core,
    )
    counts = np.zeros(index.n_species, dtype=np.float64)
    for i, name in enumerate(stoch_names):
        counts[index.name_to_index[name]] = float(proc.module.get_state()[i])

    update = proc.update({"counts": counts}, 1.0)
    assert "counts" in update
    assert update["counts"].shape == (index.n_species,)
    # Deterministic-owned slots must stay zero in the delta
    assert np.all(update["counts"][index.deterministic_mask] == 0.0)


def test_hybrid_short_run_proteins_rise():
    """mRNA is present; translation should produce protein over a short run."""
    from rover.composite import run_operator_split

    composite, index = build_hybrid_composite(DET, STOCH, dt=1.0)
    prot_i = index.name_to_index["cyt_prot__LIGAND_"]
    mrna_i = index.name_to_index["cyt_mrna__LIGAND_"]

    counts0 = np.asarray(composite.state["counts"], dtype=np.float64).copy()
    assert counts0[mrna_i] == pytest.approx(5.0, rel=1e-2)
    assert counts0[prot_i] == pytest.approx(0.0, abs=1e-6)

    counts1 = run_operator_split(composite, t_end=50.0, dt=1.0)

    # Translation from mRNA should increase ligand protein
    assert counts1[prot_i] > counts0[prot_i] + 0.1


def test_run_hybrid_helper():
    counts, index = run_hybrid(DET, STOCH, t_end=5.0, dt=1.0, engine="split")
    assert counts.shape == (index.n_species,)
    assert np.all(np.isfinite(counts))


def test_run_hybrid_composite_engine():
    counts, index = run_hybrid(DET, STOCH, t_end=5.0, dt=1.0, engine="composite")
    assert counts.shape == (index.n_species,)
    assert np.all(np.isfinite(counts))

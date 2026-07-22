"""Hybrid BNGsim + StochMod integration on the small LR SBMLs."""

from pathlib import Path

import bngsim
import numpy as np
import pytest
from stochmod import StochasticModule

from rover.engine import build_hybrid_engine, run_hybrid, run_steps
from rover.modules.stochmod_module import StochModModule
from rover.species_index import build_species_index, local_to_global
from rover.units import sbml_initial_nM

DATA = Path(__file__).resolve().parent / "data" / "LR"
DET = DATA / "deterministic-interactions.xml"
STOCH = DATA / "stochastic-gene-expression.xml"

pytestmark = pytest.mark.skipif(
    not DET.exists() or not STOCH.exists(),
    reason="LR fixtures not present",
)


def test_stochmod_standalone_advance():
    mod = StochasticModule(STOCH)
    s0 = mod.get_state()
    s1 = mod.advance(10.0)
    assert s1.shape == s0.shape
    traj = mod.run(0.0, 5.0, 1.0)
    assert traj.ndim == 2
    assert traj.shape[1] == len(mod.species_names)


def test_bngsim_standalone_advance():
    model = bngsim.Model.from_sbml(str(DET))
    kernel = bngsim.ReactionKernel(model, method="ode")
    s0 = kernel.get_state()
    s1 = kernel.advance(1.0)
    assert s1.shape == s0.shape


def test_stochmod_advance_from_returns_local_without_mutating_global():
    index = build_species_index(
        STOCH, DET, deterministic_sbml=DET, stochastic_sbml=STOCH
    )
    raw = StochasticModule(STOCH)
    idxs = local_to_global(raw.species_names, index).tolist()
    mod = StochModModule(
        module=raw,
        sbml_path=STOCH,
        local_indices=idxs,
        n_species=index.n_species,
    )
    stoch_nM = sbml_initial_nM(STOCH)
    counts = np.zeros(index.n_species, dtype=np.float64)
    for name in raw.species_names:
        counts[index.name_to_index[name]] = float(stoch_nM[name])
    before = counts.copy()
    local = mod.advance_from(counts, 1.0)
    np.testing.assert_array_equal(counts, before)
    assert local.shape == (len(idxs),)
    assert np.all(np.isfinite(local))
    assert np.all(local >= 0.0)


def test_hybrid_short_run_proteins_rise():
    """mRNA is present; translation should produce protein over a short run."""
    engine = build_hybrid_engine(DET, STOCH, dt=1.0)
    prot_i = engine.index.name_to_index["cyt_prot__LIGAND_"]
    mrna_i = engine.index.name_to_index["cyt_mrna__LIGAND_"]

    counts0 = engine.counts.copy()
    # Overlap seeded from BNGsim (deterministic SBML nM)
    assert counts0[mrna_i] == pytest.approx(0.001582, rel=1e-3)
    assert counts0[prot_i] == pytest.approx(0.0, abs=1e-12)

    counts1 = run_steps(engine, t_end=50.0, dt=1.0)
    assert counts1[prot_i] > counts0[prot_i]


def test_run_hybrid_helper():
    traj, index = run_hybrid(DET, STOCH, t_end=5.0, dt=1.0)
    assert traj.ndim == 2
    assert traj.shape == (6, index.n_species)
    assert np.all(np.isfinite(traj))

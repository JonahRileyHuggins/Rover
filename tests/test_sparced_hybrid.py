"""SPARCED partition smoke tests (exchange + short hybrid run)."""

from pathlib import Path

import numpy as np
import pytest

from rover import HybridSimulator
from rover.species_index import build_species_index
from rover.units import sbml_initial_nM

DATA = Path(__file__).resolve().parent / "data" / "SPARCED"
DET = DATA / "deterministic-interactions-v1.4.2.xml"
STOCH = DATA / "stochastic-gene-expression-v1.4.2.xml"

pytestmark = pytest.mark.skipif(
    not DET.exists() or not STOCH.exists(),
    reason="SPARCED v1.4.2 fixtures not present",
)


def test_sparced_membership_allows_overlap():
    index = build_species_index(
        STOCH,
        DET,
        deterministic_sbml=DET,
        stochastic_sbml=STOCH,
    )
    assert index.n_species > 1000
    assert index.deterministic_mask.sum() > 0
    assert index.stochastic_mask.sum() > 0
    assert index.overlap_mask.sum() > 0
    assert (
        index.deterministic_only_mask.sum()
        + index.stochastic_only_mask.sum()
        + index.overlap_mask.sum()
        == index.n_species
    )


def test_sparced_sbml_initial_nM():
    stoch = sbml_initial_nM(STOCH)
    assert stoch["cyt_mrna__MAPK1_201_"] == pytest.approx(0.001897488, rel=1e-6)
    assert stoch["cyt_mrna__MAPK3_201_"] == pytest.approx(0.0006324961, rel=1e-6)


def test_sparced_mrna_seeded_as_nM():
    """Overlap mRNA must stay at SBML nanomolar ICs, not molecule counts."""
    from rover.engine import build_hybrid_engine

    engine = build_hybrid_engine(DET, STOCH, dt=30.0)
    i = engine.index.name_to_index["cyt_mrna__MAPK1_201_"]
    assert engine.counts[i] == pytest.approx(0.001897488, rel=0, abs=1e-12)
    j = engine.index.name_to_index["cyt_mrna__MAPK3_201_"]
    assert engine.counts[j] == pytest.approx(0.0006324961, rel=0, abs=1e-12)
    # Det-only gene stays at true nM (not inflated by V·N_A·1e-9)
    rb1 = engine.index.name_to_index["nuc_gene_a__RB1_"]
    assert engine.counts[rb1] == pytest.approx(0.001897488, rel=0, abs=1e-12)


def test_sparced_hybrid_short_run():
    sim = HybridSimulator(
        DET,
        STOCH,
        dt=30.0,
        bngsim_kwargs={"codegen": False, "jacobian": "fd"},
    )
    bng = sim._bng
    native = np.asarray(bng.kernel.get_state(), dtype=np.float64)
    fed = np.asarray(sim.counts[bng._local_indices], dtype=np.float64)
    # Shared store is nM; BNGsim identity bridge feeds the same values.
    stoch_names = set(sim._stoch.module.species_names)
    bng_only = np.array(
        [name not in stoch_names for name in bng.kernel.state_names],
        dtype=bool,
    )
    np.testing.assert_allclose(fed[bng_only], native[bng_only], rtol=1e-6, atol=1e-12)

    traj = sim.run(t_end=60.0, dt=30.0)
    assert traj.shape == (3, sim.index.n_species)
    assert np.all(np.isfinite(traj))
    assert np.all(traj >= 0.0)
    assert sim._bng.kernel.time == pytest.approx(30.0, rel=1e-9, abs=1e-9)
    assert sim.time == pytest.approx(60.0)

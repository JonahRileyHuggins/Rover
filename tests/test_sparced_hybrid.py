"""SPARCED partition smoke tests (exchange + short hybrid run)."""

from pathlib import Path

import numpy as np
import pytest

from rover import HybridSimulator
from rover.species_index import build_species_index
from rover.units import stochmod_to_molecule_scales

DATA = Path(__file__).resolve().parent / "data" / "SPARCED"
DET = DATA / "deterministic-interactions.xml"
STOCH = DATA / "stochastic-gene-expression.xml"

pytestmark = pytest.mark.skipif(
    not DET.exists() or not STOCH.exists(),
    reason="SPARCED fixtures not present",
)


def test_sparced_membership_allows_overlap():
    index = build_species_index(
        STOCH,
        DET,
        deterministic_sbml=DET,
        stochastic_sbml=STOCH,
    )
    assert index.n_species == 1201
    assert index.deterministic_mask.sum() > 0
    assert index.stochastic_mask.sum() > 0
    assert index.overlap_mask.sum() > 0
    assert (
        index.deterministic_only_mask.sum()
        + index.stochastic_only_mask.sum()
        + index.overlap_mask.sum()
        == index.n_species
    )


def test_sparced_stochmod_scales_substance_units():
    scales = stochmod_to_molecule_scales(STOCH, companion_deterministic_sbml=DET)
    assert scales.shape[0] == 452
    assert np.all(scales > 0.0)
    assert np.all(np.isfinite(scales))
    # Dual-encoded overlap + genes are already molecule counts in the stoch file
    assert np.all(scales == 1.0)


def test_sparced_mrna_seeded_as_copy_counts():
    """Overlap mRNA must stay at SBML numerics (~1-20), not V·N_A·1e-9 molecules."""
    from rover.engine import build_hybrid_engine
    from rover.units import overlap_currency_modes

    modes = overlap_currency_modes(STOCH, DET)
    assert modes.get("cyt_mrna__MAPK1_201_") == "numeric"

    engine = build_hybrid_engine(DET, STOCH, dt=30.0)
    i = engine.index.name_to_index["cyt_mrna__MAPK1_201_"]
    assert engine.counts[i] == pytest.approx(6.0, rel=0, abs=1e-9)
    j = engine.index.name_to_index["cyt_mrna__MAPK3_201_"]
    assert engine.counts[j] == pytest.approx(2.0, rel=0, abs=1e-9)


def test_sparced_hybrid_short_run():
    sim = HybridSimulator(
        DET,
        STOCH,
        dt=30.0,
        bngsim_kwargs={"codegen": False, "jacobian": "fd"},
    )
    bng = sim._bng
    native = bng.kernel.get_state()
    fed = bng._converter.storage_from_counts(sim.counts[bng._local_indices])
    # Overlap seeded from BNGsim; det-only must match native exactly.
    stoch_names = set(sim._stoch.module.species_names)
    bng_only = np.array(
        [name not in stoch_names for name in bng.kernel.state_names],
        dtype=bool,
    )
    np.testing.assert_allclose(fed[bng_only], native[bng_only], rtol=1e-6, atol=1e-9)

    traj = sim.run(t_end=60.0, dt=30.0)
    assert traj.shape == (3, sim.index.n_species)
    assert np.all(np.isfinite(traj))
    assert np.all(traj >= 0.0)
    assert sim._bng.kernel.time == pytest.approx(30.0, rel=1e-9, abs=1e-9)
    assert sim.time == pytest.approx(60.0)

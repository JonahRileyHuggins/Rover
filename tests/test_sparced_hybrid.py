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


def test_sparced_ownership_not_all_stochastic():
    index = build_species_index(
        STOCH,
        DET,
        deterministic_sbml=DET,
        stochastic_sbml=STOCH,
    )
    assert index.n_species == 1201
    assert index.deterministic_mask.sum() > 0
    assert index.stochastic_mask.sum() > 0
    assert index.deterministic_mask.sum() + index.stochastic_mask.sum() == index.n_species


def test_sparced_stochmod_scales_nanomole_to_molecule():
    scales = stochmod_to_molecule_scales(STOCH)
    assert scales.shape[0] == 452
    assert np.all(scales > 1.0)  # concentration-based nM model


def test_sparced_hybrid_short_run():
    sim = HybridSimulator(
        DET,
        STOCH,
        dt=30.0,
        bngsim_kwargs={"codegen": False, "jacobian": "fd"},
    )
    # Fed BNGsim state should match native ICs after unit bridge
    bng = sim._bng
    native = bng.kernel.get_state()
    fed = bng._converter.storage_from_counts(sim.counts[bng._local_indices])
    np.testing.assert_allclose(fed, native, rtol=1e-6, atol=1e-9)

    traj = sim.run(t_end=60.0, dt=30.0)
    assert traj.shape == (3, sim.index.n_species)
    assert np.all(np.isfinite(traj))

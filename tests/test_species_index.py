"""Tests for dense species index and post-step exchange."""

from pathlib import Path

import numpy as np
import pytest

from rover.species_index import (
    SpeciesIndex,
    build_species_index,
    exchange_counts,
    local_to_global,
)

DATA = Path(__file__).resolve().parent / "data" / "LR"
DET = DATA / "deterministic-interactions.xml"
STOCH = DATA / "stochastic-gene-expression.xml"

pytestmark = pytest.mark.skipif(
    not DET.exists() or not STOCH.exists(),
    reason="LR fixtures not present",
)


def test_species_index_union_stable_order():
    index = build_species_index(STOCH, DET, deterministic_sbml=DET, stochastic_sbml=STOCH)
    assert index.n_species == 9
    assert "cyt_mrna__LIGAND_" in index.name_to_index
    assert "cyt_prot__LIGAND_" in index.name_to_index
    assert index.names[0] == "cyt_prot__LIGAND_"


def test_membership_masks_allow_overlap():
    index = build_species_index(
        STOCH, DET, deterministic_sbml=DET, stochastic_sbml=STOCH
    )
    # LR partitions share all 9 species ids
    assert int(index.overlap_mask.sum()) == 9
    assert int(index.deterministic_only_mask.sum()) == 0
    assert int(index.stochastic_only_mask.sum()) == 0
    assert np.all(index.deterministic_mask)
    assert np.all(index.stochastic_mask)


def test_exchange_overlap_keeps_producer_delta():
    """Modifier-only module (Δ≈0) must not wipe the producer’s overlap update."""
    names = ("a", "b", "shared")
    index = SpeciesIndex(
        names=names,
        name_to_index={n: i for i, n in enumerate(names)},
        deterministic_mask=np.array([True, False, True]),
        stochastic_mask=np.array([False, True, True]),
    )
    s0 = np.array([10.0, 20.0, 5.0])
    # stoch evolves shared +5; bng leaves shared unchanged (modifier)
    stoch_local = np.array([25.0, 10.0])  # b, shared
    bng_local = np.array([12.0, 5.0])  # a, shared
    out = exchange_counts(
        s0,
        stoch_local=stoch_local,
        stoch_indices=np.array([1, 2]),
        bng_local=bng_local,
        bng_indices=np.array([0, 2]),
        index=index,
    )
    assert out[0] == pytest.approx(12.0)  # det-only from bng
    assert out[1] == pytest.approx(25.0)  # stoch-only from stoch
    assert out[2] == pytest.approx(10.0)  # 5 + 5 + 0


def test_exchange_clips_negative_counts():
    names = ("a", "b", "shared")
    index = SpeciesIndex(
        names=names,
        name_to_index={n: i for i, n in enumerate(names)},
        deterministic_mask=np.array([True, False, True]),
        stochastic_mask=np.array([False, True, True]),
    )
    s0 = np.array([1.0, 1.0, 1.0])
    out = exchange_counts(
        s0,
        stoch_local=np.array([-5.0, -2.0]),  # b, shared
        stoch_indices=np.array([1, 2]),
        bng_local=np.array([-3.0, 0.5]),  # a, shared
        bng_indices=np.array([0, 2]),
        index=index,
    )
    assert np.all(out >= 0.0)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)
    # overlap: 1 + (-2-1) + (0.5-1) = -2.5 → 0
    assert out[2] == pytest.approx(0.0)


def test_local_to_global():
    index = build_species_index(STOCH, DET, deterministic_sbml=DET, stochastic_sbml=STOCH)
    local = ["cyt_prot__LIGAND_", "cyt_mrna__LIGAND_"]
    idxs = local_to_global(local, index)
    assert idxs[0] == index.name_to_index["cyt_prot__LIGAND_"]
    assert idxs[1] == index.name_to_index["cyt_mrna__LIGAND_"]

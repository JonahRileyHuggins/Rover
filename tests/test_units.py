"""Unit conversion round-trips for nM ↔ molecule counts."""

from pathlib import Path

import numpy as np
import pytest

from rover.units import (
    counts_to_nM,
    molecules_to_nM,
    nM_to_counts,
    nM_to_molecules,
    nM_to_molecules_factors,
    sbml_initial_nM,
    species_volumes_L,
)

DATA = Path(__file__).resolve().parent / "data" / "LR"
DET = DATA / "deterministic-interactions.xml"
STOCH = DATA / "stochastic-gene-expression.xml"

pytestmark = pytest.mark.skipif(not DET.exists(), reason="LR fixtures not present")

CYTO_V = 5.25e-12
NUC_V = 1.75e-12


def test_scalar_nM_round_trip():
    # ~2 molecules in nucleus at 0.001898 nM
    counts = nM_to_counts(0.001898, NUC_V)
    assert counts == pytest.approx(2.0, rel=1e-3)
    assert counts_to_nM(counts, NUC_V) == pytest.approx(0.001898, rel=1e-3)

    counts_cyto = nM_to_counts(0.001582, CYTO_V)
    assert counts_cyto == pytest.approx(5.0, rel=1e-3)


def test_vector_nM_molecule_round_trip():
    names, volumes = species_volumes_L(STOCH)
    factors = nM_to_molecules_factors(volumes)
    nM = np.asarray([sbml_initial_nM(STOCH)[n] for n in names], dtype=np.float64)
    mol = nM_to_molecules(nM, factors)
    back = molecules_to_nM(mol, factors)
    np.testing.assert_allclose(back, nM, rtol=1e-12, atol=1e-18)


def test_lr_sbml_initial_nM():
    det = sbml_initial_nM(DET)
    stoch = sbml_initial_nM(STOCH)
    assert det["nuc_gene_a__LIGAND_"] == pytest.approx(0.001898, rel=1e-6)
    assert stoch["cyt_mrna__LIGAND_"] == pytest.approx(0.0172528, rel=1e-6)

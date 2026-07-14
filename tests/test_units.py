"""Unit conversion round-trips for nM ↔ molecule counts."""

from pathlib import Path

import bngsim
import numpy as np
import pytest

from rover.units import (
    AVOGADRO,
    bngsim_storage_from_counts,
    counts_from_bngsim_storage,
    counts_to_nM,
    nM_to_counts,
)

DATA = Path(__file__).resolve().parent / "data" / "LR"
DET = DATA / "deterministic-interactions.xml"

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


def test_bngsim_storage_round_trip():
    model = bngsim.Model.from_sbml(str(DET))
    kernel = bngsim.ReactionKernel(model, method="ode")
    uc = bngsim.UnitConverter.from_model(model)

    storage0 = kernel.get_state()
    counts = counts_from_bngsim_storage(storage0, uc)
    # Gene / mRNA ICs should map to ~2 and ~5 molecules
    names = list(kernel.state_names)
    gene_i = names.index("nuc_gene_a__LIGAND_")
    mrna_i = names.index("cyt_mrna__LIGAND_")
    assert counts[gene_i] == pytest.approx(2.0, rel=1e-3)
    assert counts[mrna_i] == pytest.approx(5.0, rel=1e-3)

    storage1 = bngsim_storage_from_counts(counts, uc)
    np.testing.assert_allclose(storage1, storage0, rtol=1e-6, atol=1e-12)


def test_lr_stochmod_scales_stay_molecule_counts_with_companion():
    from rover.units import stochmod_to_molecule_scales

    stoch = DATA / "stochastic-gene-expression.xml"
    scales = stochmod_to_molecule_scales(stoch, companion_deterministic_sbml=DET)
    assert scales.shape[0] == 9
    assert np.all(scales == 1.0)

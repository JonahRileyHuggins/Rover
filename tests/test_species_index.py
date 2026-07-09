"""Tests for dense species index and ownership masks."""

from pathlib import Path

import numpy as np
import pytest

from rover.species_index import build_species_index, local_to_global, scatter_delta

DATA = Path(__file__).resolve().parent / "data"
DET = DATA / "deterministic-interactions.xml"
STOCH = DATA / "stochastic-gene-expression.xml"


def test_species_index_union_stable_order():
    index = build_species_index(STOCH, DET, ownership_sbml=STOCH)
    assert index.n_species == 9
    assert "cyt_mrna__LIGAND_" in index.name_to_index
    assert "cyt_prot__LIGAND_" in index.name_to_index
    # First file order wins for shared IDs
    assert index.names[0] == "cyt_prot__LIGAND_"


def test_ownership_masks_partition():
    index = build_species_index(STOCH, DET, ownership_sbml=STOCH)
    # Every species owned by exactly one process
    assert np.all(index.deterministic_mask | index.stochastic_mask)
    assert not np.any(index.deterministic_mask & index.stochastic_mask)

    prot_i = index.name_to_index["cyt_prot__LIGAND_"]
    mrna_i = index.name_to_index["cyt_mrna__LIGAND_"]
    gene_i = index.name_to_index["nuc_gene_a__LIGAND_"]
    assert index.deterministic_mask[prot_i]
    assert index.stochastic_mask[mrna_i]
    assert index.stochastic_mask[gene_i]


def test_local_to_global_and_scatter():
    index = build_species_index(STOCH, DET, ownership_sbml=STOCH)
    local = ["cyt_prot__LIGAND_", "cyt_mrna__LIGAND_"]
    idxs = local_to_global(local, index)
    delta = scatter_delta(
        np.array([1.0, 2.0]),
        idxs,
        index.n_species,
        ownership_mask=index.deterministic_mask,
    )
    assert delta[index.name_to_index["cyt_prot__LIGAND_"]] == 1.0
    # mRNA delta masked out by deterministic ownership
    assert delta[index.name_to_index["cyt_mrna__LIGAND_"]] == 0.0

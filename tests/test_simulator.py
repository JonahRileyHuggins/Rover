"""Tests for HybridSimulator update / run / reset API."""

from pathlib import Path

import numpy as np
import pytest

from rover import HybridSimulator

DATA = Path(__file__).resolve().parent / "data"
DET = DATA / "deterministic-interactions.xml"
STOCH = DATA / "stochastic-gene-expression.xml"


@pytest.fixture(scope="module")
def sim() -> HybridSimulator:
    return HybridSimulator(DET, STOCH, dt=1.0)


def test_update_species_by_name(sim: HybridSimulator):
    sim.reset()
    sim.update("cyt_mrna__LIGAND_", 42.0)
    assert sim.get("cyt_mrna__LIGAND_") == pytest.approx(42.0)
    assert sim.counts[sim.index.name_to_index["cyt_mrna__LIGAND_"]] == 42.0


def test_update_bngsim_parameter(sim: HybridSimulator):
    sim.reset()
    sim.update("kTL1_1", 3.5)
    assert sim.get("kTL1_1") == pytest.approx(3.5)
    assert sim._bng.kernel.model.get_param("kTL1_1") == pytest.approx(3.5)


def test_update_stochmod_parameter(sim: HybridSimulator):
    sim.reset()
    sim.update("kTC1_1", 0.02)
    assert sim.get("kTC1_1") == pytest.approx(0.02)


def test_update_mapping_and_kwargs(sim: HybridSimulator):
    sim.reset()
    sim.update({"cyt_prot__LIGAND_": 7.0, "kTL1_1": 1.5})
    sim.update(cyt_mrna__RECEPTOR_=9.0)
    assert sim.get("cyt_prot__LIGAND_") == pytest.approx(7.0)
    assert sim.get("kTL1_1") == pytest.approx(1.5)
    assert sim.get("cyt_mrna__RECEPTOR_") == pytest.approx(9.0)


def test_update_unknown_raises(sim: HybridSimulator):
    with pytest.raises(KeyError, match="Unknown"):
        sim.update("not_a_real_id", 1.0)


def test_run_after_update_and_dataframe(sim: HybridSimulator):
    sim.reset()
    sim.update("cyt_mrna__LIGAND_", 20.0)
    counts = sim.run(t_end=10.0)
    assert counts.shape == (sim.index.n_species,)
    assert sim.time == pytest.approx(10.0)
    # Translation should produce protein from elevated mRNA
    prot = sim.get("cyt_prot__LIGAND_")
    assert prot > 0.0

    df = sim.to_dataframe()
    assert list(df.columns) == list(sim.species_names)
    assert df.shape == (1, sim.index.n_species)
    assert float(df["cyt_prot__LIGAND_"].iloc[0]) == pytest.approx(prot)


def test_reset_restores_state(sim: HybridSimulator):
    sim.reset()
    mrna0 = sim.get("cyt_mrna__LIGAND_")
    k0 = sim.get("kTL1_1")
    sim.update(cyt_mrna__LIGAND_=99.0, kTL1_1=9.0)
    sim.run(t_end=5.0)
    sim.reset()
    assert sim.time == 0.0
    assert sim.get("cyt_mrna__LIGAND_") == pytest.approx(mrna0)
    assert sim.get("kTL1_1") == pytest.approx(k0)


def test_set_counts_vector(sim: HybridSimulator):
    sim.reset()
    vec = sim.counts
    vec[:] = 0.0
    vec[sim.index.name_to_index["nuc_gene_a__LIGAND_"]] = 3.0
    sim.set_counts(vec)
    assert sim.get("nuc_gene_a__LIGAND_") == pytest.approx(3.0)

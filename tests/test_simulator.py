"""Tests for HybridSimulator update / run / reset API."""

from pathlib import Path

import numpy as np
import pytest

from rover import HybridSimulator

DATA = Path(__file__).resolve().parent / "data"
_LR = DATA / "LR"
_SPARCED = DATA / "SPARCED"
DET = (
    _LR / "deterministic-interactions.xml"
    if (_LR / "deterministic-interactions.xml").exists()
    else _SPARCED / "deterministic-interactions.xml"
)
STOCH = (
    _LR / "stochastic-gene-expression.xml"
    if (_LR / "stochastic-gene-expression.xml").exists()
    else _SPARCED / "stochastic-gene-expression.xml"
)

pytestmark = pytest.mark.skipif(
    not DET.exists() or not STOCH.exists(),
    reason="hybrid SBML fixtures not present",
)


@pytest.fixture(scope="module")
def sim() -> HybridSimulator:
    return HybridSimulator(
        DET,
        STOCH,
        dt=1.0,
        bngsim_kwargs={"codegen": False, "jacobian": "fd"},
    )


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


def test_run_records_full_trajectory(sim: HybridSimulator):
    sim.reset()
    sim.update("cyt_mrna__LIGAND_", 20.0)
    traj = sim.run(t_end=10.0)
    assert traj.ndim == 2
    assert traj.shape == (11, sim.index.n_species)  # t=0..10 inclusive
    assert sim.times is not None
    assert sim.times.shape == (11,)
    np.testing.assert_allclose(sim.times, np.arange(0.0, 11.0, 1.0))
    np.testing.assert_allclose(sim.counts, traj[-1])
    assert sim.time == pytest.approx(10.0)
    assert sim.get("cyt_prot__LIGAND_") > 0.0

    df = sim.to_dataframe()
    assert "time" in df.columns
    assert list(df.columns[1:]) == list(sim.species_names)
    assert df.shape == (11, sim.index.n_species + 1)
    assert float(df["cyt_prot__LIGAND_"].iloc[-1]) == pytest.approx(
        sim.get("cyt_prot__LIGAND_")
    )


def test_bngsim_clock_stays_local_per_coupling_step(sim: HybridSimulator):
    """Each BNGsim advance is a local [0, dt] window, not accumulating wall time."""
    sim.reset()
    dt = float(sim.dt)
    n_steps = 10
    traj = sim.run(t_end=n_steps * dt, dt=dt)
    assert traj.shape == (n_steps + 1, sim.index.n_species)
    assert sim.time == pytest.approx(n_steps * dt)
    assert sim._bng.kernel.time == pytest.approx(dt, rel=1e-9, abs=1e-12)
    assert np.all(np.isfinite(traj))


def test_run_memmap_trajectory(sim: HybridSimulator, tmp_path: Path):
    sim.reset()
    out = tmp_path / "traj.npy"
    traj = sim.run(t_end=5.0, results_path=out, results_backend="memmap")
    assert out.exists()
    assert (tmp_path / "traj_times.npy").exists()
    assert isinstance(traj, np.memmap)
    assert traj.shape == (6, sim.index.n_species)
    loaded = np.load(out)
    np.testing.assert_allclose(loaded, traj)


def test_run_record_false_returns_final_vector(sim: HybridSimulator):
    sim.reset()
    final = sim.run(t_end=5.0, record=False)
    assert final.ndim == 1
    assert final.shape == (sim.index.n_species,)


def test_reset_restores_state(sim: HybridSimulator):
    sim.reset()
    mrna0 = sim.get("cyt_mrna__LIGAND_")
    k0 = sim.get("kTL1_1")
    sim.update(cyt_mrna__LIGAND_=99.0, kTL1_1=9.0)
    sim.run(t_end=5.0)
    sim.reset()
    assert sim.time == 0.0
    assert sim.results is None
    assert sim.get("cyt_mrna__LIGAND_") == pytest.approx(mrna0)
    assert sim.get("kTL1_1") == pytest.approx(k0)


def test_set_counts_vector(sim: HybridSimulator):
    sim.reset()
    vec = sim.counts
    vec[:] = 0.0
    vec[sim.index.name_to_index["nuc_gene_a__LIGAND_"]] = 3.0
    sim.set_counts(vec)
    assert sim.get("nuc_gene_a__LIGAND_") == pytest.approx(3.0)

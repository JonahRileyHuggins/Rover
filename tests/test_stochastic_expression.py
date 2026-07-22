"""SPARCED stochastic-expression seed / short-run checks (nM shared store)."""

from pathlib import Path

import numpy as np
import pytest

from rover import HybridSimulator

DATA = Path(__file__).resolve().parent / "data" / "SPARCED"
DET = DATA / "deterministic-interactions-v1.4.2.xml"
STOCH = DATA / "stochastic-gene-expression-v1.4.2.xml"

pytestmark = pytest.mark.skipif(
    not DET.exists() or not STOCH.exists(),
    reason="SPARCED v1.4.2 fixtures not present",
)


def test_stochastic_expression_seed_is_nanomolar():
    sim = HybridSimulator(
        DET,
        STOCH,
        dt=30.0,
        bngsim_kwargs={"codegen": False, "jacobian": "fd"},
    )
    assert sim.get("cyt_mrna__MAPK1_201_") == pytest.approx(0.001897488, abs=1e-12)
    assert sim.get("nuc_gene_a__MAPK1_") == pytest.approx(0.001897488, abs=1e-12)
    assert sim.get("nuc_gene_a__RB1_") == pytest.approx(0.001897488, abs=1e-12)
    # Gene totals stay at true nM scale (not ~2107 molecule inflation)
    a = sim.get("nuc_gene_a__RB1_")
    i = sim.get("nuc_gene_i__RB1_")
    assert a + i == pytest.approx(0.001897488, abs=1e-12)


def test_stochastic_expression_short_hybrid_finite():
    sim = HybridSimulator(
        DET,
        STOCH,
        dt=30.0,
        bngsim_kwargs={"codegen": False, "jacobian": "fd"},
    )
    traj = sim.run(t_end=90.0, dt=30.0)
    assert traj.shape[0] == 4
    assert np.all(np.isfinite(traj))
    assert np.all(traj >= 0.0)
    assert sim.get("nuc_gene_a__RB1_") + sim.get("nuc_gene_i__RB1_") == pytest.approx(
        0.001897488, rel=1e-6, abs=1e-12
    )

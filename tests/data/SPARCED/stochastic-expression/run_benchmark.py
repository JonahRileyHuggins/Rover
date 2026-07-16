"""Run SPARCED stochastic-expression Fig2B observables via Rover HybridSimulator."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from rover import HybridSimulator

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
DET = DATA / "deterministic-interactions-v1.4.2.xml"
STOCH = DATA / "stochastic-gene-expression-v1.4.2.xml"
CYTO_V = 5.25e-12
N_A = 6.022e23
NM_TO_MOL_CYTO = N_A * 1e-9 * CYTO_V

# v1.4.2 uses ERK/MEK naming (observables.tsv historically said MAPK/MAP2K).
ERK_SPECIES = (
    "cyt_cong__p_p_ERK__B2L11_1_",
    "cyt_cong__p_p_ERK__BAD_",
    "cyt_cong__EGFR_1__p_p_ERK_",
    "cyt_cong__ERBB2_1__p_p_ERK_",
    "cyt_cong__ERBB4_1__p_p_ERK_",
    "nuc_cong__ERK_",
    "nuc_cong__p_p_ERK_",
    "cyt_cong__ERK_",
    "cyt_cong__p_ERK_",
    "cyt_cong__p_p_ERK_",
    "cyt_cong__GRB2__SOS1_1__p_p_ERK_",
    "cyt_cong__RAF1_1__p_p_ERK_",
    "cyt_cong__NF1_1__p_p_ERK_",
    "cyt_cong__ERK__p_p_MEK_",
    "cyt_cong__p_ERK__p_p_MEK_",
    "cyt_cong__RPS6KA__p_p_ERK_",
    "nuc_cong__p_p_ERK__DUS1_",
    "nuc_cong__FOS_1__p_p_ERK_",
    "cyt_cong__p_ERK__DUS6_1_",
    "cyt_cong__p_p_ERK__DUS6_1_",
    "cyt_cong__p_p_ERK__p_DUS1_",
)


def fig2b_observables(sim: HybridSimulator) -> dict[str, float]:
    g = sim.get
    erk = sum(float(g(sid)) for sid in ERK_SPECIES) * NM_TO_MOL_CYTO
    return {
        "MAPK1ac": float(g("nuc_gene_a__MAPK1_")),
        "MAPK3ac": float(g("nuc_gene_a__MAPK3_")),
        "MAPK1_mRNA": float(g("cyt_mrna__MAPK1_201_")),
        "MAPK3_mRNA": float(g("cyt_mrna__MAPK3_201_")),
        "ERK_total": erk,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sim = HybridSimulator(DET, STOCH, dt=30.0)
    t0 = fig2b_observables(sim)
    print("t=0 Fig2B:", t0, flush=True)
    for name, val in t0.items():
        if name != "ERK_total":
            assert val < 10.0, f"{name} unexpectedly large for nM store: {val}"

    sim.run(t_end=86400.0, dt=30.0, results_path=HERE / "out_traj.npy")
    final = fig2b_observables(sim)
    print("t=86400 Fig2B:", final, flush=True)
    assert all(np.isfinite(v) and v >= 0.0 for v in final.values())
    for name in ("MAPK1ac", "MAPK3ac", "MAPK1_mRNA", "MAPK3_mRNA"):
        assert final[name] < 10.0, f"{name} not nM-scale: {final[name]}"


if __name__ == "__main__":
    main()

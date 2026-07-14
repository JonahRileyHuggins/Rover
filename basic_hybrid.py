"""Minimal entry point for the hybrid BNGsim + StochMod engine."""

import logging
from pathlib import Path

from rover import HybridSimulator

DATA = Path(__file__).resolve().parent / "tests" / "data"
# Prefer small LR fixtures; fall back to SPARCED if LR is absent.
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sim = HybridSimulator(DET, STOCH, dt=1.0)
    traj = sim.run(t_end=60.0)
    print("trajectory shape:", traj.shape)
    print("species:", sim.species_names)
    print(sim.to_dataframe().tail())


if __name__ == "__main__":
    main()

"""Minimal entry point for the hybrid process-bigraph engine."""

import logging
from pathlib import Path

from rover import HybridSimulator

DATA = Path(__file__).resolve().parent / "tests" / "data"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sim = HybridSimulator(
        DATA / "deterministic-interactions.xml",
        DATA / "stochastic-gene-expression.xml",
        dt=1.0,
    )
    counts = sim.run(t_end=60.0)
    print("species:", sim.species_names)
    print("final counts:", counts)
    print(sim.to_dataframe())


if __name__ == "__main__":
    main()

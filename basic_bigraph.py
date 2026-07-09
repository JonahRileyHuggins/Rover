"""Minimal entry point for the hybrid process-bigraph engine."""

from pathlib import Path

from rover.composite import run_hybrid

DATA = Path(__file__).resolve().parent / "tests" / "data"


def main() -> None:
    counts, index = run_hybrid(
        DATA / "deterministic-interactions.xml",
        DATA / "stochastic-gene-expression.xml",
        t_end=10.0,
        dt=1.0,
    )
    print("species:", index.names)
    print("final counts:", counts)


if __name__ == "__main__":
    main()

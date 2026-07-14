"""StochMod tau-leap module: advance one leap from a shared count snapshot."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("rover.stochmod")


class StochModModule:
    """Advance StochMod one ``dt`` from shared molecule counts; return locals.

    StochMod's compartment normalizer rewrites kinetic laws for **count-based**
    propensities, so the runtime state is molecule counts (same currency as the
    shared store). Companion nM→molecule scales are only for reading SBML ICs
    in the engine — not for per-step bridging.

    Does not mutate the global count vector — the orchestrator exchanges results.
    """

    def __init__(
        self,
        *,
        module: Any,
        sbml_path: str | Path,
        local_indices: list[int] | np.ndarray,
        n_species: int,
        companion_deterministic_sbml: str | Path | None = None,
    ) -> None:
        del sbml_path, companion_deterministic_sbml  # IC scaling is engine-side
        self._module = module
        self._local_indices = np.asarray(local_indices, dtype=np.int64)
        self._n_species = int(n_species)
        self._local_names = list(self._module.species_names)
        if len(self._local_indices) != len(self._local_names):
            raise ValueError(
                f"local_indices length {len(self._local_indices)} != "
                f"module species {len(self._local_names)}"
            )
        logger.info(
            "StochMod module ready (%d species; count-based state)",
            len(self._local_names),
        )

    def advance_from(self, counts: np.ndarray, dt: float) -> np.ndarray:
        """Advance one tau-leap of size ``dt``; return post-step local counts."""
        local_counts = np.asarray(counts[self._local_indices], dtype=np.float64).copy()
        self._module.set_state(local_counts)
        return np.asarray(self._module.advance(float(dt)), dtype=np.float64)

    @property
    def local_indices(self) -> np.ndarray:
        return self._local_indices

    @property
    def module(self):
        return self._module

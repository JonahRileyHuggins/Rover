"""StochMod tau-leap module: convert shared nM ↔ molecules, advance one leap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from rover.units import (
    molecules_to_nM,
    nM_to_molecules,
    nM_to_molecules_factors,
    species_volumes_L,
)

logger = logging.getLogger("rover.stochmod")


class StochModModule:
    """Advance StochMod one ``dt`` from shared nanomolar state; return local nM.

    StochMod's compartment normalizer rewrites kinetic laws for **count-based**
    propensities, so the runtime state is molecule counts. The shared store is
    nM; this module converts at the boundary (SingleCell-shaped).

    Does not mutate the global vector — the orchestrator exchanges results.
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
        del companion_deterministic_sbml  # unused; shared currency is nM
        self._module = module
        self._local_indices = np.asarray(local_indices, dtype=np.int64)
        self._n_species = int(n_species)
        self._local_names = list(self._module.species_names)
        if len(self._local_indices) != len(self._local_names):
            raise ValueError(
                f"local_indices length {len(self._local_indices)} != "
                f"module species {len(self._local_names)}"
            )

        sbml_names, volumes = species_volumes_L(sbml_path)
        name_to_vol = dict(zip(sbml_names, volumes, strict=True))
        local_volumes = np.asarray(
            [name_to_vol[n] for n in self._local_names], dtype=np.float64
        )
        self._nM_to_mol = nM_to_molecules_factors(local_volumes)

        logger.info(
            "StochMod module ready (%d species; nM↔molecules bridge)",
            len(self._local_names),
        )
        self.last_integrate_s = 0.0
        self.last_bridge_s = 0.0

    def advance_from(self, counts: np.ndarray, dt: float) -> np.ndarray:
        """Advance one tau-leap of size ``dt``; return post-step local nM.

        ``last_integrate_s`` / ``last_bridge_s`` split the leap from Rover
        gather/convert/set_state overhead for progress logs.
        """
        import time

        t0 = time.perf_counter()
        local_nM = np.asarray(counts[self._local_indices], dtype=np.float64)
        molecules = nM_to_molecules(local_nM, self._nM_to_mol)
        self._module.set_state(molecules)
        t1 = time.perf_counter()
        out_mol = np.asarray(self._module.advance(float(dt)), dtype=np.float64)
        t2 = time.perf_counter()
        out_nM = molecules_to_nM(out_mol, self._nM_to_mol)
        t3 = time.perf_counter()
        self.last_bridge_s = (t1 - t0) + (t3 - t2)
        self.last_integrate_s = t2 - t1
        return out_nM

    @property
    def local_indices(self) -> np.ndarray:
        return self._local_indices

    @property
    def module(self):
        return self._module

"""process-bigraph Process wrapping StochMod StochasticModule."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from process_bigraph import Process

from rover.species_index import gather, scatter_delta
from rover.units import stochmod_to_molecule_scales

logger = logging.getLogger("rover.stochmod")


class StochModProcess(Process):
    """Advance a stochastic SBML model via StochMod; exchange molecule counts.

    Config
    ------
    sbml_path : str
        Path to the stochastic SBML file (used for unit scales / load).
    module : object, optional
        Pre-built ``stochmod.StochasticModule``.
    local_indices : list[int]
        Global dense indices for this module's ``species_names`` order.
    ownership_mask : list[bool] | None
        Global mask of species this process may write.
    n_species : int
        Global store length.
    time_step : float
        Default coupling interval (informational).
    """

    config_schema = {
        "sbml_path": {"_type": "string", "_default": ""},
        "module": {"_type": "maybe[node]", "_default": None},
        "local_indices": "list[integer]",
        "ownership_mask": {"_type": "maybe[list[boolean]]", "_default": None},
        "n_species": "integer",
        "time_step": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        from stochmod import StochasticModule

        path = self.config.get("sbml_path") or ""
        module = self.config.get("module")
        if module is None:
            if not path:
                raise ValueError("StochModProcess requires 'module' or 'sbml_path'")
            module = StochasticModule(path)
            logger.info(
                "StochMod loaded %s (%d species)",
                Path(path).name,
                len(module.species_names),
            )
        else:
            logger.info(
                "StochMod using pre-built module (%d species)",
                len(module.species_names),
            )

        self._module = module
        self._local_indices = np.asarray(self.config["local_indices"], dtype=np.int64)
        self._n_species = int(self.config["n_species"])
        mask = self.config.get("ownership_mask")
        self._ownership_mask = (
            np.asarray(mask, dtype=bool) if mask is not None else None
        )
        self._owned_local = None
        if self._ownership_mask is not None:
            self._owned_local = self._ownership_mask[self._local_indices]
        self._local_names = list(self._module.species_names)
        if len(self._local_indices) != len(self._local_names):
            raise ValueError(
                f"local_indices length {len(self._local_indices)} != "
                f"module species {len(self._local_names)}"
            )

        if not path:
            raise ValueError("StochModProcess requires sbml_path for unit scales")
        scales = stochmod_to_molecule_scales(path)
        if scales.shape[0] != len(self._local_names):
            raise ValueError(
                f"StochMod unit scales length {scales.shape[0]} != "
                f"module species {len(self._local_names)}"
            )
        self._to_molecules = scales
        self._to_stochmod = np.where(scales != 0.0, 1.0 / scales, 0.0)
        n_scaled = int(np.sum(scales != 1.0))
        if n_scaled:
            logger.info(
                "StochMod unit bridge: %d/%d species scaled nanomole→molecule",
                n_scaled,
                len(scales),
            )

    def inputs(self) -> dict[str, Any]:
        return {
            "counts": {
                "_type": "array",
                "_shape": (self._n_species,),
                "_data": "float64",
            }
        }

    def outputs(self) -> dict[str, Any]:
        return {
            "counts": {
                "_type": "array",
                "_shape": (self._n_species,),
                "_data": "float64",
            }
        }

    def update(self, state: dict[str, Any], interval: float) -> dict[str, Any]:
        global_counts = np.asarray(state["counts"], dtype=np.float64)
        local_counts = gather(global_counts, self._local_indices)
        stoch_state = local_counts * self._to_stochmod
        self._module.set_state(stoch_state)
        new_stoch = np.asarray(self._module.advance(float(interval)), dtype=np.float64)
        new_counts = new_stoch * self._to_molecules
        local_delta = new_counts - local_counts
        if self._owned_local is not None:
            local_delta = np.where(self._owned_local, local_delta, 0.0)
        delta = scatter_delta(
            local_delta,
            self._local_indices,
            self._n_species,
            ownership_mask=None,
        )
        return {"counts": delta}

    def apply_inplace(self, counts: np.ndarray, interval: float) -> None:
        """Operator-split step that mutates ``counts`` in place (lean runner)."""
        local_counts = counts[self._local_indices].copy()
        stoch_state = local_counts * self._to_stochmod
        self._module.set_state(stoch_state)
        new_stoch = np.asarray(self._module.advance(float(interval)), dtype=np.float64)
        new_counts = new_stoch * self._to_molecules
        local_delta = new_counts - local_counts
        if self._owned_local is not None:
            local_delta = np.where(self._owned_local, local_delta, 0.0)
        counts[self._local_indices] += local_delta

    @property
    def module(self):
        return self._module

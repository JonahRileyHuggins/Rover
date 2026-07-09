"""process-bigraph Process wrapping StochMod StochasticModule."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from process_bigraph import Process

from rover.species_index import gather, scatter_delta


class StochModProcess(Process):
    """Advance a stochastic SBML model via StochMod; exchange molecule counts.

    Config
    ------
    sbml_path : str
        Path to the stochastic SBML file.
    local_indices : list[int]
        Global dense indices for this module's ``species_names`` order.
    ownership_mask : list[bool] | None
        Global mask of species this process may write (genes + mRNA).
    n_species : int
        Global store length.
    time_step : float
        Default coupling interval (informational).
    """

    config_schema = {
        "sbml_path": "string",
        "local_indices": "list[integer]",
        "ownership_mask": {"_type": "maybe[list[boolean]]", "_default": None},
        "n_species": "integer",
        "time_step": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        from stochmod import StochasticModule

        path = Path(self.config["sbml_path"])
        self._module = StochasticModule(path)
        self._local_indices = np.asarray(self.config["local_indices"], dtype=np.int64)
        self._n_species = int(self.config["n_species"])
        mask = self.config.get("ownership_mask")
        self._ownership_mask = (
            np.asarray(mask, dtype=bool) if mask is not None else None
        )
        self._local_names = list(self._module.species_names)
        if len(self._local_indices) != len(self._local_names):
            raise ValueError(
                f"local_indices length {len(self._local_indices)} != "
                f"module species {len(self._local_names)}"
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
        self._module.set_state(local_counts)
        new_counts = self._module.advance(float(interval))
        local_delta = np.asarray(new_counts, dtype=np.float64) - local_counts
        delta = scatter_delta(
            local_delta,
            self._local_indices,
            self._n_species,
            ownership_mask=self._ownership_mask,
        )
        return {"counts": delta}

    @property
    def module(self):
        return self._module

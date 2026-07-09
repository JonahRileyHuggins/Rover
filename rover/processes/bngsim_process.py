"""process-bigraph Process wrapping BNGsim ReactionKernel (ODE)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from process_bigraph import Process

from rover.species_index import gather, scatter_delta
from rover.units import bngsim_storage_from_counts, counts_from_bngsim_storage


class BngsimProcess(Process):
    """Advance a deterministic SBML model via BNGsim; exchange molecule counts.

    Config
    ------
    sbml_path : str
        Path to the deterministic SBML file.
    local_indices : list[int]
        Global dense indices for this kernel's ``state_names`` order.
    ownership_mask : list[bool] | None
        Global mask of species this process may write (proteins).
    n_species : int
        Global store length.
    method : str
        BNGsim method (default ``ode``).
    simulator_kwargs : dict
        Forwarded to ``ReactionKernel`` / ``Simulator``.
    time_step : float
        Default coupling interval (informational; Composite supplies interval).
    """

    config_schema = {
        "sbml_path": "string",
        "local_indices": "list[integer]",
        "ownership_mask": {"_type": "maybe[list[boolean]]", "_default": None},
        "n_species": "integer",
        "method": {"_type": "string", "_default": "ode"},
        "simulator_kwargs": {"_type": "node", "_default": {}},
        "time_step": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        import bngsim
        from bngsim.kernel import ReactionKernel

        path = Path(self.config["sbml_path"])
        self._model = bngsim.Model.from_sbml(str(path))
        sim_kwargs = dict(self.config.get("simulator_kwargs") or {})
        self._kernel = ReactionKernel(
            self._model,
            method=self.config.get("method", "ode"),
            **sim_kwargs,
        )
        self._uc = bngsim.UnitConverter.from_model(self._model)
        self._local_indices = np.asarray(self.config["local_indices"], dtype=np.int64)
        self._n_species = int(self.config["n_species"])
        mask = self.config.get("ownership_mask")
        self._ownership_mask = (
            np.asarray(mask, dtype=bool) if mask is not None else None
        )
        self._local_names = list(self._kernel.state_names)
        if len(self._local_indices) != len(self._local_names):
            raise ValueError(
                f"local_indices length {len(self._local_indices)} != "
                f"kernel species {len(self._local_names)}"
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
        storage = bngsim_storage_from_counts(local_counts, self._uc)
        self._kernel.set_state(storage)
        new_storage = self._kernel.advance(float(interval))
        new_counts = counts_from_bngsim_storage(new_storage, self._uc)
        local_delta = new_counts - local_counts
        delta = scatter_delta(
            local_delta,
            self._local_indices,
            self._n_species,
            ownership_mask=self._ownership_mask,
        )
        return {"counts": delta}

    @property
    def kernel(self):
        return self._kernel

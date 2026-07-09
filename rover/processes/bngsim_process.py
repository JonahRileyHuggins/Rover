"""process-bigraph Process wrapping BNGsim ReactionKernel (ODE)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from process_bigraph import Process

from rover.species_index import gather, scatter_delta
from rover.units import CountConverter

logger = logging.getLogger("rover.bngsim")


def _configure_bngsim_logging(*, level: int = logging.INFO) -> None:
    """Attach a handler to BNGsim's logger (codegen / load messages).

    After the kernel is built, callers typically drop this to WARNING so
    per-step ``run_until`` INFO lines do not dominate wall time / stdout.
    """
    bng_logger = logging.getLogger("bngsim")
    if not bng_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
        )
        bng_logger.addHandler(handler)
    bng_logger.setLevel(level)
    bng_logger.propagate = False


def quiet_bngsim_logging() -> None:
    """Suppress BNGsim per-step INFO spam after kernel construction."""
    logging.getLogger("bngsim").setLevel(logging.WARNING)


def build_reaction_kernel(
    sbml_path: str | Path,
    *,
    method: str = "ode",
    simulator_kwargs: dict[str, Any] | None = None,
):
    """Build a BNGsim ReactionKernel with codegen preferred and clear logging."""
    import bngsim
    from bngsim.kernel import ReactionKernel

    _configure_bngsim_logging()
    path = Path(sbml_path)
    model = bngsim.Model.from_sbml(str(path))
    kwargs = dict(simulator_kwargs or {})
    want_codegen = bool(kwargs.pop("codegen", True))

    kernel = None
    codegen_active = False
    if want_codegen:
        try:
            kernel = ReactionKernel(model, method=method, codegen=True, **kwargs)
            sim = kernel.simulator
            codegen_active = bool(
                getattr(sim, "_codegen_so_path", "")
                or getattr(sim, "_codegen_c_source", "")
            )
            if codegen_active:
                so = getattr(sim, "_codegen_so_path", "") or ""
                logger.info(
                    "BNGsim codegen active for %s (%s)",
                    path.name,
                    so or "JIT source",
                )
            else:
                logger.warning(
                    "BNGsim codegen requested for %s but no compiled RHS attached",
                    path.name,
                )
        except Exception as exc:
            logger.warning(
                "BNGsim codegen failed for %s (%s); falling back to interpreted RHS",
                path.name,
                exc,
            )
            kernel = None

    if kernel is None:
        kernel = ReactionKernel(model, method=method, codegen=False, **kwargs)
        logger.info(
            "BNGsim kernel ready for %s (method=%s, codegen=False, n_species=%d)",
            path.name,
            method,
            len(kernel.state_names),
        )

    # Kernel construction is done — drop BNGsim to WARNING so advance() does
    # not emit thousands of "Running ODE simulation" INFO lines.
    quiet_bngsim_logging()
    return kernel, codegen_active


class BngsimProcess(Process):
    """Advance a deterministic SBML model via BNGsim; exchange molecule counts.

    Config
    ------
    sbml_path : str
        Path to the deterministic SBML file (used if ``kernel`` is not given).
    kernel : object, optional
        Pre-built ``bngsim.ReactionKernel`` (avoids reloading SBML).
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
        "sbml_path": {"_type": "string", "_default": ""},
        "kernel": {"_type": "maybe[node]", "_default": None},
        "local_indices": "list[integer]",
        "ownership_mask": {"_type": "maybe[list[boolean]]", "_default": None},
        "n_species": "integer",
        "method": {"_type": "string", "_default": "ode"},
        "simulator_kwargs": {"_type": "node", "_default": {}},
        "time_step": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        import bngsim

        kernel = self.config.get("kernel")
        if kernel is None:
            path = self.config.get("sbml_path") or ""
            if not path:
                raise ValueError("BngsimProcess requires 'kernel' or 'sbml_path'")
            kernel, self.codegen_active = build_reaction_kernel(
                path,
                method=self.config.get("method", "ode"),
                simulator_kwargs=self.config.get("simulator_kwargs") or {},
            )
        else:
            sim = kernel.simulator
            self.codegen_active = bool(
                getattr(sim, "_codegen_so_path", "")
                or getattr(sim, "_codegen_c_source", "")
            )
            logger.info(
                "BNGsim using pre-built kernel (codegen=%s, n_species=%d)",
                self.codegen_active,
                len(kernel.state_names),
            )
            quiet_bngsim_logging()

        self._kernel = kernel
        self._uc = bngsim.UnitConverter.from_model(kernel.model)
        self._converter = CountConverter.from_unit_converter(self._uc)
        self._local_indices = np.asarray(self.config["local_indices"], dtype=np.int64)
        self._n_species = int(self.config["n_species"])
        mask = self.config.get("ownership_mask")
        self._ownership_mask = (
            np.asarray(mask, dtype=bool) if mask is not None else None
        )
        self._owned_local = None
        if self._ownership_mask is not None:
            # Local positions that this process is allowed to write.
            self._owned_local = self._ownership_mask[self._local_indices]
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
        storage = self._converter.storage_from_counts(local_counts)
        self._kernel.set_state(storage)
        new_storage = self._kernel.advance(float(interval))
        new_counts = self._converter.counts_from_storage(new_storage)
        local_delta = new_counts - local_counts
        if self._owned_local is not None:
            local_delta = np.where(self._owned_local, local_delta, 0.0)
        delta = scatter_delta(
            local_delta,
            self._local_indices,
            self._n_species,
            ownership_mask=None,  # already masked locally
        )
        return {"counts": delta}

    def apply_inplace(self, counts: np.ndarray, interval: float) -> None:
        """Operator-split step that mutates ``counts`` in place (lean runner)."""
        local_counts = counts[self._local_indices]
        storage = self._converter.storage_from_counts(local_counts)
        self._kernel.set_state(storage)
        new_storage = self._kernel.advance(float(interval))
        new_counts = self._converter.counts_from_storage(new_storage)
        local_delta = new_counts - local_counts
        if self._owned_local is not None:
            local_delta = np.where(self._owned_local, local_delta, 0.0)
        counts[self._local_indices] += local_delta

    @property
    def kernel(self):
        return self._kernel

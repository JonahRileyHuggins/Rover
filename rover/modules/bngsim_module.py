"""BNGsim ODE module: identity bridge from shared nM, advance one [0, dt] window."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

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
    """Build a BNGsim ReactionKernel with codegen + analytical Jacobian preferred.

    Falls back to interpreted RHS if codegen fails, and to ``jacobian="fd"``
    if the analytical Jacobian path fails at construction.
    """
    import bngsim
    from bngsim.kernel import ReactionKernel

    _configure_bngsim_logging()
    path = Path(sbml_path)
    model = bngsim.Model.from_sbml(str(path))
    kwargs = dict(simulator_kwargs or {})
    want_codegen = bool(kwargs.pop("codegen", True))
    rtol = float(kwargs.pop("rtol", 1e-4))
    atol = float(kwargs.pop("atol", 1e-6))
    max_steps = int(kwargs.pop("max_steps", 100_000_000))
    # Default analytical; callers may override (e.g. tests use "fd").
    jacobian = str(kwargs.pop("jacobian", "analytical"))

    def _make(codegen: bool, jac: str):
        return ReactionKernel(
            model,
            method=method,
            codegen=codegen,
            jacobian=jac,
            **kwargs,
        )

    kernel = None
    codegen_active = False
    used_jac = jacobian

    if want_codegen:
        try:
            kernel = _make(True, jacobian)
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
        try:
            kernel = _make(False, jacobian)
            logger.info(
                "BNGsim kernel ready for %s (method=%s, codegen=False, n_species=%d)",
                path.name,
                method,
                len(kernel.state_names),
            )
        except Exception as exc:
            if jacobian in ("analytical", "auto"):
                logger.warning(
                    "BNGsim jacobian=%s failed for %s (%s); falling back to fd",
                    jacobian,
                    path.name,
                    exc,
                )
                kernel = _make(False, "fd")
                used_jac = "fd"
                logger.info(
                    "BNGsim kernel ready for %s (method=%s, codegen=False, "
                    "jacobian=fd, n_species=%d)",
                    path.name,
                    method,
                    len(kernel.state_names),
                )
            else:
                raise

    # If analytical failed only under codegen, interpreted path already ran above.
    # Construction-time FD fallback is handled in the interpreted branch.

    sim = kernel.simulator
    if used_jac != jacobian:
        try:
            sim._jacobian = used_jac
        except Exception:
            pass
    sim._rtol = rtol
    sim._atol = atol
    sim._max_steps = max_steps
    logger.info(
        "BNGsim ODE opts: rtol=%g atol=%g max_steps=%d jacobian=%s",
        rtol,
        atol,
        max_steps,
        getattr(sim, "_jacobian", used_jac),
    )

    quiet_bngsim_logging()
    return kernel, codegen_active


class BngsimModule:
    """Advance BNGsim one local ``[0, dt]`` window from shared nanomolar state.

    Shared store and BNGsim storage are both nM — identity bridge (no convert).
    Does not mutate the global vector — the orchestrator exchanges results.
    Absolute trajectory time is owned by the orchestrator, not this module.
    """

    def __init__(
        self,
        *,
        kernel,
        local_indices: list[int] | np.ndarray,
        n_species: int,
        codegen_active: bool = False,
        stoch_owned_global: np.ndarray | None = None,
    ) -> None:
        self._kernel = kernel
        self.codegen_active = bool(codegen_active)
        self._local_names = list(self._kernel.state_names)
        self._local_indices = np.asarray(local_indices, dtype=np.int64)
        self._n_species = int(n_species)
        if len(self._local_indices) != len(self._local_names):
            raise ValueError(
                f"local_indices length {len(self._local_indices)} != "
                f"kernel species {len(self._local_names)}"
            )
        # Stoch-owned overlap (mRNA): hold fixed over the ODE window.
        self._freeze_mask = np.zeros(len(self._local_names), dtype=bool)
        if stoch_owned_global is not None:
            stoch_owned_global = np.asarray(stoch_owned_global, dtype=bool)
            for li, gi in enumerate(self._local_indices):
                if int(gi) < len(stoch_owned_global) and stoch_owned_global[int(gi)]:
                    self._freeze_mask[li] = True
        logger.info(
            "BNGsim module ready (codegen=%s, n_species=%d, identity nM bridge; "
            "%d stoch-owned mRNA frozen over ODE window)",
            self.codegen_active,
            len(self._local_names),
            int(self._freeze_mask.sum()),
        )
        quiet_bngsim_logging()
        self.last_integrate_s = 0.0
        self.last_bridge_s = 0.0

    def _rewind_clock(self) -> None:
        """Rewind BNGsim interactive clock to 0 without restoring SBML ICs."""
        self._kernel.simulator._current_time = 0.0
        self._kernel._last_result = None

    def advance_from(self, counts: np.ndarray, dt: float) -> np.ndarray:
        """Advance one local ``[0, dt]`` window; return post-step local nM.

        ``last_integrate_s`` / ``last_bridge_s`` attribute the CVODE call vs
        Rover set_state / rewind work for progress logs.
        """
        import time

        t0 = time.perf_counter()
        local_nM = np.asarray(counts[self._local_indices], dtype=np.float64)
        self._kernel.set_state(local_nM)
        self._rewind_clock()
        t1 = time.perf_counter()
        new_storage = self._kernel.advance(float(dt))
        t2 = time.perf_counter()
        out = np.asarray(new_storage, dtype=np.float64)
        if np.any(self._freeze_mask):
            out = out.copy()
            out[self._freeze_mask] = local_nM[self._freeze_mask]
        self.last_bridge_s = t1 - t0
        self.last_integrate_s = t2 - t1
        return out

    @property
    def local_indices(self) -> np.ndarray:
        return self._local_indices

    @property
    def kernel(self):
        return self._kernel

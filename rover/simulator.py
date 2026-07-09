"""HybridSimulator: load-once session for BNGsim + StochMod coupling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rover.composite import (
    build_hybrid_composite,
    run_operator_split,
    _ensure_rover_logging,
    _get_process_instance,
)

logger = logging.getLogger("rover")


class HybridSimulator:
    """Stateful hybrid engine: load models once, then ``update`` / ``run``.

    Parameters
    ----------
    deterministic_sbml, stochastic_sbml :
        Paths to the partitioned SBML models.
    dt :
        Default coupling step for ``run``.
    bngsim_kwargs :
        Forwarded to BNGsim (default ``{"codegen": True}``).
    initial_counts :
        Optional length-N molecule-count vector; otherwise seeded from SBMLs.

    Examples
    --------
    >>> sim = HybridSimulator(det_xml, stoch_xml, dt=1.0)
    >>> sim.update("cyt_mrna__LIGAND_", 10)
    >>> sim.update({"kTL1_1": 2.0, "kTC1_1": 0.01})
    >>> counts = sim.run(t_end=60.0)
    >>> df = sim.to_dataframe()
    """

    def __init__(
        self,
        deterministic_sbml: str | Path,
        stochastic_sbml: str | Path,
        *,
        dt: float = 1.0,
        bngsim_kwargs: dict[str, Any] | None = None,
        initial_counts: np.ndarray | None = None,
    ) -> None:
        _ensure_rover_logging()
        self.dt = float(dt)
        self.deterministic_sbml = Path(deterministic_sbml)
        self.stochastic_sbml = Path(stochastic_sbml)

        self.composite, self.index = build_hybrid_composite(
            self.deterministic_sbml,
            self.stochastic_sbml,
            dt=self.dt,
            initial_counts=initial_counts,
            bngsim_kwargs=bngsim_kwargs,
        )
        self._bng = _get_process_instance(self.composite, "bngsim")
        self._stoch = _get_process_instance(self.composite, "stochmod")
        self._counts = np.asarray(self.composite.state["counts"], dtype=np.float64).copy()
        self._t = 0.0

        # Snapshot for reset() + parameter membership sets
        self._initial_counts = self._counts.copy()
        self._bng_param_names = {str(n) for n in self._bng.kernel.model.param_names}
        self._stoch_param_names = {str(n) for n in self._stoch.module.parameter_ids}
        self._stoch_param_index = {
            n: i for i, n in enumerate(self._stoch.module.parameter_ids)
        }
        self._initial_bng_params = {
            n: float(self._bng.kernel.model.get_param(n)) for n in self._bng_param_names
        }
        self._initial_stoch_params = {
            n: float(self._stoch.module.parameter_values[i])
            for n, i in self._stoch_param_index.items()
        }

        logger.info(
            "HybridSimulator ready: %d species, %d bng params, %d stoch params",
            self.index.n_species,
            len(self._initial_bng_params),
            len(self._initial_stoch_params),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def species_names(self) -> tuple[str, ...]:
        return self.index.names

    @property
    def counts(self) -> np.ndarray:
        """Live shared molecule-count vector (copy)."""
        return self._counts.copy()

    @property
    def time(self) -> float:
        return self._t

    @property
    def parameter_names(self) -> list[str]:
        """Union of BNGsim + StochMod parameter ids (BNGsim first)."""
        seen: list[str] = []
        seen_set: set[str] = set()
        for name in list(self._initial_bng_params) + list(self._initial_stoch_params):
            if name not in seen_set:
                seen.append(name)
                seen_set.add(name)
        return seen

    def get(self, name: str) -> float:
        """Return a species count or parameter value by id.

        If a parameter exists in both models, the BNGsim value is returned.
        """
        if name in self.index.name_to_index:
            return float(self._counts[self.index.name_to_index[name]])
        if name in self._bng_param_names:
            return float(self._bng.kernel.model.get_param(name))
        if name in self._stoch_param_names:
            return float(
                self._stoch.module.parameter_values[self._stoch_param_index[name]]
            )
        raise KeyError(f"Unknown species or parameter id: {name!r}")

    def to_dataframe(self):
        """One-row pandas DataFrame of current counts (columns = species ids)."""
        import pandas as pd

        return pd.DataFrame([self._counts], columns=list(self.index.names))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(
        self,
        key: str | Mapping[str, float] | None = None,
        value: float | None = None,
        /,
        **kwargs: float,
    ) -> None:
        """Update species counts and/or parameters by id.

        Forms::

            sim.update("cyt_mrna__LIGAND_", 10.0)
            sim.update({"kTL1_1": 2.0, "cyt_prot__LIGAND_": 100.0})
            sim.update(kTL1_1=2.0, cyt_mrna__LIGAND_=10.0)

        Species writes go to the shared count store (and sync into both
        kernels' local state on the next ``run`` step). Parameter writes are
        routed to BNGsim and/or StochMod by id; if both models define the same
        id, both are updated.
        """
        updates: dict[str, float] = {}
        if isinstance(key, Mapping):
            updates.update({str(k): float(v) for k, v in key.items()})
        elif key is not None:
            if value is None:
                raise TypeError("update(name, value) requires a value")
            updates[str(key)] = float(value)
        updates.update({str(k): float(v) for k, v in kwargs.items()})
        if not updates:
            return

        for name, val in updates.items():
            self._apply_one(name, val)

    def _apply_one(self, name: str, value: float) -> None:
        is_species = name in self.index.name_to_index
        in_bng = name in self._bng_param_names
        in_stoch = name in self._stoch_param_names

        if not is_species and not in_bng and not in_stoch:
            raise KeyError(f"Unknown species or parameter id: {name!r}")

        if is_species:
            self._counts[self.index.name_to_index[name]] = value
            self.composite.state["counts"] = self._counts

        if in_bng:
            self._bng.kernel.model.set_param(name, value)
        if in_stoch:
            self._stoch.module.update(name, value)

    def set_counts(self, counts: np.ndarray | Mapping[str, float]) -> None:
        """Replace the shared count vector (array or {species: count} map)."""
        if isinstance(counts, Mapping):
            for name, val in counts.items():
                if name not in self.index.name_to_index:
                    raise KeyError(f"Unknown species id: {name!r}")
                self._counts[self.index.name_to_index[name]] = float(val)
        else:
            arr = np.asarray(counts, dtype=np.float64).reshape(-1)
            if arr.shape != (self.index.n_species,):
                raise ValueError(
                    f"counts shape {arr.shape} != ({self.index.n_species},)"
                )
            self._counts[:] = arr
        self.composite.state["counts"] = self._counts

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(
        self,
        t_end: float | None = None,
        *,
        dt: float | None = None,
        t_span: tuple[float, float] | None = None,
        engine: str = "split",
        progress_every: int | None = None,
    ) -> np.ndarray:
        """Advance the hybrid simulation; return the live counts vector.

        Parameters
        ----------
        t_end :
            Integrate from the current ``time`` to ``t_end`` (duration =
            ``t_end - time`` when ``time > 0``, else ``t_end`` from 0).
        t_span :
            ``(t0, t1)`` absolute window; resets the clock to ``t0`` conceptually
            for the step count ``(t1 - t0) / dt`` without clearing state.
        dt :
            Coupling step (default: constructor ``dt``).
        engine :
            ``"split"`` (default) or ``"composite"``.
        """
        step = float(self.dt if dt is None else dt)
        if t_span is not None:
            t0, t1 = float(t_span[0]), float(t_span[1])
            duration = t1 - t0
            if duration <= 0:
                raise ValueError(f"t_span must have t1 > t0, got {t_span}")
        elif t_end is not None:
            t_end = float(t_end)
            duration = t_end - self._t if self._t > 0 else t_end
            if duration <= 0:
                raise ValueError(
                    f"t_end={t_end} must be greater than current time={self._t}"
                )
            t1 = self._t + duration
        else:
            raise TypeError("run() requires t_end=... or t_span=(t0, t1)")

        # Sync shared counts into the composite store before stepping
        self.composite.state["counts"] = self._counts

        n_steps = int(np.floor(duration / step))
        if n_steps < 1:
            raise ValueError(f"duration={duration} / dt={step} yields no steps")

        if engine == "split":
            if n_steps > 1000:
                logger.warning(
                    "Running %d coupling steps at dt=%g. Prefer a larger dt when possible.",
                    n_steps,
                    step,
                )
            self._counts = run_operator_split(
                self.composite,
                t_end=duration,
                dt=step,
                progress_every=progress_every,
            )
        elif engine == "composite":
            # Temporarily set process intervals
            for name in ("bngsim", "stochmod"):
                node = self.composite.state[name]
                if isinstance(node, dict):
                    node["interval"] = step
            self.composite.run(float(duration))
            self._counts = np.asarray(self.composite.state["counts"], dtype=np.float64)
        else:
            raise ValueError(f"Unknown engine={engine!r}; use 'split' or 'composite'")

        self._t = float(t1) if t_span is not None else self._t + duration
        return self.counts

    def reset(self) -> None:
        """Restore initial counts and parameters; set time to 0."""
        self._counts[:] = self._initial_counts
        self.composite.state["counts"] = self._counts
        for name, val in self._initial_bng_params.items():
            self._bng.kernel.model.set_param(name, val)
        for name, val in self._initial_stoch_params.items():
            self._stoch.module.update(name, val)
        # Reset BNGsim interactive clock / IC if available
        reset = getattr(self._bng.kernel, "reset", None)
        if callable(reset):
            reset()
        self._t = 0.0

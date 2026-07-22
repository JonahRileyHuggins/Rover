"""Trajectory result buffers: in-memory or memory-mapped .npy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger("rover")

# Auto-switch to memmap when the dense trajectory would exceed this many bytes.
DEFAULT_MEMMAP_THRESHOLD_BYTES = 256 * 1024 * 1024  # 256 MiB


class TrajectoryStore:
    """Pre-sized (n_points, n_species) result buffer + parallel time vector.

    Live coupling state stays in a separate 1-D RAM vector. This store only
    receives one row write per coupling step (sequential, OS-cached). That is
    far cheaper than having BNGsim/StochMod open files themselves each step.
    """

    def __init__(
        self,
        n_points: int,
        n_species: int,
        *,
        backend: Literal["auto", "memory", "memmap"] = "auto",
        path: str | Path | None = None,
        memmap_threshold_bytes: int = DEFAULT_MEMMAP_THRESHOLD_BYTES,
    ) -> None:
        if n_points < 1:
            raise ValueError("n_points must be >= 1")
        if n_species < 1:
            raise ValueError("n_species must be >= 1")

        nbytes = n_points * n_species * 8
        if backend == "auto":
            backend = "memmap" if (path is not None or nbytes >= memmap_threshold_bytes) else "memory"

        self.backend = backend
        self.n_points = int(n_points)
        self.n_species = int(n_species)
        self._path: Path | None = None
        self._times_path: Path | None = None

        if backend == "memory":
            self.counts = np.zeros((n_points, n_species), dtype=np.float64)
            self.times = np.zeros(n_points, dtype=np.float64)
            logger.info(
                "Trajectory store: memory  shape=(%d, %d)  ~%.1f MiB",
                n_points,
                n_species,
                nbytes / (1024 * 1024),
            )
        elif backend == "memmap":
            if path is None:
                raise ValueError("memmap backend requires path=... to a .npy file")
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            times_path = path.with_name(path.stem + "_times" + path.suffix)
            self._path = path
            self._times_path = times_path
            self.counts = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float64, shape=(n_points, n_species)
            )
            self.times = np.lib.format.open_memmap(
                times_path, mode="w+", dtype=np.float64, shape=(n_points,)
            )
            logger.info(
                "Trajectory store: memmap  shape=(%d, %d)  ~%.1f MiB  -> %s",
                n_points,
                n_species,
                nbytes / (1024 * 1024),
                path,
            )
        else:
            raise ValueError(f"Unknown trajectory backend={backend!r}")

    def record(self, row: int, t: float, counts: np.ndarray) -> None:
        """Write one timepoint (copies the live 1-D counts into row ``row``)."""
        self.times[row] = t
        self.counts[row, :] = counts

    def flush(self) -> None:
        if isinstance(self.counts, np.memmap):
            self.counts.flush()
        if isinstance(self.times, np.memmap):
            self.times.flush()

    @property
    def path(self) -> Path | None:
        return self._path

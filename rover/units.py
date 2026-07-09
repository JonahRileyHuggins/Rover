"""Unit conversion between shared molecule counts and BNGsim nM storage.

BNGsim ``UnitConverter.to_amounts`` returns substance amounts in the model's
substance unit. For the Rover test SBML (substance = nanomole, concentrations
in nM), that is nanomoles. Molecule counts are::

    counts = nanomoles * 1e-9 * N_A
           = storage * volume_factor * 1e-9 * N_A

Precompute the per-species scale once and use elementwise multiply in the
hot path (no UnitConverter method calls per step).
"""

from __future__ import annotations

from typing import Any

import numpy as np

AVOGADRO = 6.022e23
NANOMOLAR_SCALE = 1e-9  # nanomole → mole


class CountConverter:
    """Precomputed counts ↔ BNGsim storage conversion for one model."""

    __slots__ = ("_to_counts", "_to_storage", "n_species")

    def __init__(self, volume_factors: np.ndarray) -> None:
        vf = np.asarray(volume_factors, dtype=np.float64)
        self._to_counts = vf * NANOMOLAR_SCALE * AVOGADRO
        self._to_storage = 1.0 / self._to_counts
        self.n_species = int(vf.shape[0])

    @classmethod
    def from_unit_converter(cls, unit_converter: Any) -> CountConverter:
        return cls(unit_converter.volume_factors)

    def counts_from_storage(self, storage: np.ndarray) -> np.ndarray:
        return np.asarray(storage, dtype=np.float64) * self._to_counts

    def storage_from_counts(self, counts: np.ndarray) -> np.ndarray:
        return np.asarray(counts, dtype=np.float64) * self._to_storage


def counts_from_bngsim_storage(storage: np.ndarray, unit_converter: Any) -> np.ndarray:
    """BNGsim storage (nM-style) → molecule counts."""
    return CountConverter.from_unit_converter(unit_converter).counts_from_storage(storage)


def bngsim_storage_from_counts(counts: np.ndarray, unit_converter: Any) -> np.ndarray:
    """Molecule counts → BNGsim storage (nM-style)."""
    return CountConverter.from_unit_converter(unit_converter).storage_from_counts(counts)


def nM_to_counts(concentration_nM: float, volume_L: float) -> float:
    """Convert a scalar nM concentration in a compartment to molecule counts."""
    return float(concentration_nM) * float(volume_L) * NANOMOLAR_SCALE * AVOGADRO


def counts_to_nM(counts: float, volume_L: float) -> float:
    """Convert molecule counts to nM concentration."""
    return float(counts) / (float(volume_L) * NANOMOLAR_SCALE * AVOGADRO)

"""Unit conversion between shared molecule counts and BNGsim nM storage.

BNGsim ``UnitConverter.to_amounts`` returns substance amounts in the model's
substance unit. For the Rover test SBML (substance = nanomole, concentrations
in nM), that is nanomoles. Molecule counts are::

    counts = nanomoles * 1e-9 * N_A

which is equivalent to ``nM * V_L * 1e-9 * N_A``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

AVOGADRO = 6.022e23
NANOMOLAR_SCALE = 1e-9  # nanomole → mole


def counts_from_bngsim_storage(storage: np.ndarray, unit_converter: Any) -> np.ndarray:
    """BNGsim storage (nM-style) → molecule counts."""
    nanomoles = np.asarray(unit_converter.to_amounts(storage), dtype=np.float64)
    return nanomoles * NANOMOLAR_SCALE * AVOGADRO


def bngsim_storage_from_counts(counts: np.ndarray, unit_converter: Any) -> np.ndarray:
    """Molecule counts → BNGsim storage (nM-style)."""
    nanomoles = np.asarray(counts, dtype=np.float64) / (NANOMOLAR_SCALE * AVOGADRO)
    return np.asarray(unit_converter.from_amounts(nanomoles), dtype=np.float64)


def nM_to_counts(concentration_nM: float, volume_L: float) -> float:
    """Convert a scalar nM concentration in a compartment to molecule counts."""
    return float(concentration_nM) * float(volume_L) * NANOMOLAR_SCALE * AVOGADRO


def counts_to_nM(counts: float, volume_L: float) -> float:
    """Convert molecule counts to nM concentration."""
    return float(counts) / (float(volume_L) * NANOMOLAR_SCALE * AVOGADRO)

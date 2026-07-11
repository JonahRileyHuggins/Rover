"""Unit conversion between shared molecule counts and BNGsim / StochMod.

BNGsim storage is nM-style concentration. Shared store currency is molecule
counts::

    counts = storage * volume_factor * 1e-9 * N_A

StochMod stores ``initialAmount`` as-is (counts) or
``initialConcentration * compartment_size``. For nM models the latter is
**nanomoles**, not molecules — convert at the StochMod boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

AVOGADRO = 6.022e23
NANOMOLAR_SCALE = 1e-9  # nanomole → mole
NANOMOLE_TO_MOLECULES = NANOMOLAR_SCALE * AVOGADRO


class CountConverter:
    """Precomputed counts ↔ BNGsim storage conversion for one model."""

    __slots__ = ("_to_counts", "_to_storage", "n_species")

    def __init__(self, volume_factors: np.ndarray) -> None:
        vf = np.asarray(volume_factors, dtype=np.float64)
        self._to_counts = vf * NANOMOLE_TO_MOLECULES
        self._to_storage = 1.0 / self._to_counts
        self.n_species = int(vf.shape[0])

    @classmethod
    def from_unit_converter(cls, unit_converter: Any) -> CountConverter:
        return cls(unit_converter.volume_factors)

    def counts_from_storage(self, storage: np.ndarray) -> np.ndarray:
        return np.asarray(storage, dtype=np.float64) * self._to_counts

    def storage_from_counts(self, counts: np.ndarray) -> np.ndarray:
        return np.asarray(counts, dtype=np.float64) * self._to_storage


def stochmod_to_molecule_scales(sbml_path: str | Path) -> np.ndarray:
    """Per-species factor: StochMod internal amount → molecule counts.

    ``initialAmount`` / substance-unit species → 1.0.
    ``initialConcentration`` species in an nM/nanomole model → ``1e-9 * N_A``
    (StochMod stores ``conc * V``, i.e. nanomoles).
    """
    import libsbml

    doc = libsbml.readSBMLFromFile(str(sbml_path))
    model = doc.getModel()
    if model is None:
        raise ValueError(f"No model in SBML: {sbml_path}")

    nanomole_substance = False
    for ud in model.getListOfUnitDefinitions():
        if ud.getId() not in ("substance", "nM"):
            continue
        for u in ud.getListOfUnits():
            if u.getKind() == libsbml.UNIT_KIND_MOLE and u.getScale() == -9:
                nanomole_substance = True

    scales: list[float] = []
    for sp in model.getListOfSpecies():
        if sp.getHasOnlySubstanceUnits() or (
            sp.isSetInitialAmount() and not sp.isSetInitialConcentration()
        ):
            scales.append(1.0)
        elif sp.isSetInitialConcentration() and nanomole_substance:
            scales.append(NANOMOLE_TO_MOLECULES)
        elif sp.isSetInitialConcentration():
            scales.append(AVOGADRO)  # mol/L × L → moles → molecules
        else:
            scales.append(1.0)
    return np.asarray(scales, dtype=np.float64)


def counts_from_bngsim_storage(storage: np.ndarray, unit_converter: Any) -> np.ndarray:
    """BNGsim storage (nM-style) → molecule counts."""
    return CountConverter.from_unit_converter(unit_converter).counts_from_storage(storage)


def bngsim_storage_from_counts(counts: np.ndarray, unit_converter: Any) -> np.ndarray:
    """Molecule counts → BNGsim storage (nM-style)."""
    return CountConverter.from_unit_converter(unit_converter).storage_from_counts(counts)


def nM_to_counts(concentration_nM: float, volume_L: float) -> float:
    """Convert a scalar nM concentration in a compartment to molecule counts."""
    return float(concentration_nM) * float(volume_L) * NANOMOLE_TO_MOLECULES


def counts_to_nM(counts: float, volume_L: float) -> float:
    """Convert molecule counts to nM concentration."""
    return float(counts) / (float(volume_L) * NANOMOLE_TO_MOLECULES)

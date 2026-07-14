"""Unit conversion between shared molecule counts and BNGsim / StochMod.

BNGsim storage is nM-style concentration. Shared store currency is molecule
counts::

    counts = storage * volume_factor * 1e-9 * N_A

StochMod keeps SBML amounts as its internal *file* representation. At runtime
StochMod uses count-based propensities (compartment normalizer), so the live
state matches the shared molecule-count store (identity bridge).

When a companion deterministic SBML is available, ``stochmod_to_molecule_scales``
converts SBML ICs that are still numerically nM into molecule counts for
seeding (no species-name rules). Zero-IC overlap inherits the majority vote.
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


def _nanomole_substance(model) -> bool:
    import libsbml

    for ud in model.getListOfUnitDefinitions():
        if ud.getId() not in ("substance", "nM"):
            continue
        for u in ud.getListOfUnits():
            if u.getKind() == libsbml.UNIT_KIND_MOLE and u.getScale() == -9:
                return True
    return False


def _compartment_sizes(model) -> dict[str, float]:
    return {
        c.getId(): float(c.getSize()) if c.isSetSize() else 1.0
        for c in model.getListOfCompartments()
    }


def _species_amount(sp) -> float:
    if sp.isSetInitialAmount():
        return float(sp.getInitialAmount())
    if sp.isSetInitialConcentration():
        return float(sp.getInitialConcentration())
    return 0.0


def _det_nM_and_volumes(deterministic_sbml: str | Path) -> dict[str, tuple[float, float]]:
    """Map species id → (nM-style IC, compartment volume_L)."""
    import libsbml

    doc = libsbml.readSBMLFromFile(str(deterministic_sbml))
    model = doc.getModel()
    if model is None:
        raise ValueError(f"No model in SBML: {deterministic_sbml}")
    sizes = _compartment_sizes(model)
    out: dict[str, tuple[float, float]] = {}
    for sp in model.getListOfSpecies():
        V = sizes.get(sp.getCompartment(), 1.0)
        if sp.isSetInitialConcentration():
            nM = float(sp.getInitialConcentration())
        elif sp.isSetInitialAmount() and V > 0.0 and _nanomole_substance(model):
            # Amount in nanomoles → nM = amount / V
            nM = float(sp.getInitialAmount()) / V
        else:
            nM = float(sp.getInitialAmount()) if sp.isSetInitialAmount() else 0.0
        out[sp.getId()] = (nM, V)
    return out


def stochmod_to_molecule_scales(
    sbml_path: str | Path,
    *,
    companion_deterministic_sbml: str | Path | None = None,
) -> np.ndarray:
    """Per-species factor: StochMod **SBML IC amount** → molecule counts.

    Used when seeding the shared store from the stochastic file. Runtime
    StochMod advances use molecule counts directly (identity bridge).

    Default from SBML flags. With ``companion_deterministic_sbml``, overlapping
    species whose StochMod amount matches deterministic nM (not molecule count)
    use ``V * 1e-9 * N_A``. Zero-IC overlap inherits the majority vote.
    """
    import libsbml

    doc = libsbml.readSBMLFromFile(str(sbml_path))
    model = doc.getModel()
    if model is None:
        raise ValueError(f"No model in SBML: {sbml_path}")

    nanomole_substance = _nanomole_substance(model)
    sizes = _compartment_sizes(model)

    scales: list[float] = []
    species = list(model.getListOfSpecies())
    for sp in species:
        if sp.getHasOnlySubstanceUnits() or (
            sp.isSetInitialAmount() and not sp.isSetInitialConcentration()
        ):
            scales.append(1.0)
        elif sp.isSetInitialConcentration() and nanomole_substance:
            scales.append(NANOMOLE_TO_MOLECULES)
        elif sp.isSetInitialConcentration():
            scales.append(AVOGADRO)
        else:
            scales.append(1.0)

    if companion_deterministic_sbml is not None:
        # Overlap species share IDs with the deterministic SBML. Align by IC:
        # amount≈nM → scale V*N_A*1e-9; amount≈molecules → scale 1. Zero-IC
        # overlap inherits the majority vote so newly produced species keep
        # the same StochMod currency as the rest of the overlap set.
        det = _det_nM_and_volumes(companion_deterministic_sbml)
        overlap_idx: list[int] = []
        votes_nm = 0
        votes_mol = 0
        decided: dict[int, str] = {}
        for i, sp in enumerate(species):
            sid = sp.getId()
            if sid not in det:
                continue
            overlap_idx.append(i)
            amt = _species_amount(sp)
            nM, V = det[sid]
            molecules = nM * V * NANOMOLE_TO_MOLECULES
            if amt == 0.0 and nM == 0.0:
                continue
            err_nm = abs(amt - nM) / max(abs(nM), abs(amt), 1e-12)
            err_mol = abs(amt - molecules) / max(abs(molecules), abs(amt), 1e-12)
            if err_nm < err_mol and err_nm < 0.05:
                votes_nm += 1
                decided[i] = "nm"
            elif err_mol <= err_nm and err_mol < 0.05:
                votes_mol += 1
                decided[i] = "mol"
        if votes_nm + votes_mol > 0:
            default = "nm" if votes_nm >= votes_mol else "mol"
            for i in overlap_idx:
                mode = decided.get(i, default)
                _nM, V = det[species[i].getId()]
                if mode == "nm":
                    scales[i] = float(V) * NANOMOLE_TO_MOLECULES
                else:
                    scales[i] = 1.0

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

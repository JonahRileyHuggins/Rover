"""Unit conversion between shared molecule counts and BNGsim / StochMod.

BNGsim storage is nM-style concentration. Shared store currency is molecule
counts for physically concentration-based species::

    counts = storage * volume_factor * 1e-9 * N_A

SPARCED-style partitions dual-encode some overlap species: the stochastic
file stores molecule counts while the deterministic file stores the *same
numbers* as ``initialConcentration`` in nM. For those species the intended
hybrid currency is the shared numeric value itself (1–20 for mRNA), and the
BNGsim bridge is identity (storage ↔ counts), matching how the SBML kinetics
were written. Detection uses IC comparison only (no species-name rules).

StochMod runtime state is count-based (compartment normalizer) — identity
bridge to the shared store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

AVOGADRO = 6.022e23
NANOMOLAR_SCALE = 1e-9  # nanomole → mole
NANOMOLE_TO_MOLECULES = NANOMOLAR_SCALE * AVOGADRO

OverlapMode = Literal["numeric", "physical"]


class CountConverter:
    """Precomputed counts ↔ BNGsim storage conversion for one model."""

    __slots__ = ("_to_counts", "_to_storage", "n_species")

    def __init__(self, to_counts: np.ndarray) -> None:
        self._to_counts = np.asarray(to_counts, dtype=np.float64)
        if np.any(self._to_counts == 0.0):
            raise ValueError("count conversion factors must be nonzero")
        self._to_storage = 1.0 / self._to_counts
        self.n_species = int(self._to_counts.shape[0])

    @classmethod
    def from_unit_converter(
        cls,
        unit_converter: Any,
        *,
        species_names: list[str] | tuple[str, ...] | None = None,
        identity_species: set[str] | frozenset[str] | None = None,
    ) -> CountConverter:
        """Build converter; ``identity_species`` use factor 1 (storage == counts)."""
        vf = np.asarray(unit_converter.volume_factors, dtype=np.float64)
        to_counts = vf * NANOMOLE_TO_MOLECULES
        if identity_species and species_names is not None:
            if len(species_names) != to_counts.shape[0]:
                raise ValueError(
                    f"species_names length {len(species_names)} != "
                    f"volume_factors {to_counts.shape[0]}"
                )
            for i, name in enumerate(species_names):
                if name in identity_species:
                    to_counts[i] = 1.0
        return cls(to_counts)

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
            nM = float(sp.getInitialAmount()) / V
        else:
            nM = float(sp.getInitialAmount()) if sp.isSetInitialAmount() else 0.0
        out[sp.getId()] = (nM, V)
    return out


def overlap_currency_modes(
    stochastic_sbml: str | Path,
    deterministic_sbml: str | Path,
) -> dict[str, OverlapMode]:
    """Classify overlap species by comparing StochMod amounts to det ICs.

    ``numeric``
        Stoch amount ≈ det nM label (dual-encoded). Shared value equals BNGsim
        storage; BNGsim bridge is identity.
    ``physical``
        Stoch amount ≈ det nM · V · N_A · 1e-9. Shared value is true molecules;
        BNGsim bridge uses volume conversion.

    Zero-IC overlap inherits the majority vote among nonzero overlap.
    """
    import libsbml

    doc = libsbml.readSBMLFromFile(str(stochastic_sbml))
    model = doc.getModel()
    if model is None:
        raise ValueError(f"No model in SBML: {stochastic_sbml}")
    det = _det_nM_and_volumes(deterministic_sbml)

    decided: dict[str, OverlapMode] = {}
    votes_numeric = 0
    votes_physical = 0
    zero_ids: list[str] = []
    for sp in model.getListOfSpecies():
        sid = sp.getId()
        if sid not in det:
            continue
        amt = _species_amount(sp)
        nM, V = det[sid]
        molecules = nM * V * NANOMOLE_TO_MOLECULES
        if amt == 0.0 and nM == 0.0:
            zero_ids.append(sid)
            continue
        err_nm = abs(amt - nM) / max(abs(nM), abs(amt), 1e-12)
        err_mol = abs(amt - molecules) / max(abs(molecules), abs(amt), 1e-12)
        if err_nm < err_mol and err_nm < 0.05:
            decided[sid] = "numeric"
            votes_numeric += 1
        elif err_mol <= err_nm and err_mol < 0.05:
            decided[sid] = "physical"
            votes_physical += 1

    if votes_numeric + votes_physical > 0:
        default: OverlapMode = (
            "numeric" if votes_numeric >= votes_physical else "physical"
        )
        for sid in zero_ids:
            decided[sid] = default
    return decided


def stochmod_to_molecule_scales(
    sbml_path: str | Path,
    *,
    companion_deterministic_sbml: str | Path | None = None,
) -> np.ndarray:
    """Per-species factor: StochMod **SBML IC amount** → shared molecule counts.

    Runtime StochMod advances use molecule counts directly (identity bridge).
    With a companion deterministic SBML, overlap species in either currency mode
    already store the shared count value in the stochastic file (scale 1).
    """
    import libsbml

    doc = libsbml.readSBMLFromFile(str(sbml_path))
    model = doc.getModel()
    if model is None:
        raise ValueError(f"No model in SBML: {sbml_path}")

    nanomole_substance = _nanomole_substance(model)
    species = list(model.getListOfSpecies())
    scales: list[float] = []
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
        modes = overlap_currency_modes(sbml_path, companion_deterministic_sbml)
        for i, sp in enumerate(species):
            if sp.getId() in modes:
                # Dual-encoded numeric and true-molecule ICs both already equal
                # the shared count currency in the stochastic file.
                scales[i] = 1.0

    return np.asarray(scales, dtype=np.float64)


def counts_from_bngsim_storage(
    storage: np.ndarray,
    unit_converter: Any,
    *,
    species_names: list[str] | tuple[str, ...] | None = None,
    identity_species: set[str] | frozenset[str] | None = None,
) -> np.ndarray:
    """BNGsim storage (nM-style) → shared counts."""
    return CountConverter.from_unit_converter(
        unit_converter,
        species_names=species_names,
        identity_species=identity_species,
    ).counts_from_storage(storage)


def bngsim_storage_from_counts(
    counts: np.ndarray,
    unit_converter: Any,
    *,
    species_names: list[str] | tuple[str, ...] | None = None,
    identity_species: set[str] | frozenset[str] | None = None,
) -> np.ndarray:
    """Shared counts → BNGsim storage (nM-style)."""
    return CountConverter.from_unit_converter(
        unit_converter,
        species_names=species_names,
        identity_species=identity_species,
    ).storage_from_counts(counts)


def nM_to_counts(concentration_nM: float, volume_L: float) -> float:
    """Convert a scalar nM concentration in a compartment to molecule counts."""
    return float(concentration_nM) * float(volume_L) * NANOMOLE_TO_MOLECULES


def counts_to_nM(counts: float, volume_L: float) -> float:
    """Convert molecule counts to nM concentration."""
    return float(counts) / (float(volume_L) * NANOMOLE_TO_MOLECULES)

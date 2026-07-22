"""Unit conversion between shared nanomolar state and StochMod molecules.

Shared store currency is **nanomolar** (matches BNGsim storage and SBML
``initialConcentration``). BNGsim uses an identity bridge. StochMod leaps in
molecule counts (SingleCell-shaped)::

    molecules = nM * volume_L * 1e-9 * N_A
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

AVOGADRO = 6.022e23
NANOMOLAR_SCALE = 1e-9  # nanomole → mole
NANOMOLE_TO_MOLECULES = NANOMOLAR_SCALE * AVOGADRO


def _compartment_sizes(model) -> dict[str, float]:
    return {
        c.getId(): float(c.getSize()) if c.isSetSize() else 1.0
        for c in model.getListOfCompartments()
    }


def species_volumes_L(sbml_path: str | Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Return ``(species_ids, volumes_L)`` in SBML species order."""
    import libsbml

    doc = libsbml.readSBMLFromFile(str(sbml_path))
    model = doc.getModel()
    if model is None:
        raise ValueError(f"No model in SBML: {sbml_path}")
    sizes = _compartment_sizes(model)
    names: list[str] = []
    volumes: list[float] = []
    for sp in model.getListOfSpecies():
        names.append(sp.getId())
        volumes.append(float(sizes.get(sp.getCompartment(), 1.0)))
    return tuple(names), np.asarray(volumes, dtype=np.float64)


def nM_to_molecules_factors(volumes_L: np.ndarray) -> np.ndarray:
    """Per-species factor: nM → molecule counts (``V * N_A * 1e-9``)."""
    factors = np.asarray(volumes_L, dtype=np.float64) * NANOMOLE_TO_MOLECULES
    if np.any(factors == 0.0):
        raise ValueError("nM→molecule factors must be nonzero")
    return factors


def nM_to_molecules(concentration_nM: np.ndarray, factors: np.ndarray) -> np.ndarray:
    return np.asarray(concentration_nM, dtype=np.float64) * factors


def molecules_to_nM(molecules: np.ndarray, factors: np.ndarray) -> np.ndarray:
    return np.asarray(molecules, dtype=np.float64) / factors


def sbml_initial_nM(sbml_path: str | Path) -> dict[str, float]:
    """Map species id → initial concentration in nM from an SBML file."""
    import libsbml

    doc = libsbml.readSBMLFromFile(str(sbml_path))
    model = doc.getModel()
    if model is None:
        raise ValueError(f"No model in SBML: {sbml_path}")
    sizes = _compartment_sizes(model)
    out: dict[str, float] = {}
    for sp in model.getListOfSpecies():
        V = float(sizes.get(sp.getCompartment(), 1.0))
        if sp.isSetInitialConcentration():
            out[sp.getId()] = float(sp.getInitialConcentration())
        elif sp.isSetInitialAmount() and V > 0.0:
            out[sp.getId()] = float(sp.getInitialAmount()) / V
        else:
            out[sp.getId()] = 0.0
    return out


def nM_to_counts(concentration_nM: float, volume_L: float) -> float:
    """Convert a scalar nM concentration in a compartment to molecule counts."""
    return float(concentration_nM) * float(volume_L) * NANOMOLE_TO_MOLECULES


def counts_to_nM(counts: float, volume_L: float) -> float:
    """Convert molecule counts to nM concentration."""
    return float(counts) / (float(volume_L) * NANOMOLE_TO_MOLECULES)

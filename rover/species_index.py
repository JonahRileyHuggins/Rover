"""Dense species index and ownership masks for hybrid partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import libsbml
except ImportError as exc:  # pragma: no cover
    raise ImportError("python-libsbml is required for species indexing") from exc


@dataclass(frozen=True)
class SpeciesIndex:
    """Ordered union of species IDs with ownership masks."""

    names: tuple[str, ...]
    name_to_index: dict[str, int]
    deterministic_mask: np.ndarray  # bool, length N — BNGsim writes these
    stochastic_mask: np.ndarray  # bool, length N — StochMod writes these

    @property
    def n_species(self) -> int:
        return len(self.names)

    def indices_of(self, names: Sequence[str]) -> np.ndarray:
        return np.asarray([self.name_to_index[n] for n in names], dtype=np.int64)


def _species_ids_from_sbml(path: str | Path) -> list[str]:
    doc = libsbml.readSBMLFromFile(str(path))
    model = doc.getModel()
    if model is None:
        raise ValueError(f"Failed to parse SBML (no model): {path}")
    return [sp.getId() for sp in model.getListOfSpecies()]


def _annotation_text(species) -> str:
    ann = species.getAnnotationString() or ""
    # Prefer plain <annotation>...</annotation> body if present
    if "<annotation>" in ann and "</annotation>" in ann:
        start = ann.find("<annotation>") + len("<annotation>")
        end = ann.find("</annotation>")
        return ann[start:end].strip()
    return ann.strip()


def ownership_from_sbml(path: str | Path) -> dict[str, str]:
    """Map species id → 'deterministic' | 'stochastic' from SBML annotations.

    Annotations look like ``Cytoplasm 0 Deterministic`` / ``Nucleus 1 Stochastic``.
    Defaults to ``stochastic`` if the role token is missing.
    """
    doc = libsbml.readSBMLFromFile(str(path))
    model = doc.getModel()
    if model is None:
        raise ValueError(f"No model in SBML: {path}")

    roles: dict[str, str] = {}
    for sp in model.getListOfSpecies():
        text = _annotation_text(sp).lower()
        if "deterministic" in text:
            roles[sp.getId()] = "deterministic"
        elif "stochastic" in text:
            roles[sp.getId()] = "stochastic"
        else:
            roles[sp.getId()] = "stochastic"
    return roles


def build_species_index(
    *sbml_paths: str | Path,
    ownership_sbml: str | Path | None = None,
) -> SpeciesIndex:
    """Build a dense index from the ordered union of species across SBML files.

    Species order: first occurrence across ``sbml_paths`` (stable).
    Ownership is read from ``ownership_sbml`` (default: first path).
    """
    if not sbml_paths:
        raise ValueError("At least one SBML path is required")

    seen: list[str] = []
    seen_set: set[str] = set()
    for path in sbml_paths:
        for sid in _species_ids_from_sbml(path):
            if sid not in seen_set:
                seen.append(sid)
                seen_set.add(sid)

    roles = ownership_from_sbml(ownership_sbml or sbml_paths[0])
    n = len(seen)
    det = np.zeros(n, dtype=bool)
    stoch = np.zeros(n, dtype=bool)
    for i, name in enumerate(seen):
        role = roles.get(name, "stochastic")
        if role == "deterministic":
            det[i] = True
        else:
            stoch[i] = True

    return SpeciesIndex(
        names=tuple(seen),
        name_to_index={n: i for i, n in enumerate(seen)},
        deterministic_mask=det,
        stochastic_mask=stoch,
    )


def local_to_global(
    local_names: Sequence[str],
    index: SpeciesIndex,
) -> np.ndarray:
    """Map a local species-name order to global dense indices."""
    missing = [n for n in local_names if n not in index.name_to_index]
    if missing:
        raise KeyError(f"Species not in global index: {missing[:5]}")
    return np.asarray([index.name_to_index[n] for n in local_names], dtype=np.int64)


def gather(global_counts: np.ndarray, local_indices: np.ndarray) -> np.ndarray:
    return np.asarray(global_counts, dtype=np.float64)[local_indices].copy()


def scatter_delta(
    local_delta: np.ndarray,
    local_indices: np.ndarray,
    n_global: int,
    ownership_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Scatter a local delta into a zero global delta array, optionally masked."""
    out = np.zeros(n_global, dtype=np.float64)
    out[local_indices] = local_delta
    if ownership_mask is not None:
        out = np.where(ownership_mask, out, 0.0)
    return out

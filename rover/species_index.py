"""Dense species index from SBML file membership (no ownership annotations)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import libsbml
except ImportError as exc:  # pragma: no cover
    raise ImportError("python-libsbml is required for species indexing") from exc


@dataclass(frozen=True)
class SpeciesIndex:
    """Ordered union of species IDs with membership masks.

    ``deterministic_mask`` / ``stochastic_mask`` mean “present in that SBML”,
    not write ownership. Overlap is ``deterministic_mask & stochastic_mask``.
    """

    names: tuple[str, ...]
    name_to_index: dict[str, int]
    deterministic_mask: np.ndarray  # bool — in deterministic SBML
    stochastic_mask: np.ndarray  # bool — in stochastic SBML

    @property
    def n_species(self) -> int:
        return len(self.names)

    @property
    def overlap_mask(self) -> np.ndarray:
        return self.deterministic_mask & self.stochastic_mask

    @property
    def deterministic_only_mask(self) -> np.ndarray:
        return self.deterministic_mask & ~self.stochastic_mask

    @property
    def stochastic_only_mask(self) -> np.ndarray:
        return self.stochastic_mask & ~self.deterministic_mask

    def indices_of(self, names: Sequence[str]) -> np.ndarray:
        return np.asarray([self.name_to_index[n] for n in names], dtype=np.int64)


def _species_ids_from_sbml(path: str | Path) -> list[str]:
    doc = libsbml.readSBMLFromFile(str(path))
    model = doc.getModel()
    if model is None:
        raise ValueError(f"Failed to parse SBML (no model): {path}")
    return [sp.getId() for sp in model.getListOfSpecies()]


def build_species_index(
    *sbml_paths: str | Path,
    deterministic_sbml: str | Path | None = None,
    stochastic_sbml: str | Path | None = None,
    **_unused,
) -> SpeciesIndex:
    """Build a dense index from the ordered union of species across SBML files.

    Membership only: a species is marked deterministic / stochastic if it
    appears in that partition file. Overlap is allowed (shared modifiers /
    exchange species).
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

    if deterministic_sbml is None or stochastic_sbml is None:
        if len(sbml_paths) >= 2:
            stochastic_sbml = stochastic_sbml or sbml_paths[0]
            deterministic_sbml = deterministic_sbml or sbml_paths[1]
        else:
            stochastic_sbml = stochastic_sbml or sbml_paths[0]
            deterministic_sbml = deterministic_sbml or sbml_paths[0]

    det_ids = set(_species_ids_from_sbml(deterministic_sbml))
    stoch_ids = set(_species_ids_from_sbml(stochastic_sbml))

    n = len(seen)
    det = np.zeros(n, dtype=bool)
    stoch = np.zeros(n, dtype=bool)
    for i, name in enumerate(seen):
        det[i] = name in det_ids
        stoch[i] = name in stoch_ids

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


def exchange_counts(
    s0: np.ndarray,
    *,
    stoch_local: np.ndarray,
    stoch_indices: np.ndarray,
    bng_local: np.ndarray,
    bng_indices: np.ndarray,
    index: SpeciesIndex,
) -> np.ndarray:
    """Merge parallel module results into a new shared count vector.

    - Stochastic-only / deterministic-only: take that module's post-step value
    - Overlap: ``s0 + Δ_stoch + Δ_bng`` (modifier-only modules contribute ~0)
    """
    out = np.asarray(s0, dtype=np.float64).copy()
    stoch_full = out.copy()
    bng_full = out.copy()
    stoch_full[stoch_indices] = np.asarray(stoch_local, dtype=np.float64)
    bng_full[bng_indices] = np.asarray(bng_local, dtype=np.float64)

    only_stoch = index.stochastic_only_mask
    only_bng = index.deterministic_only_mask
    overlap = index.overlap_mask

    out[only_stoch] = stoch_full[only_stoch]
    out[only_bng] = bng_full[only_bng]
    out[overlap] = (
        s0[overlap]
        + (stoch_full[overlap] - s0[overlap])
        + (bng_full[overlap] - s0[overlap])
    )
    # Molecule counts are non-negative; ODE/tau-leap + delta exchange can
    # undershoot and poison the next CVODE window if negatives are kept.
    np.maximum(out, 0.0, out=out)
    return out

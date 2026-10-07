"""In-memory records used while constructing a PXRD basis library.

The public artifact returned by :func:`ariadnex.library.build_library` is the
path-based ``LibraryArtifact`` from ``ariadnex.contracts``.  These records are
kept deliberately small and serialisable so that adapters can be replaced in
tests without importing GSAS-II, PyTorch, or pandas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CifMetadata:
    """Metadata extracted from one CIF file.

    ``source_id`` is the stable identity used by every downstream stage.  A
    validator may leave optional fields empty when the source does not provide
    them; validation status and a human-readable reason are always recorded.
    """

    source_id: str
    source_path: Path
    parse_status: str = "pass"
    formula: str = ""
    elements: tuple[str, ...] = ()
    space_group: str = ""
    space_group_number: int | None = None
    cell_volume: float | None = None
    n_sites: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        if isinstance(self.elements, str):
            parsed_elements = re.findall(r"[A-Z][a-z]?", self.elements)
        else:
            parsed_elements = [str(item) for item in self.elements]
        object.__setattr__(self, "elements", tuple(sorted(set(parsed_elements))))

    @property
    def sort_key(self) -> tuple[str, str, str]:
        """Order calculations by source content, independently of input filenames."""

        metadata = self.metadata or {}
        accession = str(metadata.get("source_accession", ""))
        digest = str(metadata.get("source_digest", ""))
        return (f"{accession}_" if accession else digest or self.source_id, digest, self.source_id)


@dataclass(frozen=True)
class PxrdPattern:
    """One simulated PXRD pattern on a shared theta grid."""

    source_id: str
    x: np.ndarray
    y: np.ndarray
    source_path: Path | None = None

    def __post_init__(self) -> None:
        # Keep GSAS-II's export precision through basis preprocessing.  The
        # encoder adapter performs its own explicit float32 conversion, while
        # reducing basis patterns to float32 here can move signature knots and
        # measurably change an otherwise identical MILP solution.
        x = np.asarray(self.x, dtype=np.float64).reshape(-1)
        y = np.asarray(self.y, dtype=np.float64).reshape(-1)
        if x.size != y.size or x.size < 2:
            raise ValueError(
                "PXRD pattern x/y arrays must have the same length (at least two points)."
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("PXRD pattern contains non-finite values.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))


@dataclass(frozen=True)
class ClusterResult:
    """Numerical result of deterministic hierarchical clustering."""

    labels: np.ndarray
    representative_indices: np.ndarray
    assignment_representative_indices: np.ndarray
    distances_to_representative: np.ndarray
    threshold: float
    selection: str
    linkage_matrix: np.ndarray
    threshold_scan: tuple[Mapping[str, Any], ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", np.asarray(self.labels, dtype=np.int32))
        object.__setattr__(
            self, "representative_indices", np.asarray(self.representative_indices, dtype=np.int32)
        )
        object.__setattr__(
            self,
            "assignment_representative_indices",
            np.asarray(self.assignment_representative_indices, dtype=np.int32),
        )
        object.__setattr__(
            self,
            "distances_to_representative",
            np.asarray(self.distances_to_representative, dtype=np.float32),
        )
        object.__setattr__(
            self, "linkage_matrix", np.asarray(self.linkage_matrix, dtype=np.float64)
        )

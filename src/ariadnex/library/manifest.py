"""Candidate CIF discovery and manifest parsing.

The library stage has one input seam: a directory, a single CIF, or a CSV
manifest.  Parsing that seam once keeps path resolution, stable IDs, labels,
and ordering in one place before validation or numerical work begins.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CIF_PATH_ALIASES = ("cif_path", "cif", "path", "source_path", "file", "filename")
_PHASE_ID_ALIASES = ("phase_id", "source_id", "sample_id", "id")
_LABEL_ALIASES = ("label", "display_name", "name", "human_readable_label")


@dataclass(frozen=True)
class CifEntry:
    """One discovered CIF and the optional fields supplied by its manifest."""

    path: Path
    explicit_id: str | None = None
    label: str | None = None


def _field_map(fieldnames: Any) -> dict[str, Any]:
    return {str(key).strip().lower(): key for key in fieldnames if key is not None}


def _read_rows(path: Path, *, empty_message: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{empty_message}: {path}")
    return rows


def _resolve_phase_path(raw_path: str, *, input_path: Path, manifest_path: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    # Keep this order stable.  Existing phase manifests sometimes contain
    # repository-root prefixes while the configured raw-CIF directory is the
    # authoritative source tree.
    candidates = [manifest_path.parent / candidate, input_path / candidate]
    basename_candidate = input_path / candidate.name
    if basename_candidate not in candidates:
        candidates.append(basename_candidate)
    selected = next((item for item in candidates if item.is_file()), candidates[0])
    return selected.resolve()


def _manifest_entries(
    input_path: Path,
    manifest_path: Path,
    *,
    phase_manifest: bool,
) -> list[CifEntry]:
    rows = _read_rows(
        manifest_path,
        empty_message="Phase manifest is empty" if phase_manifest else "CIF manifest is empty",
    )
    fields = _field_map(rows[0])
    cif_key = next((fields[key] for key in _CIF_PATH_ALIASES if key in fields), None)
    if cif_key is None:
        kind = "Phase manifest" if phase_manifest else "CIF manifest"
        raise ValueError(
            f"{kind} must contain a CIF path column (one of {', '.join(_CIF_PATH_ALIASES)}): {manifest_path}"
        )
    id_key = next((fields[key] for key in _PHASE_ID_ALIASES if key in fields), None)
    label_key = next((fields[key] for key in _LABEL_ALIASES if key in fields), None)

    entries: list[CifEntry] = []
    for index, row in enumerate(rows, start=1):
        raw_path = str(row.get(cif_key, "") or "").strip()
        if not raw_path:
            kind = "Phase manifest" if phase_manifest else "CIF manifest"
            raise ValueError(
                f"{kind} row {index} has no CIF path; expected one of "
                f"{', '.join(_CIF_PATH_ALIASES)}: {manifest_path}"
            )

        if phase_manifest:
            path = _resolve_phase_path(raw_path, input_path=input_path, manifest_path=manifest_path)
        else:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = manifest_path.parent / path
            path = path.resolve()
        if not path.is_file():
            kind = "Phase manifest" if phase_manifest else "CIF manifest"
            raise FileNotFoundError(f"{kind} row {index} references missing CIF: {path}")

        raw_id = row.get(id_key, "") if id_key else ""
        explicit_id = str(raw_id or "").strip() or None
        raw_label = row.get(label_key, "") if label_key else ""
        label = str(raw_label or "").strip() or None
        entries.append(CifEntry(path=path, explicit_id=explicit_id, label=label))

    return sorted(entries, key=lambda item: (item.explicit_id or item.path.stem, str(item.path)))


def discover_cifs(input_path: Path, phase_manifest: Path | None = None) -> list[CifEntry]:
    """Discover candidate CIFs once, preserving the historical resolution order."""

    input_path = Path(input_path).expanduser().resolve()
    if phase_manifest is not None:
        manifest_path = Path(phase_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Phase manifest does not exist: {manifest_path}")
        return _manifest_entries(input_path, manifest_path, phase_manifest=True)

    if input_path.is_file():
        if input_path.suffix.lower() == ".csv":
            return _manifest_entries(input_path, input_path, phase_manifest=False)
        if input_path.suffix.lower() == ".cif":
            return [CifEntry(path=input_path)]
        raise ValueError(f"Unsupported CIF input file: {input_path}")
    if not input_path.is_dir():
        raise FileNotFoundError(f"CIF input path does not exist: {input_path}")
    return [
        CifEntry(path=path)
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() == ".cif"
    ]


__all__ = ["CifEntry", "discover_cifs"]

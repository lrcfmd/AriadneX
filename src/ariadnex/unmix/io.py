"""Input loading for the anomaly-aware unmixing stage.

This module owns manifest/path decoding and basis preprocessing. The solver
receives serialisable in-memory records, while mixture artifacts use the
canonical reader from ariadnex.mixtures.pipeline.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ariadnex.config import ProjectConfig
from ariadnex.contracts import LibraryArtifact
from ariadnex.mixtures.clean import preprocess_pattern, reduce_signature, signature_widths
from ariadnex.mixtures.load import load_pattern_file

from .element_prior import parse_formula_elements
from .solver import _resolve_config_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_phase_order_manifest(path: Path) -> list[str]:
    """Read and validate a solver phase-order manifest."""

    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            raw_fieldnames = list(reader.fieldnames or ())
            phase_key = next(
                (name for name in raw_fieldnames if str(name).strip().lower() == "phase_id"), None
            )
            if phase_key is None:
                raise ValueError(f"phase order manifest needs a phase_id column: {path}")
            ids: list[str] = []
            for row in reader:
                value = str(row.get(phase_key, "") or "").strip()
                if not value:
                    raise ValueError(f"phase order manifest contains a blank phase_id: {path}")
                ids.append(value)
    except OSError as exc:
        raise FileNotFoundError(f"phase order manifest not found: {path}") from exc
    if not ids:
        raise ValueError(f"phase order manifest is empty: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"phase order manifest contains duplicate phase_id values: {path}")
    return ids


def _apply_phase_order_manifest(
    config: ProjectConfig, library: Sequence[Mapping[str, Any]], manifest_value: Any
) -> list[Mapping[str, Any]]:
    """Reorder basis records according to a solver-order manifest.

    Leave-one-out experiment views intentionally omit one phase, so they use
    the induced order of the full manifest.  Standard runs require an exact
    ID-set match.
    """

    path = _resolve_config_path(config, manifest_value)
    order_ids = _read_phase_order_manifest(path)
    library_ids = [str(item.get("phase_id", "")).strip() for item in library]
    if any(not value for value in library_ids):
        raise ValueError("loaded library records must have a phase_id")
    if len(set(library_ids)) != len(library_ids):
        raise ValueError("loaded library contains duplicate phase_id values")
    library_set = set(library_ids)
    order_set = set(order_ids)
    leave_one_out = str(config.get("experiment.type", "standard")).lower() == "leave_one_out"
    if leave_one_out:
        if not library_set <= order_set:
            missing = sorted(library_set - order_set)
            raise ValueError(
                f"phase order manifest is missing library phase_id values: {', '.join(missing)}"
            )
    elif library_set != order_set:
        missing = sorted(library_set - order_set)
        extra = sorted(order_set - library_set)
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if extra:
            detail.append(f"unexpected {', '.join(extra)}")
        raise ValueError(
            f"phase order manifest ID set does not match loaded library ({'; '.join(detail)})"
        )
    records_by_id = {phase_id: record for phase_id, record in zip(library_ids, library)}
    return [records_by_id[phase_id] for phase_id in order_ids if phase_id in records_by_id]


def _split_elements(value: Any) -> set[str]:
    if value is None:
        return set()
    text = str(value).replace(",", " ").replace(";", " ").replace("|", " ")
    values = [item.strip() for item in text.split() if item.strip()]
    if len(values) == 1:
        return parse_formula_elements(values[0])
    return set(values)


def _archive_basis(path: Path, phase_id: str):
    def _numeric_array(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise ValueError(f"Basis archive {path} {name} must contain a numeric array")
        if array.ndim not in {1, 2}:
            raise ValueError(f"Basis archive {path} {name} must be one- or two-dimensional")
        return array

    def _phase_ids(value: Any) -> list[str]:
        array = np.asarray(value)
        if array.ndim == 0:
            values = [array.item()]
        elif array.ndim == 1:
            values = array.tolist()
        elif array.ndim == 2 and 1 in array.shape:
            values = array.reshape(-1).tolist()
        else:
            raise ValueError(f"Basis archive {path} phase_ids must be one-dimensional")
        ids = [str(item).strip() for item in values]
        if any(not item for item in ids):
            raise ValueError(f"Basis archive {path} contains a blank phase_id")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Basis archive {path} contains duplicate phase_ids")
        return ids

    def _select(x_value: Any, y_value: Any, ids_value: Any | None = None):
        x_array = _numeric_array(x_value, "x")
        y_array = _numeric_array(y_value, "y")
        if ids_value is None:
            if x_array.ndim == 2 and x_array.shape[0] != 1:
                raise ValueError(f"Basis archive {path} has multiple x/y rows but no phase_ids")
            if y_array.ndim == 2 and y_array.shape[0] != 1:
                raise ValueError(f"Basis archive {path} has multiple x/y rows but no phase_ids")
            x_row = x_array.reshape(-1)
            y_row = y_array.reshape(-1)
        else:
            ids = _phase_ids(ids_value)
            expected_rows = len(ids)
            y_rows = 1 if y_array.ndim == 1 else y_array.shape[0]
            if y_rows != expected_rows:
                raise ValueError(
                    f"Basis archive {path} has inconsistent x/y row counts: "
                    f"{expected_rows} phase_ids but {y_rows} y rows"
                )
            if x_array.ndim == 2 and x_array.shape[0] != expected_rows:
                raise ValueError(
                    f"Basis archive {path} has inconsistent x/y row counts: "
                    f"{expected_rows} phase_ids but {x_array.shape[0]} x rows"
                )
            try:
                index = ids.index(str(phase_id))
            except ValueError as exc:
                raise ValueError(
                    f"Basis archive {path} does not contain requested phase_id {phase_id!r}"
                ) from exc
            x_row = x_array if x_array.ndim == 1 else x_array[index]
            y_row = y_array if y_array.ndim == 1 else y_array[index]
        if x_row.size != y_row.size:
            raise ValueError(
                f"Basis archive {path} has inconsistent x/y lengths ({x_row.size} vs {y_row.size})"
            )
        # Return independent arrays so callers never retain views into an
        # NpzFile that is closed when this function exits.
        return np.array(x_row, copy=True), np.array(y_row, copy=True)

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            lowered = {str(key).lower(): key for key in keys}
            x_key = next(
                (lowered[name] for name in ("x", "theta", "positions") if name in lowered), None
            )
            y_key = next(
                (
                    lowered[name]
                    for name in ("y", "intensity", "weights", "patterns")
                    if name in lowered
                ),
                None,
            )
            if x_key is None or y_key is None:
                raise ValueError(f"Basis archive {path} needs x/y (or positions/weights) arrays")
            # Copy every value needed by the parser before closing the archive.
            x_value = np.array(archive[x_key], copy=True)
            y_value = np.array(archive[y_key], copy=True)
            ids_value = (
                np.array(archive[lowered["phase_ids"]], copy=True)
                if "phase_ids" in lowered
                else None
            )
        return _select(x_value, y_value, ids_value)

    # ``.npy`` is a supported pattern suffix.  It has no named phase rows, so
    # only its single-image [x, y] representation is meaningful here.
    array = _numeric_array(np.load(path, allow_pickle=False), "pattern")
    if array.ndim == 2 and array.shape[0] == 2 and array.shape[1] > 2:
        return _select(np.array(array[0], copy=True), np.array(array[1], copy=True))
    if array.ndim == 2 and array.shape[1] >= 2:
        return _select(np.array(array[:, 0], copy=True), np.array(array[:, 1], copy=True))
    raise ValueError(f"Basis archive {path} needs x/y (or positions/weights) arrays")


def _load_library(
    library: LibraryArtifact,
    *,
    instrument: str | Path | None,
    signature_points: int | str | None,
    preprocessing: Mapping[str, Any] | None,
):
    """Load exactly the representative rows listed by the library manifest."""

    manifest_path = Path(library.representatives_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Representative manifest not found: {manifest_path}")
    rows = _read_csv(manifest_path)
    if not rows:
        raise ValueError(f"Representative manifest is empty: {manifest_path}")
    directory = Path(library.basis_patterns_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Basis pattern directory not found: {directory}")
    points = int(signature_points) if signature_points not in (None, "") else None
    records: list[dict[str, Any]] = []
    for row in rows:
        phase_id = str(row["phase_id"]).strip()
        label = row.get("label")
        candidate = Path(str(row["basis_pattern_path"])).expanduser()
        if not candidate.is_absolute():
            candidate = directory / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            # The recorded path is absolute; a moved run keeps the file name.
            fallback = directory / candidate.name
            if not fallback.is_file():
                raise FileNotFoundError(
                    f"Basis pattern for representative {phase_id!r} not found: {candidate}"
                )
            candidate = fallback.resolve()
        if candidate.suffix.lower() in {".npy", ".npz"}:
            x, y = _archive_basis(candidate, phase_id)
        else:
            loaded = load_pattern_file(candidate, sample_id=phase_id)
            x, y = loaded["x"], loaded["y"]
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.size != y.size or x.size < 2:
            raise ValueError(f"Basis pattern {candidate} has invalid x/y arrays")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Basis pattern {candidate} contains non-finite x/y values")
        if np.any(y < 0.0):
            raise ValueError(f"Basis pattern {candidate} contains negative intensities")
        if float(y.sum()) <= 0.0:
            raise ValueError(f"Basis pattern {candidate} has no positive intensity")
        order = np.argsort(x, kind="mergesort")
        x, y = x[order], y[order]
        if preprocessing is not None:
            processed = preprocess_pattern(
                {"sample_id": phase_id, "source_path": str(candidate), "x": x, "y": y},
                signature_points=points,
                instrument=instrument,
                **preprocessing,
            )
            x = np.asarray(processed["clean_x"], dtype=float)
            y = np.asarray(processed["clean_y"], dtype=float)
            positions = np.asarray(processed["signature_positions"], dtype=float)
            weights = np.asarray(processed["signature_weights"], dtype=float)
            widths = processed.get("signature_widths")
        else:
            total = float(y.sum())
            y /= total
            positions, weights = reduce_signature(x, y, points=points)
            widths, _width_info = signature_widths(x, y, positions, instrument=instrument)
        elements = _split_elements(row["elements"])
        records.append(
            {
                "phase_id": phase_id,
                "label": str(label or phase_id),
                "path": str(candidate),
                "signature": (positions, weights),
                "clean_x": x,
                "clean_y": y,
                "elements": elements,
                "signature_widths": widths,
                "source_row": row,
            }
        )
    return records

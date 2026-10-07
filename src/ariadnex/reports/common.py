"""Shared path, field, and configuration helpers for reports."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .models import (
    _ANOMALY_NAMES,
    _OPEN_WORLD_SAMPLE_ID,
    _Sample,
)


def _normalise_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _path_value(value: object | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    return None


def _read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                _normalise_key(key): str(value).strip()
                for key, value in row.items()
                if key is not None
            }
            for row in csv.DictReader(handle)
        ]


def _first(row: Mapping[str, object], names: Sequence[str], default: str = "") -> str:
    normalised = {_normalise_key(key): value for key, value in row.items()}
    for name in names:
        key = _normalise_key(name)
        if key in normalised and str(normalised[key]).strip() != "":
            return str(normalised[key]).strip()
    return default


def _float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip().replace("%", "")
    if not text:
        return default
    try:
        number = float(text)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _ordered_curve(
    axis: object | None,
    values: object,
    *,
    context: str = "report curve",
) -> tuple[np.ndarray, np.ndarray]:
    """Return a finite, ascending axis and its correspondingly ordered curve.

    Workflow preprocessing normally emits an ascending two-theta grid.  Report
    artifacts can also be supplied independently, though, and a descending
    grid makes ``numpy.interp`` return an apparently empty curve.  Keep the
    common ascending path byte-for-byte unchanged while handling reversed and
    repeated coordinates deterministically.  A supplied axis must have the
    same length as its curve so a malformed artifact cannot be reinterpreted
    silently on an arbitrary index grid.
    """

    curve = np.asarray(values, dtype=float).reshape(-1)
    if axis is None:
        x = np.arange(curve.size, dtype=float)
    else:
        x = np.asarray(axis, dtype=float).reshape(-1)
        if x.size != curve.size:
            raise ValueError(f"{context} axis and intensity lengths differ")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(curve)):
        raise ValueError(f"{context} contains non-finite axis or intensity values")
    if x.size < 2:
        return x, curve
    if np.all(np.diff(x) > 0.0):
        return x, curve

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    curve = curve[order]
    if np.any(np.diff(x) == 0.0):
        unique, inverse = np.unique(x, return_inverse=True)
        sums = np.zeros(unique.size, dtype=float)
        counts = np.zeros(unique.size, dtype=float)
        np.add.at(sums, inverse, curve)
        np.add.at(counts, inverse, 1.0)
        x = unique
        curve = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0.0)
    return x, curve


def _bool(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "selected", "present"}:
        return True
    if text in {"0", "false", "no", "n", "unselected", "absent"}:
        return False
    return None


def _is_anomaly_phase(phase_id: str, row: Mapping[str, object] | None = None) -> bool:
    if row is not None:
        for key in ("is_anomaly", "anomaly", "unknown", "is_unknown"):
            if _normalise_key(key) in {_normalise_key(item) for item in row}:
                explicit = _bool(_first(row, (key,)))
                if explicit is not None:
                    return explicit
    return _normalise_key(phase_id) in _ANOMALY_NAMES or _normalise_key(phase_id).startswith(
        "anomaly"
    )


def _sample_identity(sample_id: object) -> str:
    """Return the stable key shared by open-world manifests and source tables.

    Source tables may use a complete filename stem (for example
    ``NH273_BaYRuSiO_1300``), while the converted manifest uses ``NH273``.
    Other sample identifiers remain unchanged.
    """

    text = str(sample_id or "").strip()
    match = _OPEN_WORLD_SAMPLE_ID.search(text)
    return match.group(1).upper() if match is not None else text


def _truth_for_sample(
    truth: Mapping[str, Mapping[str, float]],
    sample_id: str,
) -> Mapping[str, float]:
    direct = truth.get(sample_id)
    if direct is not None:
        return direct
    identity = _sample_identity(sample_id)
    matches = [row for key, row in truth.items() if _sample_identity(key) == identity]
    if not matches:
        return {}
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Ground truth has multiple matches for sample {sample_id!r}; "
        "use matching sample_id values in the measurements and ground-truth table"
    )


def _percentage_scale(values: Iterable[float], *, header: str = "") -> float:
    values_list = [abs(float(value)) for value in values if math.isfinite(float(value))]
    if not values_list:
        return 1.0
    header_key = _normalise_key(header)
    if "pct" in header_key or "percent" in header_key or "%" in header:
        return 1.0
    # A table of fractions has a row sum close to one.  Single values are
    # intentionally treated as fractions too, which matches the artifact
    # contract and avoids mixing fraction and percentage outputs.
    return 100.0 if max(values_list) <= 1.0 else 1.0


def _sample(samples: dict[str, _Sample], sample_id: str) -> _Sample:
    clean = str(sample_id).strip()
    if not clean:
        clean = "sample_"
    if clean not in samples:
        samples[clean] = _Sample(clean)
    return samples[clean]


def _safe_report_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "holdout"

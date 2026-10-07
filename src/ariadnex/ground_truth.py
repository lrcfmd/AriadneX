"""Read optional phase amounts for preflight and report evaluation.

The reader preserves sample and phase identifiers and returns percentages.
It depends only on the standard library, so checking input does not load the
solver or plotting modules. Blank cells in wide tables mean an absent phase.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Mapping

_SAMPLE_KEYS = ("sample_id", "sample", "mixture_name", "mixture", "datapoint", "id")
_PHASE_KEYS = ("phase_id", "phase", "phase_name", "component", "component_id", "label", "name")
_VALUE_KEYS = (
    "fraction",
    "fraction_pct",
    "abundance",
    "abundance_pct",
    "value",
    "amount",
    "wt_pct",
    "percent",
)
_UNITS = {"fraction": 100.0, "ratio": 100.0, "percent": 1.0, "wtpct": 1.0, "pct": 1.0}


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _amount(value: object, context: str) -> float:
    try:
        amount = float(str(value).strip().removesuffix("%"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}: expected a non-negative numeric phase amount, got {value!r}"
        ) from exc
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"{context}: phase amounts must be finite and non-negative")
    return amount


def _scale(values: list[float], header: str = "") -> float:
    key = _normalise(header)
    if "pct" in key or "percent" in key or "%" in header:
        return 1.0
    return 100.0 if values and max(values) <= 1.0 else 1.0


def _percentage(value: object, scale: float, context: str) -> float:
    amount = _amount(value, context) * (1.0 if str(value).strip().endswith("%") else scale)
    if not math.isfinite(amount):
        raise ValueError(f"{context}: phase amount overflows when converted to percent")
    return amount


def load_ground_truth(
    value: object | None,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Read long/wide CSV or an inline sample-to-phase mapping.

    Long tables can declare ``fraction_unit`` per row. Without units, the
    existing convention is retained: values at most one are fractions.
    Identifiers are case-sensitive; duplicate entries and invalid values fail
    with their source location rather than being discarded or overwritten.
    """

    if value is None:
        return {}, {}
    if isinstance(value, Mapping):
        path_value = next(
            (
                value[key]
                for key in ("path", "csv_path", "ground_truth_path", "truth_path")
                if isinstance(value.get(key), (str, Path))
            ),
            None,
        )
        if path_value is not None:
            value = path_value
        else:
            truth: dict[str, dict[str, float]] = {}
            raw_values = []
            for sample_id, row in value.items():
                if not str(sample_id).strip() or not isinstance(row, Mapping):
                    raise ValueError(
                        "Ground truth needs non-empty sample IDs mapped to phase amounts"
                    )
                for phase_id, amount in row.items():
                    if not str(phase_id).strip():
                        raise ValueError(f"Ground truth for {sample_id!r} has an empty phase ID")
                    raw_values.append(_amount(amount, f"Ground truth {sample_id!r}/{phase_id!r}"))
            scale = _scale(raw_values)
            for sample_id, row in value.items():
                truth[str(sample_id)] = {
                    str(phase_id): _percentage(
                        amount, scale, f"Ground truth {sample_id!r}/{phase_id!r}"
                    )
                    for phase_id, amount in row.items()
                }
            return truth, {phase: phase for row in truth.values() for phase in row}
    if not isinstance(value, (str, Path)):
        raise ValueError("Ground truth must be a CSV path or sample-to-phase mapping")
    path = Path(value).expanduser()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = [key.strip() for key in reader.fieldnames or []]
        if not headers or any(not key for key in headers) or len(set(headers)) != len(headers):
            raise ValueError(f"Ground truth {path} needs unique, non-empty column names")
        reader.fieldnames = headers
        rows = list(reader)
    if not rows:
        raise ValueError(f"Configured ground-truth file is empty: {path}")
    for index, row in enumerate(rows, start=2):
        if None in row or any(item is None for item in row.values()):
            raise ValueError(
                f"Ground truth {path}, row {index}: column count differs from the header"
            )

    def column(aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            matches = [key for key in headers if _normalise(key) == _normalise(alias)]
            if len(matches) > 1:
                raise ValueError(f"Ground truth {path} has ambiguous columns for {alias!r}")
            if matches:
                return matches[0]
        return None

    sample_key = column(_SAMPLE_KEYS)
    if sample_key is None:
        raise ValueError(f"Ground truth {path} needs a sample_id column")
    phase_key, value_key = column(_PHASE_KEYS), column(_VALUE_KEYS)
    truth = {}
    labels: dict[str, str] = {}
    if phase_key is not None and value_key is not None:
        raw_values = [
            _amount(row[value_key], f"Ground truth {path}, row {index}")
            for index, row in enumerate(rows, start=2)
        ]
        scale = _scale(raw_values, value_key)
        unit_key, label_key = column(("fraction_unit", "unit")), column(("phase_label",))
        for index, row in enumerate(rows, start=2):
            context = f"Ground truth {path}, row {index}"
            sample_id, phase_id = row[sample_key].strip(), row[phase_key].strip()
            if not sample_id or not phase_id:
                raise ValueError(f"{context}: sample_id and phase_id must be non-empty")
            unit = _normalise(row[unit_key]) if unit_key else ""
            if unit and unit not in _UNITS:
                raise ValueError(f"{context}: unknown fraction_unit {row[unit_key]!r}")
            sample_truth = truth.setdefault(sample_id, {})
            if phase_id in sample_truth:
                raise ValueError(f"{context}: duplicate entry for {sample_id!r}/{phase_id!r}")
            sample_truth[phase_id] = _percentage(row[value_key], _UNITS.get(unit, scale), context)
            labels[phase_id] = (row[label_key].strip() if label_key else "") or phase_id
        return truth, labels

    phase_columns = [key for key in headers if key != sample_key]
    if not phase_columns:
        raise ValueError(f"Ground truth {path} needs at least one phase column")
    raw_values = [
        _amount(row[key], f"Ground truth {path}, row {index}, phase {key!r}")
        for index, row in enumerate(rows, start=2)
        for key in phase_columns
        if row[key].strip()
    ]
    scale = _scale(raw_values)
    for index, row in enumerate(rows, start=2):
        context = f"Ground truth {path}, row {index}"
        sample_id = row[sample_key].strip()
        if not sample_id or sample_id in truth:
            raise ValueError(f"{context}: empty or duplicate sample_id {sample_id!r}")
        truth[sample_id] = {
            phase: _percentage(row[phase], scale, f"{context}, phase {phase!r}")
            for phase in phase_columns
            if row[phase].strip()
        }
    return truth, {phase: phase for phase in phase_columns}

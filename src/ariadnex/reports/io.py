"""Readers for standard unmix and optional report reference artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..contracts import UnmixArtifact
from .common import (
    _first,
    _float,
    _is_anomaly_phase,
    _normalise_key,
    _path_value,
    _percentage_scale,
    _read_rows,
    _sample,
)
from .models import _BasisProfile, _ResultData, _Sample
from .phases import formula_key


def _parse_summary(path: Path | None, samples: dict[str, _Sample]) -> None:
    rows = _read_rows(path)
    for row in rows:
        sample_id = _first(row, ("sample_id",))
        if not sample_id:
            continue
        sample = _sample(samples, sample_id)
        sample.summary_fields.update(row)
        selected_ids = _first(row, ("selected_phase_ids",))
        if selected_ids and not sample.selected_phase_ids:
            sample.selected_phase_ids.extend(
                phase.strip() for phase in re.split(r"[;,|]", selected_ids) if phase.strip()
            )
        anomaly = _float(_first(row, ("anomaly_pct",)))
        if anomaly is not None:
            sample.anomaly_pct = max(0.0, anomaly)
        objective = _float(_first(row, ("objective",)))
        if objective is not None:
            sample.objective = objective
        sample.solver_status = _first(row, ("solver_status",))
        sample.warnings = _first(row, ("warnings",))


def _npz_mapping(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=True) as archive:
            return {str(key): np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError, EOFError):
        return {}


def _find_array(mapping: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray | None:
    by_key = {_normalise_key(key): value for key, value in mapping.items()}
    for name in names:
        value = by_key.get(_normalise_key(name))
        if value is not None:
            return np.asarray(value)
    return None


def _sample_ids_from_array(array: np.ndarray | None, count: int) -> list[str]:
    if array is None:
        return [f"sample_{idx + 1}" for idx in range(count)]
    values = np.asarray(array).reshape(-1)
    return [
        str(value.item() if hasattr(value, "item") else value).strip() for value in values[:count]
    ]


def _row_array(array: np.ndarray | None, index: int, sample_count: int) -> np.ndarray | None:
    if array is None:
        return None
    values = np.asarray(array)
    if values.ndim == 0:
        return np.asarray(values, dtype=float)
    if values.ndim == 1:
        # A one-dimensional vector is a shared axis when its length does not
        # equal the number of samples; otherwise it is a per-sample scalar.
        if values.dtype == object and values.shape[0] == sample_count:
            return np.asarray(values[index], dtype=float).reshape(-1)
        if sample_count == 1 or values.shape[0] != sample_count:
            return np.asarray(values, dtype=float)
        return np.asarray(values[index], dtype=float)
    if values.shape[0] == sample_count:
        row = values[index]
    elif values.shape[1] == sample_count:
        row = values[:, index]
    else:
        raise ValueError(f"array shape {values.shape} has no axis matching {sample_count} samples")
    # Standard result arrays are (samples, points).  A component dimension is
    # reduced only for an explicitly named curve, by summing non-point axes.
    return np.asarray(row, dtype=float).reshape(-1)


def _component_curve(components: np.ndarray, index: int) -> np.ndarray | None:
    """Collapse a (sample, phase, support) component array to one curve."""

    try:
        row = np.asarray(components[index], dtype=float)
        if row.ndim >= 2:
            return np.sum(row, axis=0).reshape(-1)
        return row.reshape(-1)
    except (IndexError, TypeError, ValueError):
        try:
            values = np.asarray(components[index], dtype=object).reshape(-1)
            curves = [
                np.asarray(item, dtype=float).reshape(-1)
                for item in values
                if np.asarray(item).size
            ]
            if not curves:
                return None
            length = min(curve.size for curve in curves)
            return np.sum(np.vstack([curve[:length] for curve in curves]), axis=0)
        except (IndexError, TypeError, ValueError):
            return None


def _component_signatures(
    components: np.ndarray,
    index: int,
    phase_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return target-space component masses keyed by basis phase."""

    try:
        rows = np.asarray(components[index], dtype=float)
    except (IndexError, TypeError, ValueError):
        return {}
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return {
        phase_id: np.asarray(rows[phase_index], dtype=float).reshape(-1)
        for phase_index, phase_id in enumerate(phase_ids)
        if phase_index < rows.shape[0] and float(np.sum(rows[phase_index])) > 0.0
    }


def _basis_profiles(path: Path | None) -> dict[str, _BasisProfile]:
    mapping = _npz_mapping(path)
    phase_array = _find_array(mapping, ("phase_ids",))
    positions_array = _find_array(mapping, ("basis_positions",))
    weights_array = _find_array(mapping, ("basis_weights",))
    x_array = _find_array(mapping, ("basis_x",))
    y_array = _find_array(mapping, ("basis_y",))
    if any(
        value is None for value in (phase_array, positions_array, weights_array, x_array, y_array)
    ):
        return {}
    phase_ids = [str(value) for value in np.asarray(phase_array).reshape(-1)]
    profiles: dict[str, _BasisProfile] = {}
    for index, phase_id in enumerate(phase_ids):
        x = _row_array(x_array, index, len(phase_ids))
        y = _row_array(y_array, index, len(phase_ids))
        positions = _row_array(positions_array, index, len(phase_ids))
        weights = _row_array(weights_array, index, len(phase_ids))
        if any(value is None for value in (x, y, positions, weights)):
            continue
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.maximum(np.asarray(y, dtype=float).reshape(-1), 0.0)
        positions = np.asarray(positions, dtype=float).reshape(-1)
        weights = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
        if x.size != y.size or positions.size != weights.size or x.size < 3:
            continue
        profiles[phase_id] = _BasisProfile(x=x, y=y, positions=positions, weights=weights)
    return profiles


def _parse_npz(
    path: Path | None, samples: dict[str, _Sample], phase_order: list[str] | None = None
) -> None:
    mapping = _npz_mapping(path)
    if not mapping:
        return
    sample_array = _find_array(mapping, ("sample_ids",))
    phase_array = _find_array(mapping, ("phase_ids",))
    abundance_array = _find_array(mapping, ("abundance",))
    anomaly_array = _find_array(mapping, ("anomalies",))
    components = _find_array(mapping, ("components",))
    count_candidates = [
        int(np.asarray(array).shape[0])
        for array in (anomaly_array, components, abundance_array)
        if array is not None and np.asarray(array).ndim > 1
    ]
    declared_count = (
        int(np.asarray(sample_array).reshape(-1).size) if sample_array is not None else None
    )
    count = declared_count if declared_count is not None else max(count_candidates, default=0)
    if count <= 0:
        return
    if declared_count is not None and declared_count != count:
        raise ValueError(
            f"results NPZ sample_ids count ({declared_count}) does not match curve arrays ({count})"
        )
    for name, array in (
        ("anomalies", anomaly_array),
        ("components", components),
        ("abundance", abundance_array),
    ):
        if array is None or np.asarray(array).ndim <= 1:
            continue
        shape = np.asarray(array).shape
        if count not in shape[:2]:
            raise ValueError(
                f"results NPZ {name!r} has shape {shape}; no axis matches {count} samples"
            )
    ids = (
        list(samples)
        if sample_array is None and len(samples) == count
        else _sample_ids_from_array(sample_array, count)
    )
    abundance_phases = (
        [str(value) for value in np.asarray(phase_array).reshape(-1)]
        if phase_array is not None
        else list(phase_order or ())
    )
    if phase_order is not None:
        for phase_id in abundance_phases:
            if not _is_anomaly_phase(phase_id) and phase_id not in phase_order:
                phase_order.append(phase_id)
    abundance_scale = 1.0
    if abundance_array is not None:
        abundance_values = np.asarray(abundance_array, dtype=object).reshape(-1)
        numeric_values: list[float] = []
        for value in abundance_values:
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                continue
        abundance_scale = _percentage_scale(numeric_values)
    for index, sample_id in enumerate(ids):
        sample = _sample(samples, sample_id)
        sample_count = len(ids)
        sample.anomaly = _row_array(anomaly_array, index, sample_count)
        if sample.anomaly_pct is None and sample.anomaly is not None:
            sample.anomaly_pct = max(0.0, float(np.sum(sample.anomaly)) * 100.0)
        if components is not None:
            sample.component_signatures = _component_signatures(
                components,
                index,
                abundance_phases,
            )
        if abundance_array is not None and abundance_phases:
            abundance_row = _row_array(abundance_array, index, sample_count)
            if abundance_row is not None:
                for phase_index, phase_id in enumerate(abundance_phases):
                    if phase_index >= abundance_row.size:
                        break
                    amount = max(0.0, float(abundance_row[phase_index]) * abundance_scale)
                    if _is_anomaly_phase(phase_id):
                        sample.anomaly_pct = amount
                    else:
                        sample.abundances[phase_id] = amount
        if sample.known is None and components is not None:
            sample.known = _component_curve(components, index)
        if sample.total is None and sample.known is not None and sample.anomaly is not None:
            sample.total = sample.known + sample.anomaly


def _parse_sibling_mixture_patterns(result_root: Path | None, samples: dict[str, _Sample]) -> None:
    """Fill experimental curves from the adjacent standard mixture artifact."""

    if result_root is None:
        return
    roots = [result_root.parent, *result_root.parents]
    path = next(
        (
            base / "02_mixtures" / "patterns.npz"
            for base in roots
            if (base / "02_mixtures" / "patterns.npz").exists()
        ),
        result_root.parent / "02_mixtures" / "patterns.npz",
    )
    mapping = _npz_mapping(path)
    if not mapping:
        return
    ids_array = _find_array(mapping, ("sample_ids",))
    x_array = _find_array(mapping, ("x",))
    y_array = _find_array(mapping, ("y",))
    raw_x_array = _find_array(mapping, ("raw_x",))
    raw_y_array = _find_array(mapping, ("raw_y",))
    baseline_array = _find_array(mapping, ("baseline",))
    if x_array is None or y_array is None:
        return
    count = len(np.asarray(ids_array).reshape(-1)) if ids_array is not None else len(samples)
    ids = _sample_ids_from_array(ids_array, count) if ids_array is not None else list(samples)
    signature_positions: dict[str, np.ndarray] = {}
    signature_weights: dict[str, np.ndarray] = {}
    signature_mapping = _npz_mapping(path.with_name("signatures.npz"))
    positions_array = _find_array(signature_mapping, ("positions",))
    weights_array = _find_array(signature_mapping, ("weights",))
    if positions_array is not None:
        for index, sample_id in enumerate(ids):
            try:
                position = np.asarray(positions_array[index], dtype=float).reshape(-1)
            except (IndexError, TypeError, ValueError):
                continue
            if position.size:
                signature_positions[sample_id] = position
    if weights_array is not None:
        for index, sample_id in enumerate(ids):
            try:
                weights = np.asarray(weights_array[index], dtype=float).reshape(-1)
            except (IndexError, TypeError, ValueError):
                continue
            if weights.size:
                signature_weights[sample_id] = weights
    for index, sample_id in enumerate(ids):
        sample = _sample(samples, sample_id)
        x = _row_array(x_array, index, len(ids))
        y = _row_array(y_array, index, len(ids))
        if x is None or y is None:
            continue
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.size != y.size:
            continue
        source_x = signature_positions.get(sample_id)
        if source_x is None and sample.x is not None:
            candidate = np.asarray(sample.x, dtype=float).reshape(-1)
            if sample.anomaly is not None and candidate.size == np.asarray(sample.anomaly).size:
                source_x = candidate
        if source_x is not None:
            sample.signature_positions = source_x
            sample.signature_weights = signature_weights.get(sample_id)
            if sample.known is not None and np.asarray(sample.known).size == source_x.size:
                sample.known_signature = np.asarray(sample.known, dtype=float).reshape(-1)
            if sample.anomaly is not None and np.asarray(sample.anomaly).size == source_x.size:
                sample.anomaly_signature = np.asarray(sample.anomaly, dtype=float).reshape(-1)
            if sample.total is not None and np.asarray(sample.total).size == source_x.size:
                sample.total_signature = np.asarray(sample.total, dtype=float).reshape(-1)
            if (
                sample.signature_weights is None
                and sample.mixture is not None
                and np.asarray(sample.mixture).size == source_x.size
            ):
                sample.signature_weights = np.asarray(sample.mixture, dtype=float).reshape(-1)

        # ``patterns.npz`` is authoritative for the report grid and observed
        # curve. Solver-space masses remain in the dedicated signature fields
        # above and are rebuilt after one dataset-level sigma is fitted.
        sample.x = x
        sample.mixture = y
        sample.raw_x = _row_array(raw_x_array, index, len(ids))
        sample.raw_y = _row_array(raw_y_array, index, len(ids))
        sample.baseline = _row_array(baseline_array, index, len(ids))


def _load_sibling_phase_metadata(
    result_root: Path | None,
) -> tuple[
    dict[str, str],
    dict[tuple[tuple[str, float], ...], str],
    dict[str, tuple[tuple[str, float], ...]],
]:
    if result_root is None:
        return {}, {}, {}
    complete_library = (
        result_root.parent.parent / "01_library" / "representatives.csv"
        if result_root.parent.name == "03_unmix"
        else result_root.parent / "01_library" / "representatives.csv"
    )
    candidates = [
        complete_library,
        result_root.parent / "01_library" / "representatives.csv",
        result_root / "library_view" / "representatives.csv",
        result_root.parent / "library_view" / "representatives.csv",
    ]
    holdout_name = result_root.name.removeprefix("holdout_")
    if holdout_name != result_root.name:
        candidates.append(
            result_root.parent / "library_views" / holdout_name / "representatives.csv"
        )
    candidates.extend(base / "01_library" / "representatives.csv" for base in result_root.parents)
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    manifest_candidates = [path.parent / "manifest.csv"]
    manifest_candidates.extend(base / "01_library" / "manifest.csv" for base in result_root.parents)
    manifest_path = next(
        (candidate for candidate in manifest_candidates if candidate.exists()),
        manifest_candidates[0],
    )
    manifest_rows = {
        _first(row, ("phase_id",)): row
        for row in _read_rows(manifest_path)
        if _first(row, ("phase_id",))
    }
    labels: dict[str, str] = {}
    aliases: dict[tuple[tuple[str, float], ...], str] = {}
    phase_formula_keys: dict[str, tuple[tuple[str, float], ...]] = {}
    for row in _read_rows(path):
        phase_id = _first(row, ("phase_id",))
        formula = _first(row, ("formula",))
        label = _first(row, ("label",))
        if phase_id:
            # Representative labels in the new library are often the source
            # ID itself.  A formula is more useful in recovery figures and is
            # the bridge to formula-based ground truth.
            labels[phase_id] = formula or label or phase_id
            metadata = manifest_rows.get(phase_id, {})
            recorded_formula = _first(metadata, ("source_formula",))
            recorded_key = formula_key(recorded_formula)
            if recorded_key is not None:
                phase_formula_keys[phase_id] = recorded_key
            for candidate in (recorded_formula, formula, label, phase_id):
                key = formula_key(candidate)
                if key is not None:
                    aliases.setdefault(key, phase_id)
    return labels, aliases, phase_formula_keys


def _load_result(result: UnmixArtifact) -> _ResultData:
    samples: dict[str, _Sample] = {}
    phase_order: list[str] = []
    basis_profiles: dict[str, _BasisProfile] = {}
    # The summary defines sample identity and order and carries the rounded
    # anomaly percentages; the NPZ supplies abundances, curves, and basis data.
    _parse_summary(result.sample_summary_csv, samples)
    _parse_npz(result.results_npz, samples, phase_order)
    basis_profiles = _basis_profiles(result.results_npz)
    _parse_sibling_mixture_patterns(result.root, samples)
    labels, phase_aliases, phase_formula_keys = _load_sibling_phase_metadata(result.root)
    if not samples:
        raise ValueError("No samples were found in the standard unmix artifacts")
    labels.update({phase: phase for phase in phase_order if phase not in labels})
    return _ResultData(
        samples=samples,
        phase_order=phase_order,
        labels=labels,
        phase_aliases=phase_aliases,
        phase_formula_keys=phase_formula_keys,
        basis_profiles=basis_profiles,
    )


def _load_reference(value: object | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if value is None:
        return {}

    def reference_pair(
        name: str,
        x_value: object,
        y_value: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x_value, dtype=float).reshape(-1)
        y = np.asarray(y_value, dtype=float).reshape(-1)
        if x.size != y.size or x.size < 2:
            raise ValueError(
                f"anomaly reference {name!r} needs matching x/y arrays with at least 2 points"
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError(f"anomaly reference {name!r} contains non-finite values")
        return x, y

    if isinstance(value, Mapping):
        references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, item in value.items():
            if isinstance(item, Mapping):
                x = item.get("x")
                if x is None:
                    x = item.get("theta")
                y = item.get("y")
                if y is None:
                    y = item.get("curve")
                if y is None:
                    y = item.get("intensity")
                if x is not None and y is not None:
                    references[str(key)] = reference_pair(str(key), x, y)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                references[str(key)] = reference_pair(str(key), item[0], item[1])
        if not references:
            raise ValueError("anomaly reference mapping has no usable curves")
        return references
    path = _path_value(value)
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"configured anomaly reference does not exist: {path}")

    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"configured anomaly reference is not a file or directory: {path}")
    if path.is_dir():
        references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for child in sorted(path.iterdir()):
            if not child.is_file() or child.suffix.lower() == ".npz":
                continue
            try:
                values = np.loadtxt(child)
            except (OSError, ValueError):
                raise ValueError(f"cannot read anomaly reference file: {child}") from None
            values = np.asarray(values, dtype=float)
            if values.ndim == 2 and values.shape[1] >= 2:
                references[child.stem] = reference_pair(child.stem, values[:, 0], values[:, 1])
        if not references:
            raise ValueError(f"anomaly reference directory has no usable curves: {path}")
        return references
    if path.suffix.lower() == ".npz":
        mapping = _npz_mapping(path)
        x = _find_array(mapping, ("x", "reference_x", "theta"))
        y = _find_array(mapping, ("y", "curve", "intensity", "reference_curves", "curves"))
        ids = _find_array(mapping, ("reference_ids", "reference_names", "labels", "sample_ids"))
        if x is None or y is None:
            raise ValueError(f"anomaly reference NPZ must contain x and y arrays: {path}")
        curves = np.asarray(y)
        if curves.ndim == 1:
            curves = curves[None, :]
        if curves.ndim != 2 or curves.shape[0] == 0 or curves.shape[1] < 2:
            raise ValueError(f"anomaly reference NPZ curves must be a non-empty 2-D array: {path}")
        axis = np.asarray(x, dtype=float)
        if axis.ndim == 1:
            shared_axis = axis
            per_reference_axis = None
        elif axis.ndim == 2 and axis.shape[0] in {1, curves.shape[0]}:
            shared_axis = axis[0] if axis.shape[0] == 1 else None
            per_reference_axis = axis if axis.shape[0] == curves.shape[0] else None
        else:
            raise ValueError(
                "anomaly reference NPZ x must be 1-D, one shared row, "
                f"or one row per curve; got shape {axis.shape} for {curves.shape[0]} curves"
            )
        if ids is None:
            names = _sample_ids_from_array(None, curves.shape[0])
        else:
            names = _sample_ids_from_array(ids, int(np.asarray(ids).size))
            if len(names) != curves.shape[0]:
                raise ValueError(
                    "anomaly reference NPZ reference IDs must match the number of curves"
                )
            if len(set(names)) != len(names):
                raise ValueError("anomaly reference NPZ reference IDs must be unique")
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for idx, name in enumerate(names):
            curve_axis = (
                shared_axis
                if shared_axis is not None
                else np.asarray(per_reference_axis[idx], dtype=float)
            )
            result[name] = reference_pair(name, curve_axis, curves[idx])
        return result
    rows = _read_rows(path)
    if not rows:
        try:
            values = np.asarray(np.loadtxt(path), dtype=float)
        except (OSError, ValueError):
            raise ValueError(f"cannot read anomaly reference file: {path}") from None
        if values.ndim == 2 and values.shape[1] >= 2:
            return {path.stem: reference_pair(path.stem, values[:, 0], values[:, 1])}
        raise ValueError(f"anomaly reference file needs at least two x/y columns: {path}")
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        name = _first(
            row, ("reference_id", "reference", "phase_id", "label", "sample_id"), "reference"
        )
        x = _float(_first(row, ("x", "theta", "two_theta", "angle")))
        y = _float(_first(row, ("y", "intensity", "curve", "value")))
        if x is not None and y is not None:
            grouped.setdefault(name, []).append((x, y))
    references = {
        name: reference_pair(
            name,
            np.asarray([pair[0] for pair in values]),
            np.asarray([pair[1] for pair in values]),
        )
        for name, values in grouped.items()
    }
    if not references:
        raise ValueError(f"anomaly reference file has no usable x/y rows: {path}")
    return references

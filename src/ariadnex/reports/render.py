"""Coordinate report artifacts from the public ``render_reports`` entry point."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..artifacts import ensure_unique_sanitized_ids, safe_filename
from ..config import ProjectConfig
from ..contracts import ReportArtifact, UnmixArtifact
from ..ground_truth import load_ground_truth
from .common import _bool, _first, _float, _path_value, _read_rows, _safe_report_id
from .io import _load_reference, _load_result
from .metrics import (
    _anomaly_rank_map,
    _phase_matrix,
    _write_anomaly_outputs,
    _write_anomaly_ranking_metrics,
    _write_reference_similarity,
    _write_sample_summary,
)
from .models import (
    _ANOMALY_REPORT_FILES,
    _PHASE_REPORT_FILES,
    _SAMPLE_RECONSTRUCTION_DIR,
)
from .plotting import _save_anomaly_summary, _save_phase_matrix, _save_sample_reconstructions
from .reconstruction import _rebuild_result_profiles


def _make_report_artifact(
    root: Path,
    files: Mapping[str, Path],
    metadata_path: Path,
    metadata: Mapping[str, object] | None = None,
) -> ReportArtifact:
    return ReportArtifact(
        root=root,
        files=tuple(files.values()),
        metadata_path=metadata_path,
        metadata=dict(metadata or {}),
    )


def _clear_report_family_files(output_root: Path) -> None:
    """Remove stale report files before regenerating a report directory."""

    for name in (
        *_PHASE_REPORT_FILES,
        *_ANOMALY_REPORT_FILES,
        # Retired output from reports generated before phase metrics were removed.
        "phase_recovery_metrics.csv",
        "index.html",
        "report_manifest.json",
    ):
        path = output_root / name
        if path.is_file():
            path.unlink()
    sample_dir = output_root / _SAMPLE_RECONSTRUCTION_DIR
    if sample_dir.is_symlink():
        sample_dir.unlink()
    elif sample_dir.is_dir():
        shutil.rmtree(sample_dir)


def _saved_metadata(result: UnmixArtifact) -> dict[str, object]:
    metadata = dict(result.metadata)
    if result.metadata_path.is_file():
        try:
            loaded = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            loaded = None
        if isinstance(loaded, Mapping):
            metadata.update(loaded)
    return metadata


def _aggregate_holdouts(result: UnmixArtifact) -> list[str]:
    """Return holdouts when an experiment artifact aggregates several solves."""

    metadata = _saved_metadata(result)
    values = metadata.get("holdouts")
    runs = metadata.get("runs")
    if isinstance(values, (list, tuple)) and values and isinstance(runs, Mapping):
        return [str(value) for value in values]
    if isinstance(values, (list, tuple)) and len(values) > 1:
        return [str(value) for value in values]
    holdouts = {
        value
        for row in _read_rows(result.assignments_csv)
        for value in (_first(row, ("holdout_phase_id", "holdout")),)
        if value
    }
    return sorted(holdouts)


def _is_aggregate_result(result: UnmixArtifact) -> bool:
    metadata = _saved_metadata(result)
    if "result_kind" in metadata:
        return metadata["result_kind"] == "aggregate"
    return isinstance(metadata.get("runs"), Mapping)


def _aggregate_manifest_child_roots(output_root: Path) -> list[Path]:
    """Return child roots explicitly owned by a previous aggregate report."""

    manifest_path = output_root / "report_manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return []
    if not isinstance(payload, Mapping) or payload.get("aggregate") is not True:
        return []
    reports = payload.get("reports")
    if not isinstance(reports, Mapping):
        return []

    resolved_root = output_root.expanduser().resolve()
    child_roots: list[Path] = []
    for holdout, entry in reports.items():
        if not isinstance(entry, Mapping) or not entry.get("root"):
            continue
        declared = Path(str(entry["root"])).expanduser()
        if declared.is_absolute() or len(declared.parts) != 1 or declared.name in {"", ".", ".."}:
            raise ValueError(
                "aggregate report manifest child root escapes output root: "
                f"holdout {holdout!r} -> {entry['root']!r}"
            )
        candidate = resolved_root / declared.name
        if candidate not in child_roots:
            child_roots.append(candidate)
    return child_roots


def _clear_aggregate_child_reports(output_root: Path) -> None:
    """Remove child report roots declared by the previous aggregate manifest."""

    child_roots = _aggregate_manifest_child_roots(output_root)
    selected: list[Path] = []
    for candidate in sorted(child_roots, key=lambda path: (len(path.parts), str(path))):
        if any(parent == candidate or parent in candidate.parents for parent in selected):
            continue
        selected.append(candidate)
    for child_root in selected:
        if child_root.is_symlink():
            child_root.unlink()
        elif child_root.is_dir():
            shutil.rmtree(child_root)


def _entry_path(
    entry: Mapping[str, object],
    name: str,
    local_root: Path,
    filename: str,
) -> Path:
    local = local_root / filename
    if local.is_file():
        return local
    declared = _path_value(entry.get(name))
    return declared if declared is not None and declared.is_file() else local


def _aggregate_child_result(result: UnmixArtifact, holdout: str) -> UnmixArtifact:
    """Resolve one child result from an aggregate artifact."""

    metadata = _saved_metadata(result)
    raw_entry = metadata.get("runs", {})
    entry = raw_entry.get(holdout, {}) if isinstance(raw_entry, Mapping) else {}
    if not isinstance(entry, Mapping):
        entry = {}

    local_root = result.root / f"holdout_{_safe_report_id(holdout)}"
    declared_root = _path_value(entry.get("root"))
    if local_root.is_dir():
        child_root = local_root
    elif declared_root is not None and declared_root.is_dir():
        child_root = declared_root
    else:
        child_root = local_root
    names = {
        "results_pickle": "results.pkl",
        "results_npz": "results.npz",
        "assignments_csv": "assignments.csv",
        "abundance_csv": "abundance.csv",
        "sample_summary_csv": "sample_summary.csv",
        "metadata_path": "metadata.json",
    }
    paths = {
        name: _entry_path(entry, name, child_root, filename) for name, filename in names.items()
    }
    return UnmixArtifact(root=child_root, metadata=entry, **paths)


def _write_report_index(output_root: Path, files: Mapping[str, Path], *, title: str) -> Path:
    """Write a dependency-free HTML landing page for local report browsing."""

    links: list[str] = []
    for key, path in files.items():
        try:
            relative = path.relative_to(output_root).as_posix()
        except ValueError:
            continue
        if path.suffix.lower() not in {".png", ".csv", ".npz", ".json", ".html"}:
            continue
        links.append(
            f'<li><a href="{html.escape(relative, quote=True)}">{html.escape(str(key))}</a></li>'
        )
    index_path = output_root / "index.html"
    document = "\n".join(
        (
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>{title}</title>",
            "<style>body{font:15px sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}"
            "a{color:#1769aa}li{margin:.3rem 0}</style></head>",
            f"<body><h1>{title}</h1><ul>{''.join(links)}</ul></body></html>",
        )
    )
    index_path.write_text(document + "\n", encoding="utf-8")
    return index_path


def _report_references(
    result: UnmixArtifact, explicit: object | None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load explicit references or one real leave-one-out holdout basis."""

    references = _load_reference(explicit)
    if references:
        return references
    metadata = _saved_metadata(result)
    holdout = metadata.get("holdout_phase_id")
    if not holdout and result.root.name.startswith("holdout_"):
        holdout = result.root.name.removeprefix("holdout_")
    if not holdout:
        return {}
    run_root = (
        result.root.parent.parent if result.root.parent.name == "03_unmix" else result.root.parent
    )
    candidate = run_root / "01_library" / "basis_patterns" / f"{safe_filename(str(holdout))}.npz"
    loaded = _load_reference(candidate)
    if loaded:
        return {str(holdout): next(iter(loaded.values()))}
    return {}


def _report_enabled(config: ProjectConfig, name: str) -> bool:
    value = config.reports.get(name, True)
    parsed = _bool(value)
    return True if parsed is None else parsed


def _ground_truth_value(config: ProjectConfig) -> object | None:
    return config.inputs.get("ground_truth")


def _anomaly_reference_value(config: ProjectConfig) -> object | None:
    configured = config.reports.get("anomaly_reference")
    return config.inputs.get("anomaly_reference") if configured is None else configured


def _ranking_holdouts(result: UnmixArtifact, config: ProjectConfig) -> list[str]:
    """Resolve a single leave-one-out phase for ranking metrics."""

    metadata = _saved_metadata(result)
    value = metadata.get("holdout_phase_id")
    if value:
        return [str(value)]
    values = metadata.get("holdouts")
    if isinstance(values, (list, tuple)) and len(values) == 1:
        return [str(values[0])]
    if result.root.name.startswith("holdout_"):
        return [result.root.name.removeprefix("holdout_")]
    configured = config.experiment.get("holdouts")
    if isinstance(configured, (list, tuple)) and len(configured) == 1:
        return [str(configured[0])]
    return []


def _render_aggregate_reports(
    config: ProjectConfig,
    result: UnmixArtifact,
    holdouts: Sequence[str],
    output_root: Path,
) -> ReportArtifact:
    """Render one report set per leave-one-out child result."""

    phase_recovery_enabled = _report_enabled(config, "phase_recovery")
    anomaly_enabled = _report_enabled(config, "anomaly")
    ensure_unique_sanitized_ids(holdouts, _safe_report_id, context="aggregate report output")
    _clear_aggregate_child_reports(output_root)
    _clear_report_family_files(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    index_links: dict[str, Path] = {}
    child_metadata: dict[str, object] = {}
    for holdout in holdouts:
        child = _aggregate_child_result(result, holdout)
        child_key = _safe_report_id(holdout)
        child_root = output_root / f"holdout_{child_key}"
        child_report = render_reports(config, child, child_root)
        child_files = tuple(Path(path) for path in child_report.files)
        for path in child_files:
            relative_path = path.relative_to(child_root)
            files[f"{child_key}/{relative_path.as_posix()}"] = path
        index_links[f"holdout_{child_key}"] = child_root / "index.html"
        child_metadata[holdout] = {
            "root": str(child_root.relative_to(output_root)),
            "files": [str(path.relative_to(output_root)) for path in child_files],
        }
    index_path = _write_report_index(output_root, index_links, title="AriadneX reports")
    files["index"] = index_path
    metadata_path = output_root / "report_manifest.json"
    metadata = {
        "schema_version": 1,
        "aggregate": True,
        "holdouts": list(holdouts),
        "phase_recovery": phase_recovery_enabled,
        "anomaly": anomaly_enabled,
        "reports": child_metadata,
        "files": {key: str(path.relative_to(output_root)) for key, path in files.items()},
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return _make_report_artifact(output_root, files, metadata_path, metadata)


def render_reports(
    config: ProjectConfig,
    result: UnmixArtifact,
    output_dir: Path,
) -> ReportArtifact:
    """Render reports from a completed unmix artifact and resolved config."""

    output_root = Path(output_dir).expanduser().resolve()
    holdouts = _aggregate_holdouts(result)
    if holdouts and (len(holdouts) > 1 or _is_aggregate_result(result)):
        return _render_aggregate_reports(config, result, holdouts, output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    phase_recovery_enabled = _report_enabled(config, "phase_recovery")
    anomaly_enabled = _report_enabled(config, "anomaly")
    _clear_report_family_files(output_root)
    data = _load_result(result)
    reconstruction_sigma, reconstruction_rmse, fingerprint_sigma, fingerprint_rmse, profile_grid = (
        _rebuild_result_profiles(data, config)
    )
    sigma_setting = config.reports.get("reconstruction_sigma", "auto")
    reconstruction_sigma_mode = (
        "auto"
        if sigma_setting is None or str(sigma_setting).strip().lower() == "auto"
        else "configured"
    )
    fingerprint_sigma_setting = config.reports.get("fingerprint_sigma", "auto")
    fingerprint_sigma_mode = (
        "auto"
        if fingerprint_sigma_setting is None
        or str(fingerprint_sigma_setting).strip().lower() == "auto"
        else "configured"
    )

    truth, truth_labels = load_ground_truth(_ground_truth_value(config))
    threshold = _float(config.reports.get("support_threshold_pct"), 0.5)

    sample_ids, matrix_phase_order, labels, expert, predicted = _phase_matrix(
        data, truth, truth_labels
    )
    truth_available = bool(truth)
    # Predicted entries are the MILP's selected phases, so any positive value is a selection.
    display_mask = (
        np.any(expert > threshold, axis=0) | np.any(predicted > 0.0, axis=0)
        if predicted.ndim == 2 and predicted.shape[1]
        else np.zeros(len(matrix_phase_order), dtype=bool)
    )
    display_indices = np.flatnonzero(display_mask)
    displayed_phase_ids = [matrix_phase_order[index] for index in display_indices]
    displayed_labels = [labels[index] for index in display_indices]
    displayed_expert = expert[:, display_indices] if expert.ndim == 2 else expert
    displayed_predicted = predicted[:, display_indices] if predicted.ndim == 2 else predicted
    sample_summary_path = output_root / "sample_summary.csv"
    anomaly_ranks = _anomaly_rank_map(data) if anomaly_enabled else None
    _write_sample_summary(sample_summary_path, data, anomaly_ranks=anomaly_ranks)
    matrix_png = output_root / "phase_recovery_matrix.png"
    if phase_recovery_enabled:
        _save_phase_matrix(
            matrix_png,
            sample_ids,
            displayed_labels,
            displayed_expert,
            displayed_predicted,
            truth_available=truth_available,
        )

    ranking_files: dict[str, Path] = {}
    reference_similarity = None
    similarity: dict[str, np.ndarray] = {
        "grid": np.asarray([]),
        "similarity": np.zeros((0, 0)),
    }
    anomaly_reference = _anomaly_reference_value(config)
    explicit_references = _load_reference(anomaly_reference)
    if anomaly_enabled:
        profiles_path, similarity_path, similarity = _write_anomaly_outputs(
            output_root, data, reconstruction_sigma=reconstruction_sigma
        )
        ranking_files = _write_anomaly_ranking_metrics(
            output_root,
            data,
            truth,
            truth_labels,
            sample_summary_path,
            holdouts=_ranking_holdouts(result, config),
            threshold_pct=threshold,
        )
        grid = np.asarray(similarity.get("grid", []), dtype=float)
        if explicit_references and grid.size:
            reference_similarity = _write_reference_similarity(
                output_root, data, grid, explicit_references
            )
    plot_references = _report_references(result, anomaly_reference)
    sample_reconstruction_files = _save_sample_reconstructions(
        output_root / _SAMPLE_RECONSTRUCTION_DIR,
        data,
        references=plot_references,
        truth=truth,
    )

    files: dict[str, Path] = {"sample_summary": sample_summary_path}
    files.update(sample_reconstruction_files)
    if phase_recovery_enabled:
        files["phase_recovery_matrix"] = matrix_png
    if anomaly_enabled:
        summary_png = output_root / "anomaly_summary.png"
        _save_anomaly_summary(summary_png, sample_summary_path)
        files.update(
            {
                "anomaly_profiles": profiles_path,
                "anomaly_similarity": similarity_path,
                "anomaly_summary": summary_png,
            }
        )
    files.update(ranking_files)
    if reference_similarity is not None:
        files["anomaly_reference_similarity"] = reference_similarity
    index_path = _write_report_index(output_root, files, title="AriadneX reports")
    files["index"] = index_path

    metadata_path = output_root / "report_manifest.json"
    metadata = {
        "schema_version": 1,
        "truth_available": truth_available,
        "sample_count": len(data.samples),
        "sample_reconstruction_count": len(sample_reconstruction_files),
        "phase_count": len(data.phase_order),
        "displayed_phase_count": len(displayed_phase_ids),
        "displayed_phase_ids": displayed_phase_ids,
        "phase_recovery": phase_recovery_enabled,
        "anomaly": anomaly_enabled,
        "support_threshold_pct": threshold,
        "anomaly_profiles_available": anomaly_enabled
        and bool(similarity.get("grid", np.asarray([])).size),
        "reference_similarity_available": anomaly_enabled and reference_similarity is not None,
        "anomaly_ranking_available": anomaly_enabled and bool(ranking_files),
        "profile_grid": profile_grid,
        "reconstruction_sigma_mode": reconstruction_sigma_mode,
        "reconstruction_sigma": reconstruction_sigma,
        "reconstruction_fit_rmse": reconstruction_rmse,
        "fingerprint_sigma_mode": fingerprint_sigma_mode,
        "fingerprint_sigma": fingerprint_sigma,
        "fingerprint_fit_rmse": fingerprint_rmse,
        "files": {key: str(path.relative_to(output_root)) for key, path in files.items()},
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return _make_report_artifact(output_root, files, metadata_path, metadata)


__all__ = ["render_reports"]

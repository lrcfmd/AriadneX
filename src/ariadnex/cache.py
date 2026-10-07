"""Cache input discovery and artifact reuse for workflow stages.

The workflow has four stages with different file dependencies.  Keeping that
knowledge here prevents a report-only invocation from fingerprinting the raw
inputs or the model asset used by an earlier stage.
"""

from __future__ import annotations

import csv
import os
import platform
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from ._version import __version__
from .artifacts import artifact_descriptor, cache_key, file_sha256, fingerprint_paths
from .config import ProjectConfig
from .contracts import LibraryArtifact, MixtureArtifact, ReportArtifact, UnmixArtifact
from .mixtures.load import PATTERN_PATH_ALIASES, csv_input_kind

_CIF_PATH_ALIASES = ("cif_path", "cif", "path", "source_path", "file", "filename")
_ARTIFACT_TYPES = {
    "library": LibraryArtifact,
    "mixtures": MixtureArtifact,
    "experiment": UnmixArtifact,
    "reports": ReportArtifact,
}
_SHARED_SOURCE_FILES = (
    "__init__.py",
    "_version.py",
    "artifacts.py",
    "cache.py",
    "config.py",
    "contracts.py",
    "defaults.py",
    "workflow.py",
)
_STAGE_SOURCE_PATTERNS = {
    "library": ("library/**/*.py", "runtime.py"),
    "mixtures": ("mixtures/**/*.py",),
    "experiment": ("experiments/**/*.py", "unmix/**/*.py", "mixtures/**/*.py"),
    "reports": ("reports/**/*.py", "ground_truth.py"),
}


def _rebase_payload(value: Any, old_root: Path, new_root: Path) -> Any:
    """Move run-local absolute paths when a completed run is copied."""

    if isinstance(value, Mapping):
        return {key: _rebase_payload(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase_payload(item, old_root, new_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_rebase_payload(item, old_root, new_root) for item in value)
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute() and (candidate == old_root or old_root in candidate.parents):
            return str(new_root / candidate.relative_to(old_root))
    return value


def from_descriptor(
    stage: str,
    payload: Mapping[str, Any],
    *,
    run_root: Path | None = None,
) -> Any:
    """Rebuild a path-based artifact from its status descriptor.

    ``run_root`` also lets report-only runs use a copied run directory.  Only
    paths inside the old run are rebased; external input paths retain their
    configured meaning.
    """

    data: Mapping[str, Any] = payload
    if run_root is not None:
        old_artifact_root = Path(str(payload.get("root", "")))
        if old_artifact_root.is_absolute():
            old_run_root = old_artifact_root.parent
            if old_artifact_root.name.startswith("holdout_"):
                old_run_root = old_artifact_root.parent.parent
            data = _rebase_payload(payload, old_run_root, Path(run_root))

    artifact_type = _ARTIFACT_TYPES[stage]
    allowed = {field.name for field in fields(artifact_type)}
    values = {name: data[name] for name in allowed if name in data}
    return artifact_type(**values)


def manifest_reference_paths(
    manifest: Path,
    aliases: Iterable[str],
    *,
    base_path: Path | None = None,
) -> list[Path]:
    """Resolve paths named by one CSV manifest without validating its schema.

    Validation belongs to :mod:`ariadnex.doctor`; cache discovery remains
    best-effort so malformed input reaches the stage that owns its error.
    """

    if not manifest.is_file():
        return []
    try:
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError):
        return []
    if not rows:
        return []
    columns = {
        str(key).strip().lower(): key for key in rows[0] if key is not None and str(key).strip()
    }
    reference_key = next((columns[alias] for alias in aliases if alias in columns), None)
    if reference_key is None:
        return []
    search_root = manifest.parent if base_path is None else base_path
    search_root = search_root.parent if search_root.is_file() else search_root
    paths: list[Path] = []
    for row in rows:
        raw = row.get(reference_key)
        if raw is None or not str(raw).strip():
            continue
        candidate = Path(str(raw).strip()).expanduser()
        if candidate.is_absolute():
            paths.append(candidate.resolve())
            continue
        candidates = [manifest.parent / candidate, search_root / candidate]
        basename_candidate = search_root / candidate.name
        if basename_candidate not in candidates:
            candidates.append(basename_candidate)
        paths.append(
            next((item.resolve() for item in candidates if item.is_file()), candidates[0].resolve())
        )
    return paths


def mixture_pattern_paths(config: ProjectConfig) -> list[Path]:
    """Return a sample manifest and the pattern files it names."""

    value = config.get("inputs.mixtures") or config.get("mixtures")
    if value is None:
        return []
    manifest = Path(str(value)).expanduser()
    if not manifest.is_absolute():
        manifest = config.base_dir / manifest
    manifest = manifest.resolve()
    if manifest.suffix.lower() != ".csv" or not manifest.is_file():
        return [manifest]
    try:
        if csv_input_kind(manifest) == "pattern":
            return [manifest]
    except (OSError, UnicodeError, ValueError, csv.Error):
        # Cache discovery is best-effort; the owning mixture stage reports
        # malformed or unreadable input with the useful error message.
        return [manifest]
    return [manifest, *manifest_reference_paths(manifest, PATTERN_PATH_ALIASES)]


def phase_cif_paths(config: ProjectConfig) -> list[Path]:
    """Return phase manifests and the CIF files they name."""

    input_value = (
        config.get("inputs.cifs")
        or config.get("cifs")
        or config.get("input_path")
        or config.get("library.input_path")
    )
    input_path = None if input_value is None else Path(str(input_value)).expanduser()
    if input_path is not None and not input_path.is_absolute():
        input_path = (config.base_dir / input_path).resolve()
    phase_value = (
        config.get("inputs.phase_manifest")
        or config.get("phase_manifest")
        or config.get("library.phase_manifest")
    )
    paths: list[Path] = []
    if input_path is not None:
        paths.append(input_path.resolve())
    if phase_value is not None:
        phase_manifest = Path(str(phase_value)).expanduser()
        if not phase_manifest.is_absolute():
            phase_manifest = config.base_dir / phase_manifest
        phase_manifest = phase_manifest.resolve()
        paths.extend(
            [
                phase_manifest,
                *manifest_reference_paths(phase_manifest, _CIF_PATH_ALIASES, base_path=input_path),
            ]
        )
    if input_path is not None and input_path.is_file() and input_path.suffix.lower() == ".csv":
        paths.extend(
            [
                input_path,
                *manifest_reference_paths(input_path, _CIF_PATH_ALIASES, base_path=input_path),
            ]
        )
    return paths


def _config_paths(config: ProjectConfig, *keys: str) -> list[Path]:
    paths: list[Path] = []
    for key in keys:
        value = config.get(key)
        if value is None or str(value).strip() == "":
            continue
        if isinstance(value, (str, Path)):
            paths.append(Path(str(value)).expanduser())
        elif isinstance(value, Mapping):
            for alias in (
                "path",
                "csv_path",
                "reference_path",
                "file",
                "filename",
                "directory",
                "dir",
            ):
                child = value.get(alias)
                if isinstance(child, (str, Path)) and str(child).strip():
                    paths.append(Path(str(child)).expanduser())
    return paths


def _all_configured_paths(config: ProjectConfig) -> list[Path]:
    """Collect configured paths for run provenance, including the model once."""

    paths: list[Path] = []
    output_keys = {"output_root", "output_dir", "output_folder"}

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)
        elif isinstance(value, str):
            lowered = key.lower()
            if lowered in output_keys or lowered.startswith("output_"):
                return
            if lowered.endswith(("_path", "_dir", "_folder", "_file", "_manifest")) or lowered in {
                "cifs",
                "mixtures",
                "instrument",
                "checkpoint",
                "ground_truth",
                "phase_manifest",
                "raw_cifs",
                "mixed_patterns",
                "basis_patterns",
                "manifest",
                "data_root",
            }:
                paths.append(Path(value))

    visit(config.resolved)
    paths.extend(mixture_pattern_paths(config))
    paths.extend(phase_cif_paths(config))
    return paths


def _artifact_file_paths(artifact: Any) -> list[Path]:
    """Return files consumed from an upstream artifact without walking roots."""

    paths: list[Path] = []
    for field in fields(artifact):
        if field.name == "root":
            continue
        value = getattr(artifact, field.name)
        if isinstance(value, Path):
            paths.append(value)
        elif isinstance(value, (tuple, list)) and all(isinstance(item, Path) for item in value):
            paths.extend(value)
    if isinstance(artifact, UnmixArtifact):
        metadata = artifact.metadata
        runs = metadata.get("runs") if isinstance(metadata, Mapping) else None
        if isinstance(runs, Mapping):
            for descriptor in runs.values():
                if not isinstance(descriptor, Mapping):
                    continue
                for name in (
                    "results_pickle",
                    "results_npz",
                    "assignments_csv",
                    "abundance_csv",
                    "sample_summary_csv",
                    "metadata_path",
                ):
                    value = descriptor.get(name)
                    if value is not None and str(value).strip():
                        paths.append(Path(str(value)))
    return paths


def _report_roots(artifact: Any) -> list[Path]:
    roots: list[Path] = []
    root_value = getattr(artifact, "root", None)
    if root_value is not None:
        roots.append(Path(root_value).expanduser().resolve())
    metadata = getattr(artifact, "metadata", {})
    runs = metadata.get("runs") if isinstance(metadata, Mapping) else None
    if isinstance(runs, Mapping):
        for descriptor in runs.values():
            if not isinstance(descriptor, Mapping) or descriptor.get("root") is None:
                continue
            roots.append(Path(str(descriptor["root"])).expanduser().resolve())
    return list(dict.fromkeys(roots))


def _basis_pattern_paths(manifest: Path, basis_dir: Path) -> list[Path]:
    """Read explicit basis curve paths from an existing representatives CSV."""

    if not manifest.is_file():
        return []
    try:
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            paths: list[Path] = []
            for row in rows:
                values = {
                    str(key).strip().lower(): value for key, value in row.items() if key is not None
                }
                raw = next(
                    (
                        values.get(name)
                        for name in (
                            "basis_pattern_path",
                            "pattern_path",
                            "basis_path",
                            "expected_basis_pattern",
                            "basis_simulated_pattern_path",
                            "simulated_pattern_path",
                        )
                        if values.get(name) and str(values[name]).strip()
                    ),
                    None,
                )
                if raw is None:
                    continue
                candidate = Path(str(raw).strip()).expanduser()
                if not candidate.is_absolute():
                    candidate = basis_dir / candidate
                paths.append(candidate.resolve())
            return paths
    except (OSError, csv.Error, UnicodeError):
        return []


def _report_dependency_paths(config: ProjectConfig, artifact: Any) -> list[Path]:
    paths = _artifact_file_paths(artifact)
    for root in _report_roots(artifact):
        run_root = root.parent
        if root.name.startswith("holdout_"):
            run_root = root.parent.parent
        basis_dir = run_root / "01_library" / "basis_patterns"
        for relative in (
            ("02_mixtures", "patterns.npz"),
            ("02_mixtures", "signatures.npz"),
            ("02_mixtures", "manifest.csv"),
            ("02_mixtures", "metadata.json"),
            ("01_library", "representatives.csv"),
            ("01_library", "metadata.json"),
        ):
            paths.append(run_root.joinpath(*relative))
        paths.append(root / "library_view" / "representatives.csv")
        paths.append(root / "library_view" / "metadata.json")
        paths.extend(
            _basis_pattern_paths(run_root / "01_library" / "representatives.csv", basis_dir)
        )
        paths.extend(_basis_pattern_paths(root / "library_view" / "representatives.csv", basis_dir))
        if root.name.startswith("holdout_"):
            view_root = root.parent / "library_views" / root.name.removeprefix("holdout_")
            paths.extend((view_root / "representatives.csv", view_root / "metadata.json"))
            paths.extend(_basis_pattern_paths(view_root / "representatives.csv", basis_dir))

    paths.extend(
        _config_paths(
            config,
            "inputs.ground_truth",
            "ground_truth",
            "reports.anomaly_reference",
            "inputs.anomaly_reference",
        )
    )
    return paths


def stage_input_paths(
    stage: str,
    config: ProjectConfig,
    *upstream: Any,
) -> list[Path]:
    """Return the input files fingerprinted for one stage."""

    if stage == "library":
        paths = phase_cif_paths(config)
        paths.extend(
            _config_paths(
                config,
                "inputs.phase_manifest",
                "phase_manifest",
                "library.simulation.instrument",
                "library.simulation.instrument_path",
                "library.basis_simulation.instrument",
                "library.basis_simulation.instrument_path",
                "inputs.instrument",
            )
        )
        paths.extend(_config_paths(config, "model.checkpoint", "model.checkpoint_path"))
    elif stage == "mixtures":
        paths = mixture_pattern_paths(config)
        paths.extend(_config_paths(config, "inputs.instrument", "inputs.mixtures", "mixtures"))
    elif stage == "experiment":
        paths = []
        for artifact in upstream:
            paths.extend(_artifact_file_paths(artifact))
        paths.extend(
            _config_paths(
                config,
                "inputs.instrument",
                "unmix.solver.phase_order_manifest",
                "unmix.element_prior.sample_description_path",
            )
        )
    elif stage == "reports":
        paths = []
        if upstream:
            paths.extend(_report_dependency_paths(config, upstream[0]))
        else:
            paths.extend(_config_paths(config, "inputs.ground_truth", "reports.anomaly_reference"))
    else:
        raise KeyError(f"unknown workflow stage: {stage}")
    return paths


def source_tree_fingerprint(stage: str) -> dict[str, str]:
    """Fingerprint the implementation used by one stage."""

    source_root = Path(__file__).resolve().parent
    patterns = (*_SHARED_SOURCE_FILES, *_STAGE_SOURCE_PATTERNS[stage])
    files = {path for pattern in patterns for path in source_root.glob(pattern)}
    return {path.relative_to(source_root).as_posix(): file_sha256(path) for path in sorted(files)}


def stage_cache_key(stage: str, config: ProjectConfig, *upstream: Any) -> str:
    paths = stage_input_paths(stage, config, *upstream)
    payload = {
        "version": __version__,
        "config": config.resolved,
        "upstream": [artifact_descriptor(item) for item in upstream],
        "source_tree": source_tree_fingerprint(stage),
    }
    return cache_key(stage, payload, paths)


def artifact_paths_exist(artifact: Any) -> bool:
    """Return whether every path in an artifact contract is still present."""

    for field in fields(artifact):
        value = getattr(artifact, field.name)
        if isinstance(value, Path):
            expect_directory = field.name == "root" or field.name.endswith("_dir")
            if expect_directory and not value.is_dir():
                return False
            if not expect_directory and not value.is_file():
                return False
        elif isinstance(value, (tuple, list)) and all(isinstance(item, Path) for item in value):
            if any(not item.is_file() for item in value):
                return False
    if isinstance(artifact, LibraryArtifact) and not _library_basis_paths_exist(artifact):
        return False
    if isinstance(artifact, UnmixArtifact) and not _aggregate_unmix_paths_exist(artifact):
        return False
    return True


def _aggregate_unmix_paths_exist(artifact: UnmixArtifact) -> bool:
    metadata = artifact.metadata
    if not isinstance(metadata, Mapping):
        return True
    holdouts = metadata.get("holdouts")
    is_aggregate = "runs" in metadata or (
        metadata.get("experiment_type") == "leave_one_out"
        and isinstance(holdouts, (list, tuple))
        and len(holdouts) > 1
    )
    if not is_aggregate:
        return True
    runs = metadata.get("runs")
    if not isinstance(runs, Mapping) or not runs:
        return False
    for descriptor in runs.values():
        if not isinstance(descriptor, Mapping):
            return False
        root_value = descriptor.get("root")
        if not isinstance(root_value, (str, Path)) or not str(root_value).strip():
            return False
        child_root = Path(root_value)
        if not child_root.is_dir():
            return False
        for field_name in ("metadata_path", "results_pickle", "results_npz"):
            value = descriptor.get(field_name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                return False
            if not Path(value).is_file():
                return False
        for field_name, filename in (
            ("assignments_csv", "assignments.csv"),
            ("abundance_csv", "abundance.csv"),
            ("sample_summary_csv", "sample_summary.csv"),
        ):
            value = descriptor.get(field_name, child_root / filename)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                return False
            if not Path(value).is_file():
                return False
    return True


def _library_basis_paths_exist(artifact: LibraryArtifact) -> bool:
    try:
        with artifact.representatives_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            for row in rows:
                normalised = {
                    str(key).strip().lower(): item for key, item in row.items() if key is not None
                }
                raw_path = next(
                    (
                        normalised.get(name)
                        for name in (
                            "basis_pattern_path",
                            "pattern_path",
                            "basis_path",
                            "expected_basis_pattern",
                        )
                        if normalised.get(name)
                    ),
                    None,
                )
                if raw_path is None:
                    continue
                path = Path(str(raw_path).strip()).expanduser()
                if not path.is_absolute():
                    path = artifact.basis_patterns_dir / path
                if not path.is_file():
                    return False
    except (OSError, csv.Error, UnicodeError):
        return False
    return True


def cached_artifact(store: Any, stage: str, key: str) -> Any | None:
    status = store.stage_status(stage)
    if status is None or status.state != "completed" or status.cache_key != key:
        return None
    entry = store.read_status().get("stages", {}).get(stage, {})
    descriptor = entry.get("artifact") if isinstance(entry, Mapping) else None
    if not isinstance(descriptor, Mapping):
        return None
    try:
        artifact = from_descriptor(stage, descriptor)
    except (KeyError, TypeError, ValueError):
        return None
    if not artifact_paths_exist(artifact):
        return None
    return artifact


def portable_for_provenance(value: Any, base_dir: Path) -> Any:
    """Keep provenance relocatable while retaining useful relative paths."""

    if isinstance(value, Mapping):
        return {
            str(portable_for_provenance(str(key), base_dir)): portable_for_provenance(
                item, base_dir
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [portable_for_provenance(item, base_dir) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        relative = os.path.relpath(value, base_dir)
        parts = {part.lower() for part in Path(relative).parts}
        if "home" in parts or "users" in parts:
            return f"<external>/{Path(value).name}"
        return relative
    return value


def initial_provenance(config: ProjectConfig, run_id: str) -> dict[str, Any]:
    source_commit: str | None = None
    for candidate in Path(__file__).resolve().parents:
        if not (candidate / ".git").exists():
            continue
        try:
            source_commit = (
                subprocess.run(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                or None
            )
        except (OSError, subprocess.CalledProcessError):
            source_commit = None
        break
    paths = _all_configured_paths(config)
    return {
        "schema_version": 1,
        "workflow_version": __version__,
        "run_id": run_id,
        "config_path": portable_for_provenance(str(config.source_path), config.base_dir),
        "config_hash": cache_key("config", config.resolved),
        "resolved_config": portable_for_provenance(config.as_dict(), config.base_dir),
        "source_commit": source_commit,
        "input_fingerprints": portable_for_provenance(fingerprint_paths(paths), config.base_dir),
        "python": platform.python_version(),
    }


__all__ = [
    "artifact_paths_exist",
    "cached_artifact",
    "from_descriptor",
    "initial_provenance",
    "manifest_reference_paths",
    "mixture_pattern_paths",
    "phase_cif_paths",
    "portable_for_provenance",
    "source_tree_fingerprint",
    "stage_cache_key",
    "stage_input_paths",
]

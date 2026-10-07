"""Top-level orchestration and stage seams for AriadneX."""

from __future__ import annotations

import re
import sys
import traceback
import warnings
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from . import cache
from .artifacts import ArtifactStore, cache_key, ensure_managed_output_directory
from .config import ProjectConfig, config_for_run, load_config
from .contracts import (
    LibraryArtifact,
    MixtureArtifact,
    ReportArtifact,
    RunArtifact,
    StageStatus,
    UnmixArtifact,
)
from .doctor import DoctorCheck, DoctorReport, doctor, report_preflight


class WorkflowError(RuntimeError):
    """Raised when a workflow stage cannot complete."""


_ARTIFACT_TYPES = {
    "library": LibraryArtifact,
    "mixtures": MixtureArtifact,
    "experiment": UnmixArtifact,
    "reports": ReportArtifact,
}


def build_library(config: ProjectConfig, output_dir: Path) -> LibraryArtifact:
    """Build a library through the public library module seam."""

    from .library import build_library as _build_library

    return _build_library(config, output_dir)


def prepare_mixtures(config: ProjectConfig, output_dir: Path) -> MixtureArtifact:
    """Prepare mixtures through the public mixture module seam."""

    from .mixtures import prepare_mixtures as _prepare_mixtures

    return _prepare_mixtures(config, output_dir)


def run_experiment(
    config: ProjectConfig,
    library: LibraryArtifact,
    mixtures: MixtureArtifact,
    output_dir: Path,
) -> UnmixArtifact:
    """Run standard or leave-one-out unmixing through its public seam."""

    from .experiments import run_experiment as _run_experiment

    return _run_experiment(config, library, mixtures, output_dir)


def render_reports(
    config: ProjectConfig, result: UnmixArtifact, output_dir: Path
) -> ReportArtifact:
    """Render reports from a completed numerical result."""

    from .reports import render_reports as _render_reports

    return _render_reports(config, result, output_dir)


def _sanitise(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "run"


def _progress(message: str) -> None:
    print(f"[workflow] {message}", file=sys.stderr, flush=True)


def _log_path(store: ArtifactStore, stage: str) -> Path:
    log_dir = ensure_managed_output_directory(
        store.run_root / "logs", context="workflow diagnostic logs"
    )
    return log_dir / f"{_sanitise(stage)}.log"


def _failure_log(store: ArtifactStore, stage: str, exc: BaseException) -> Path | None:
    """Keep a detailed local traceback while exposing a short CLI error."""

    try:
        path = _log_path(store, stage)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nerror: {exc!r}\n\n{traceback.format_exc()}")
        return path
    except (OSError, TypeError, ValueError):
        return None


def _stage_log_path(store: ArtifactStore, stage: str) -> Path | None:
    try:
        path = _log_path(store, stage)
        path.write_text(f"stage: {stage}\n", encoding="utf-8")
        return path
    except (OSError, TypeError, ValueError):
        return None


def _with_details(message: str, log_path: Path | None) -> str:
    return f"{message}; details: {log_path}" if log_path is not None else message


def _workflow_error(exc: BaseException, message: str, log_path: Path | None) -> WorkflowError:
    """Keep a WorkflowError's own message and describe other failures with ``message``."""

    if isinstance(exc, WorkflowError):
        message = str(exc)
    return WorkflowError(_with_details(message, log_path))


@contextmanager
def _capture_stage_output(log_handle: Any):
    if log_handle is None:
        yield
        return
    with redirect_stdout(log_handle), warnings.catch_warnings():

        def showwarning(
            message: warnings.WarningMessage | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any = None,
            line: str | None = None,
        ) -> None:
            log_handle.write(warnings.formatwarning(message, category, filename, lineno, line))
            log_handle.flush()

        warnings.showwarning = showwarning
        yield


def _preflight_error(report: DoctorReport) -> str:
    failures = [f"{check.name}: {check.detail}" for check in report.checks if not check.ok]
    return "; ".join(failures) or "input preflight failed"


def _run_stage(
    store: ArtifactStore,
    stage: str,
    key: str,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    force: bool,
) -> Any:
    if not force:
        cached = cache.cached_artifact(store, stage, key)
        if cached is not None:
            _progress(f"{stage}: using cached results")
            return cached

    _progress(f"{stage}: running")
    store.invalidate_after(stage)
    started = StageStatus.running(stage, cache_key=key)
    store.write_stage_status(started)
    log_path = _stage_log_path(store, stage)
    log_handle = log_path.open("a", encoding="utf-8") if log_path is not None else None
    try:
        with _capture_stage_output(log_handle):
            result = function(*args)
            expected = _ARTIFACT_TYPES[stage]
            if not isinstance(result, expected):
                raise TypeError(
                    f"stage '{stage}' returned {type(result).__name__}; expected {expected.__name__}"
                )
    except Exception as exc:
        if log_handle is not None:
            log_handle.flush()
            log_handle.close()
            log_handle = None
        log_path = _failure_log(store, stage, exc) or log_path
        detail = str(exc).strip() or exc.__class__.__name__
        store.write_stage_status(
            StageStatus.failed(
                stage,
                _with_details(detail, log_path),
                cache_key=key,
                started_at=started.started_at,
            )
        )
        _progress(_with_details(f"{stage}: failed ({detail})", log_path))
        raise _workflow_error(exc, f"workflow stage '{stage}' failed: {detail}", log_path) from exc
    finally:
        if log_handle is not None:
            log_handle.close()

    store.write_stage_status(
        StageStatus.completed(
            stage, cache_key=key, artifact_path=result.root, started_at=started.started_at
        ),
        artifact=result,
    )
    _progress(f"{stage}: complete")
    return result


def _write_resolved_config(config: ProjectConfig, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(config_for_run(config, path.parent), sort_keys=False), encoding="utf-8"
    )


def _record_stage_provenance(store: ArtifactStore, config: ProjectConfig) -> None:
    provenance = store.read_provenance()
    stages = store.read_status().get("stages", {})
    provenance["stages"] = cache.portable_for_provenance(stages, config.base_dir)
    store.write_provenance(provenance)


def run_project(config_path: str | Path, *, force: bool = False) -> RunArtifact:
    """Run all four stages and return their standard artifacts."""

    config = load_config(config_path)
    run_hash = cache_key("run", config.resolved)[:12]
    run_id = f"{_sanitise(config.project_name)}-{run_hash}"
    run_root = config.output_root / _sanitise(config.project_name) / run_id
    store = ArtifactStore(run_root)
    store.ensure_layout()
    try:
        preflight = doctor(config)
        if not preflight.ok:
            raise WorkflowError(_preflight_error(preflight))
    except Exception as exc:
        log_path = _failure_log(store, "preflight", exc)
        detail = str(exc).strip() or exc.__class__.__name__
        _progress(_with_details(f"preflight: failed ({detail})", log_path))
        raise _workflow_error(exc, f"workflow preflight failed: {detail}", log_path) from exc

    _write_resolved_config(config, run_root / "resolved_config.yaml")
    store.write_provenance(cache.initial_provenance(config, run_id))

    library = _run_stage(
        store,
        "library",
        cache.stage_cache_key("library", config),
        build_library,
        (config, store.stage_dir("library")),
        force=force,
    )
    mixtures = _run_stage(
        store,
        "mixtures",
        cache.stage_cache_key("mixtures", config),
        prepare_mixtures,
        (config, store.stage_dir("mixtures")),
        force=force,
    )
    unmix = _run_stage(
        store,
        "experiment",
        cache.stage_cache_key("experiment", config, library, mixtures),
        run_experiment,
        (config, library, mixtures, store.stage_dir("experiment")),
        force=force,
    )
    reports = _run_stage(
        store,
        "reports",
        cache.stage_cache_key("reports", config, unmix),
        render_reports,
        (config, unmix, store.stage_dir("reports")),
        force=force,
    )
    _record_stage_provenance(store, config)
    return RunArtifact(
        root=run_root,
        status_path=store.status_path,
        provenance_path=store.provenance_path,
        library=library,
        mixtures=mixtures,
        unmix=unmix,
        reports=reports,
        run_id=run_id,
        config_path=config.source_path,
    )


def report_run(
    run_root: str | Path,
    *,
    config_path: str | Path | None = None,
    force: bool = False,
) -> ReportArtifact:
    """Render reports for an existing run without rerunning numerical stages."""

    root = Path(run_root).expanduser().resolve()
    store = ArtifactStore(root)
    status = store.read_status()
    experiment_entry = status.get("stages", {}).get("experiment", {})
    descriptor = experiment_entry.get("artifact") if isinstance(experiment_entry, Mapping) else None
    if not isinstance(descriptor, Mapping) or experiment_entry.get("state") != "completed":
        raise WorkflowError(f"run has no completed experiment artifact: {root}")
    try:
        unmix = cache.from_descriptor("experiment", descriptor, run_root=root)
    except (KeyError, TypeError, ValueError) as exc:
        log_path = _failure_log(store, "report", exc)
        message = _with_details(f"run has an invalid experiment artifact: {exc}", log_path)
        _progress(f"report: failed ({message})")
        raise WorkflowError(message) from exc
    config_file = (
        Path(config_path).expanduser() if config_path is not None else root / "resolved_config.yaml"
    )
    try:
        config = load_config(config_file)
    except (OSError, ValueError) as exc:
        log_path = _failure_log(store, "report", exc)
        message = _with_details(f"could not load report configuration: {exc}", log_path)
        _progress(f"report: failed ({message})")
        raise WorkflowError(message) from exc
    preflight = report_preflight(config)
    if not preflight.ok:
        exc = WorkflowError(_preflight_error(preflight))
        log_path = _failure_log(store, "report-preflight", exc)
        message = _with_details(str(exc), log_path)
        _progress(f"report preflight: failed ({message})")
        raise WorkflowError(message) from exc
    reports = _run_stage(
        store,
        "reports",
        cache.stage_cache_key("reports", config, unmix),
        render_reports,
        (config, unmix, store.stage_dir("reports")),
        force=force,
    )
    _record_stage_provenance(store, config)
    return reports


__all__ = [
    "WorkflowError",
    "build_library",
    "prepare_mixtures",
    "run_experiment",
    "render_reports",
    "run_project",
    "report_run",
    "DoctorCheck",
    "DoctorReport",
    "doctor",
]

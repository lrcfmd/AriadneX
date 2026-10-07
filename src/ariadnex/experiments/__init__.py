"""Standard and non-destructive leave-one-out experiment execution."""

from __future__ import annotations

import csv
import json
import pickle
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ariadnex.artifacts import ensure_unique_sanitized_ids
from ariadnex.config import ProjectConfig
from ariadnex.contracts import LibraryArtifact, MixtureArtifact, UnmixArtifact


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError(f"invalid phase ID: {value!r}")
    return safe


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_library_view(
    library: LibraryArtifact,
    excluded_phase_ids: set[str] | frozenset[str],
    output_dir: str | Path,
) -> LibraryArtifact:
    """Create a manifest-only view while retaining the original basis directory.

    The solver must load the patterns named by ``representatives_path``.  No CIF
    or PXRD file is copied, linked, deleted, or hidden on disk.
    """

    excluded = {str(value) for value in excluded_phase_ids}
    if not excluded:
        raise ValueError("a library view requires at least one excluded phase ID")
    fields, rows = _read_csv(library.representatives_path)
    if "phase_id" not in fields:
        raise ValueError(
            f"representatives manifest needs a phase_id column: {library.representatives_path}"
        )
    available = {row["phase_id"] for row in rows}
    missing = sorted(excluded - available)
    if missing:
        raise ValueError(f"holdout phase IDs are absent from the library: {', '.join(missing)}")
    selected = [row for row in rows if row["phase_id"] not in excluded]
    if not selected:
        raise ValueError("leave-one-out view would remove the entire library")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    representatives_path = root / "representatives.csv"
    _write_csv(representatives_path, fields, selected)
    metadata = dict(library.metadata)
    metadata.update(
        {
            "view_type": "excluded_phases",
            "excluded_phase_ids": sorted(excluded),
            "source_library_root": str(library.root),
            "n_representatives": len(selected),
        }
    )
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return LibraryArtifact(
        root=root,
        manifest_path=library.manifest_path,
        representatives_path=representatives_path,
        basis_patterns_dir=library.basis_patterns_dir,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def _default_runner() -> Callable[..., UnmixArtifact]:
    from ariadnex.unmix import run_unmix

    return run_unmix


def _with_experiment_metadata(
    result: UnmixArtifact,
    *,
    experiment_type: str,
    holdouts: list[str],
) -> UnmixArtifact:
    metadata = dict(result.metadata)
    metadata.update(
        {
            "experiment_type": experiment_type,
            "result_kind": "single",
            "holdouts": list(holdouts),
        }
    )
    if len(holdouts) == 1:
        metadata["holdout_phase_id"] = holdouts[0]
    result.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    result.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return replace(result, metadata=metadata)


def _aggregate_csv(
    artifacts: list[tuple[str, UnmixArtifact]],
    attribute: str,
    destination: Path,
) -> None:
    fieldnames: list[str] = ["holdout_phase_id"]
    combined: list[dict[str, Any]] = []
    for holdout, artifact in artifacts:
        source = Path(getattr(artifact, attribute))
        fields, rows = _read_csv(source)
        for field in fields:
            if field not in fieldnames:
                fieldnames.append(field)
        for row in rows:
            combined.append({"holdout_phase_id": holdout, **row})
    _write_csv(destination, fieldnames, combined)


def _aggregate_results(
    artifacts: list[tuple[str, UnmixArtifact]],
    output_dir: Path,
) -> UnmixArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = output_dir / "assignments.csv"
    abundance = output_dir / "abundance.csv"
    summary = output_dir / "sample_summary.csv"
    _aggregate_csv(artifacts, "assignments_csv", assignments)
    _aggregate_csv(artifacts, "abundance_csv", abundance)
    _aggregate_csv(artifacts, "sample_summary_csv", summary)

    index = {
        holdout: {
            "root": str(artifact.root),
            "results_pickle": str(artifact.results_pickle),
            "results_npz": str(artifact.results_npz),
            "metadata_path": str(artifact.metadata_path),
        }
        for holdout, artifact in artifacts
    }
    results_pickle = output_dir / "results.pkl"
    with results_pickle.open("wb") as handle:
        pickle.dump(index, handle)

    results_npz = output_dir / "results.npz"
    np.savez_compressed(
        results_npz,
        holdout_phase_ids=np.asarray([item[0] for item in artifacts], dtype=str),
        result_roots=np.asarray([str(item[1].root) for item in artifacts], dtype=str),
    )
    metadata = {
        "schema_version": 1,
        "experiment_type": "leave_one_out",
        "result_kind": "aggregate",
        "holdouts": [item[0] for item in artifacts],
        "runs": index,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return UnmixArtifact(
        root=output_dir,
        results_pickle=results_pickle,
        results_npz=results_npz,
        assignments_csv=assignments,
        abundance_csv=abundance,
        sample_summary_csv=summary,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def run_experiment(
    config: ProjectConfig,
    library: LibraryArtifact,
    mixtures: MixtureArtifact,
    output_dir: str | Path,
    *,
    runner: Callable[..., UnmixArtifact] | None = None,
) -> UnmixArtifact:
    """Run a standard solve or one independent solve per held-out phase."""

    solve = runner or _default_runner()
    root = Path(output_dir)
    experiment = config.experiment
    experiment_type = str(experiment.get("type", "standard"))
    if experiment_type == "standard":
        result = solve(config, library, mixtures, root)
        return _with_experiment_metadata(result, experiment_type="standard", holdouts=[])
    if experiment_type != "leave_one_out":
        raise ValueError(f"unsupported experiment type: {experiment_type}")

    raw_holdouts = experiment.get("holdouts")
    if not isinstance(raw_holdouts, (list, tuple)) or not raw_holdouts:
        raise ValueError("leave_one_out requires a non-empty experiment.holdouts list")
    holdouts = [str(value) for value in raw_holdouts]
    if len(set(holdouts)) != len(holdouts):
        raise ValueError("experiment.holdouts contains duplicate phase IDs")
    ensure_unique_sanitized_ids(
        holdouts,
        _safe_id,
        context="leave-one-out output",
    )

    if len(holdouts) == 1:
        holdout = holdouts[0]
        print(f"[experiment] holdout: {holdout}", file=sys.stderr, flush=True)
        view = create_library_view(library, {holdout}, root / "library_view")
        view_metadata = dict(view.metadata)
        view_metadata["holdout_phase_id"] = holdout
        view = replace(view, metadata=view_metadata)
        result = solve(config, view, mixtures, root)
        return _with_experiment_metadata(
            result,
            experiment_type="leave_one_out",
            holdouts=holdouts,
        )

    artifacts: list[tuple[str, UnmixArtifact]] = []
    for index, holdout in enumerate(holdouts, 1):
        print(
            f"[experiment] holdout {index}/{len(holdouts)}: {holdout}",
            file=sys.stderr,
            flush=True,
        )
        safe = _safe_id(holdout)
        view = create_library_view(library, {holdout}, root / "library_views" / safe)
        view_metadata = dict(view.metadata)
        view_metadata["holdout_phase_id"] = holdout
        view = replace(view, metadata=view_metadata)
        result = solve(config, view, mixtures, root / f"holdout_{safe}")
        artifacts.append(
            (
                holdout,
                _with_experiment_metadata(
                    result,
                    experiment_type="leave_one_out",
                    holdouts=[holdout],
                ),
            )
        )
    return _aggregate_results(artifacts, root)


__all__ = ["create_library_view", "run_experiment"]

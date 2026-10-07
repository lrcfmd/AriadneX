"""Public CPU/MILP unmixing seam.

run_unmix consumes path-based library and mixture artifacts, coordinates
configuration, solving, and persistence, and returns one UnmixArtifact.
Input decoding, numerical sample solving, and output serialization live in
small internal modules so this public seam remains easy to follow.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ariadnex.config import ProjectConfig
from ariadnex.contracts import LibraryArtifact, MixtureArtifact, UnmixArtifact
from ariadnex.mixtures.pipeline import load_cleaned_mixtures

from .export import _write_outputs
from .io import _apply_phase_order_manifest, _load_library
from .solver import (
    _prior_for_samples,
    _solve_one,
    _solve_one_worker,
    _unmix_options,
)


def _resolve_output_dir(config: ProjectConfig, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    configured = config.get("project.output_root") or config.get("output_root") or "runs/03_unmix"
    base = config.base_dir
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if path.name != "03_unmix":
        path /= "03_unmix"
    return path


def _report_sample_progress(completed: int, total: int, sample: dict[str, Any]) -> None:
    sample_id = str(sample["sample_id"])
    print(f"[unmix] {completed}/{total} samples complete: {sample_id}", file=sys.stderr, flush=True)


def run_unmix(
    config: ProjectConfig,
    library: LibraryArtifact,
    mixtures: MixtureArtifact,
    output_dir: str | Path | None = None,
) -> UnmixArtifact:
    """Run CPU/MILP anomaly-aware unmixing and return UnmixArtifact."""

    root = _resolve_output_dir(config, output_dir)
    root.mkdir(parents=True, exist_ok=True)
    options = _unmix_options(config)
    library_records = _load_library(
        library,
        instrument=options["instrument"],
        signature_points=options["signature_points"],
        preprocessing=options["basis_preprocessing"],
    )
    if options["phase_order_manifest"] not in (None, ""):
        library_records = _apply_phase_order_manifest(
            config, library_records, options["phase_order_manifest"]
        )
    samples = load_cleaned_mixtures(mixtures)
    if not samples:
        raise ValueError("Mixture artifact contains no samples")
    sample_ids = [str(sample.get("sample_id", "")).strip() for sample in samples]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("Mixture artifact contains a blank sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Mixture artifact contains duplicate sample_id values")
    prior_map = _prior_for_samples(config, samples, options)
    workers_value = options["threads"]
    try:
        workers = int(workers_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unmix worker count must be a positive integer, got {workers_value!r}"
        ) from exc
    if workers < 1:
        raise ValueError(f"unmix worker count must be a positive integer, got {workers_value!r}")
    solved = []
    if workers > 1 and len(samples) > 1:
        # executor.map preserves input order, matching the old MPProgressMap
        # contract. Gurobi uses its own thread setting unless the user supplies
        # solver.gurobi_threads.
        arguments = ((sample, library_records, options, prior_map) for sample in samples)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for completed, (sample, result) in enumerate(
                zip(samples, executor.map(_solve_one_worker, arguments)), 1
            ):
                solved.append(result)
                _report_sample_progress(completed, len(samples), sample)
    else:
        for completed, sample in enumerate(samples, 1):
            solved.append(_solve_one(sample, library_records, options, prior_map))
            _report_sample_progress(completed, len(samples), sample)
    metadata_path = root / "metadata.json"
    paths = _write_outputs(root, library_records, samples, solved, options, metadata_path)
    results_pickle, results_npz, assignments_csv, abundance_csv, summary_csv, metadata = paths
    return UnmixArtifact(
        root=root,
        results_pickle=results_pickle,
        results_npz=results_npz,
        assignments_csv=assignments_csv,
        abundance_csv=abundance_csv,
        sample_summary_csv=summary_csv,
        metadata_path=metadata_path,
        metadata=metadata,
    )


__all__ = ["run_unmix"]

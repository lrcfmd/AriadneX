"""Library-construction workflow and its public ``build_library`` seam."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..artifacts import ensure_managed_output_directory, ensure_unique_sanitized_ids, safe_filename
from ..config import ProjectConfig
from ..contracts import LibraryArtifact
from .adapters import call_embedder, call_simulator
from .candidates import apply_deduplication, prepare_candidates
from .clustering import cluster_embeddings
from .config import resolve_library_settings
from .embedding import PxrdEmbeddingAdapter
from .export import (
    LibraryExportContext,
    export_library,
    persist_simulated_patterns,
    write_library_manifest,
)
from .manifest import CifEntry, discover_cifs
from .selection import DEFAULT_MIN_ARI
from .simulation import GsasIISimulator
from .validation import DefaultCifValidator, PymatgenCifRewriter, source_id_from_path


def _unique_source_ids(entries: Sequence[CifEntry]) -> list[tuple[Path, str]]:
    """Assign deterministic cross-stage IDs while retaining manifest order."""

    used: set[str] = set()
    output_names: set[str] = set()
    output: list[tuple[Path, str]] = []
    for entry in entries:
        candidate = (
            str(entry.explicit_id).strip()
            if entry.explicit_id
            else source_id_from_path(entry.path, existing=used)
        )
        if not candidate:
            candidate = source_id_from_path(entry.path, existing=used)
        if candidate in used:
            candidate = source_id_from_path(entry.path, existing=used)
        if not entry.explicit_id:
            base = candidate
            suffix = 2
            while candidate in used or safe_filename(candidate) in output_names:
                candidate = f"{base}__{suffix}"
                suffix += 1
        used.add(candidate)
        output_names.add(safe_filename(candidate))
        output.append((entry.path, candidate))
    return output


def _adapter_from_config(config: ProjectConfig, name: str) -> Any | None:
    adapters = config.get("adapters")
    return adapters.get(name) if isinstance(adapters, Mapping) else None


def _clear_generated_files(directory: Path) -> None:
    """Remove files from one known library output directory before rebuilding."""

    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()


def _prepare_output_directories(artifact_root: Path) -> dict[str, Path]:
    directories = {
        "basis": artifact_root / "basis_patterns",
        "cifs": artifact_root / "selected_cifs",
        "gsasii": artifact_root / "gsasii_ready_cif",
        "simulated": artifact_root / "simulated_patterns",
        "basis_simulated": artifact_root / "basis_simulated_patterns",
    }
    for directory in directories.values():
        ensure_managed_output_directory(directory, context="library generated output")
        _clear_generated_files(directory)
    return directories


def _default_validator(filter_profile: Mapping[str, Any]) -> Any:
    return DefaultCifValidator(
        occupancy_tolerance=float(filter_profile.get("occupancy_tolerance", 100.0)),
        parse_primitive=bool(filter_profile.get("parse_primitive", True)),
        max_sites=(
            None if filter_profile.get("max_sites") is None else int(filter_profile["max_sites"])
        ),
    )


def _default_rewriter(filter_profile: Mapping[str, Any]) -> Any:
    return PymatgenCifRewriter(
        primitive=bool(filter_profile.get("parse_primitive", True)),
        occupancy_tolerance=float(filter_profile.get("occupancy_tolerance", 100.0)),
        max_sites=(
            None if filter_profile.get("max_sites") is None else int(filter_profile["max_sites"])
        ),
    )


def build_library(
    config: ProjectConfig,
    output_dir: str | Path | None = None,
    *,
    validator: Any | None = None,
    rewriter: Any | None = None,
    simulator: Any | None = None,
    embedder: Any | None = None,
) -> LibraryArtifact:
    """Build a validated, simulated, embedded, and clustered basis library.

    ``config`` is a resolved ``ProjectConfig``.  ``output_dir`` is the stage
    artifact directory used by the top-level workflow.  Heavyweight
    dependencies are loaded only by their default adapters when they are
    needed.
    """

    settings = resolve_library_settings(
        config,
        Path(output_dir).expanduser() if output_dir is not None else None,
    )
    input_path = settings["input_path"]
    phase_manifest = settings.get("phase_manifest")
    artifact_root = settings["output_root"]
    if input_path is None:
        if phase_manifest is None:
            raise ValueError("Library config must provide inputs.cifs or inputs.phase_manifest")
        input_path = Path(phase_manifest).expanduser().resolve().parent
    if artifact_root is None:
        raise ValueError("Library build requires output_dir or an output_root in config")
    input_path = Path(input_path).expanduser().resolve()
    artifact_root = Path(artifact_root).expanduser()

    discovered_entries = discover_cifs(input_path, phase_manifest=phase_manifest)
    entries = _unique_source_ids(discovered_entries)
    if not entries:
        raise FileNotFoundError(f"No CIF files found under {input_path}")
    ensure_unique_sanitized_ids(
        (source_id for _, source_id in entries),
        safe_filename,
        context="library output",
    )

    artifact_root = ensure_managed_output_directory(
        artifact_root,
        context="library output",
    ).resolve()
    directories = _prepare_output_directories(artifact_root)
    labels_by_source_id = {
        source_id: entry.label or source_id
        for entry, (_, source_id) in zip(discovered_entries, entries)
    }

    filter_profile = settings["filter"]
    configured_validator = _adapter_from_config(config, "validator")
    configured_rewriter = _adapter_from_config(config, "rewriter")
    configured_cif_rewriter = _adapter_from_config(config, "cif_rewriter")
    validator = validator or configured_validator or _default_validator(filter_profile)
    rewriter = rewriter or configured_rewriter or configured_cif_rewriter
    if rewriter is None and isinstance(validator, DefaultCifValidator):
        rewriter = _default_rewriter(filter_profile)

    preparation = prepare_candidates(
        entries,
        labels_by_source_id=labels_by_source_id,
        filter_profile=filter_profile,
        validator=validator,
        rewriter=rewriter,
        gsasii_dir=directories["gsasii"],
    )
    for entry, (_, source_id) in zip(discovered_entries, entries):
        if not entry.label and not entry.explicit_id:
            row = preparation.rows_by_id[source_id]
            formula = row.get("formula")
            if formula:
                label = f"{formula} ({source_id})"
                labels_by_source_id[source_id] = label
                row["label"] = label
    deduped_records, preparation = apply_deduplication(preparation, settings["dedup"])
    print(
        f"[library] {len(entries)} CIFs: {len(preparation.records)} passed filters, "
        f"{len(deduped_records)} after deduplication",
        file=sys.stderr,
        flush=True,
    )
    records = preparation.records
    rows_by_id = preparation.rows_by_id
    stage_counts = preparation.stage_counts
    manifest_path = artifact_root / "manifest.csv"
    write_library_manifest(manifest_path, rows_by_id.values())
    if not deduped_records:
        raise ValueError(f"No CIFs passed the configured filters; see {manifest_path}")

    simulation_section = config.library.get("simulation", {})
    if not isinstance(simulation_section, Mapping):
        simulation_section = {}
    simulator = (
        simulator
        or _adapter_from_config(config, "simulator")
        or GsasIISimulator(
            scale=float(simulation_section.get("scale", 1.0)),
            quiet=bool(simulation_section.get("quiet", True)),
        )
    )
    embedder = embedder or _adapter_from_config(config, "embedder")
    if embedder is None:
        checkpoint_path = settings["checkpoint_path"]
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required unless an EmbeddingAdapter is injected")
        embedder = PxrdEmbeddingAdapter(
            checkpoint_path,
            repo_src=Path(__file__).resolve().parents[2],
        )

    try:
        patterns = call_simulator(
            simulator,
            deduped_records,
            instrument_path=settings["simulation_instrument_path"],
            theta_min=settings["theta_min"],
            theta_max=settings["theta_max"],
            n_points=settings["n_points"],
            output_dir=directories["simulated"],
        )
    except Exception:
        for record in deduped_records:
            rows_by_id[record.source_id]["simulation_status"] = "failed"
        write_library_manifest(manifest_path, rows_by_id.values())
        raise

    # Candidate simulation and checkpoint embedding may use different theta
    # windows.  Rerun on the checkpoint grid when the adapter exposes it; basis
    # files are cropped only at the final export seam.
    embedding_grid = getattr(embedder, "grid", None)
    if embedding_grid is not None:
        expected_min = float(getattr(embedding_grid, "theta_min"))
        expected_max = float(getattr(embedding_grid, "theta_max"))
        expected_points = int(getattr(embedding_grid, "n_points"))
        narrow_patterns = any(
            float(patterns[record.source_id].x[0]) > expected_min + 1e-5
            or float(patterns[record.source_id].x[-1]) < expected_max - 1e-5
            for record in deduped_records
            if record.source_id in patterns
        )
        if narrow_patterns:
            try:
                patterns = call_simulator(
                    simulator,
                    deduped_records,
                    instrument_path=settings["simulation_instrument_path"],
                    theta_min=expected_min,
                    theta_max=expected_max,
                    n_points=expected_points,
                    output_dir=directories["simulated"],
                )
            except Exception:
                # Keep a rerun failure visible in the manifest just like the
                # initial simulation failure above.  Without this update the
                # rows remain ``not_selected`` even though this stage failed.
                for record in deduped_records:
                    rows_by_id[record.source_id]["simulation_status"] = "failed"
                write_library_manifest(manifest_path, rows_by_id.values())
                raise

    eligible_records = persist_simulated_patterns(
        deduped_records,
        patterns,
        output_dir=directories["simulated"],
        rows_by_id=rows_by_id,
    )
    if not eligible_records:
        raise RuntimeError("No validated CIFs produced a simulation pattern")

    ordered_patterns = [patterns[record.source_id] for record in eligible_records]
    theta = np.asarray(ordered_patterns[0].x, dtype=np.float32)
    embeddings, embedding_theta = call_embedder(
        embedder,
        ordered_patterns,
        theta=theta,
        batch_size=settings["batch_size"],
        device=settings["device"],
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(eligible_records):
        raise ValueError(
            f"Embedding shape {embeddings.shape} does not match {len(eligible_records)} patterns"
        )
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embedding output contains non-finite values")

    clustering_section = settings["clustering"]
    cluster_selection = clustering_section.get("selection", "stable_auto")
    cluster_target = clustering_section.get("target_clusters")
    clustering_result = cluster_embeddings(
        embeddings,
        [record.source_id for record in eligible_records],
        selection=str(cluster_selection),
        target_clusters=cluster_target,
        optimal_ordering=bool(clustering_section.get("optimal_ordering", True)),
        min_ari=float(clustering_section.get("min_ari", DEFAULT_MIN_ARI)),
        threshold_lower_quantile=float(clustering_section.get("threshold_lower_quantile", 0.10)),
        threshold_upper_quantile=float(clustering_section.get("threshold_upper_quantile", 0.90)),
        n_thresholds=int(clustering_section.get("n_thresholds", 60)),
        min_clusters=int(clustering_section.get("min_clusters", 1)),
        max_clusters=clustering_section.get("max_clusters"),
        stability=clustering_section.get("stability"),
        stable_regime=clustering_section.get("stable_regime"),
    )

    representative_records = [
        eligible_records[index]
        for index in sorted(
            int(value) for value in clustering_result.representative_indices.tolist()
        )
    ]
    print(
        f"[library] {len(representative_records)} representative structures selected",
        file=sys.stderr,
        flush=True,
    )
    basis_patterns = call_simulator(
        simulator,
        representative_records,
        instrument_path=settings["basis_instrument_path"],
        theta_min=settings["basis_theta_min"],
        theta_max=settings["basis_theta_max"],
        n_points=settings["basis_n_points"],
        output_dir=directories["basis_simulated"],
    )
    if len(basis_patterns) != len(representative_records):
        missing = sorted(
            set(record.source_id for record in representative_records) - set(basis_patterns)
        )
        raise ValueError(f"Basis simulation output is missing representatives: {missing[:3]}")

    return export_library(
        LibraryExportContext(
            artifact_root=artifact_root,
            input_path=input_path,
            entries=entries,
            rows_by_id=rows_by_id,
            labels_by_source_id=labels_by_source_id,
            eligible_records=eligible_records,
            patterns=patterns,
            embeddings=embeddings,
            theta=theta,
            embedding_theta=embedding_theta,
            clustering_result=clustering_result,
            basis_patterns=basis_patterns,
            settings=settings,
            stage_counts=stage_counts,
            validated_count=len(records),
            deduplicated_count=len(deduped_records),
            cluster_selection=cluster_selection,
            cluster_target=cluster_target,
        )
    )


__all__ = ["build_library"]

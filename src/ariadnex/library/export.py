"""On-disk materialisation for a built diffraction basis library."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..artifacts import safe_filename
from ..contracts import LibraryArtifact
from .artifact import CifMetadata, ClusterResult, PxrdPattern
from .clustering import compute_cosine_distance_matrix

_MANIFEST_FIELDS = (
    "phase_id",
    "source_id",
    "label",
    "source_path",
    "gsasii_cif_path",
    "conversion_status",
    "conversion_reason",
    "parse_status",
    "audit_creation_date",
    "formula",
    "source_formula",
    "source_formula_sum",
    "elements",
    "space_group",
    "space_group_number",
    "source_space_group",
    "source_space_group_number",
    "source_accession",
    "source_block",
    "cell_volume",
    "n_sites",
    "accepted",
    "smact_status",
    "rejection_stage",
    "rejection_reason",
    "dedup_status",
    "dedup_reason",
    "dedup_canonical_source_id",
    "dedup_group_size",
    "duplicate_group",
    "simulation_status",
    "pattern_path",
    "basis_simulated_pattern_path",
    "simulated_pattern_path",
    "cluster_eligible",
    "cluster_label",
    "representative",
)


@dataclass
class LibraryExportContext:
    """All state needed by the single library export seam."""

    artifact_root: Path
    input_path: Path
    entries: Sequence[tuple[Path, str]]
    rows_by_id: dict[str, dict[str, Any]]
    labels_by_source_id: Mapping[str, str]
    eligible_records: Sequence[CifMetadata]
    patterns: Mapping[str, PxrdPattern]
    embeddings: np.ndarray
    theta: np.ndarray
    embedding_theta: np.ndarray
    clustering_result: ClusterResult
    basis_patterns: Mapping[str, PxrdPattern]
    settings: Mapping[str, Any]
    stage_counts: Mapping[str, int]
    validated_count: int
    deduplicated_count: int
    cluster_selection: Any
    cluster_target: Any


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        # Unavailable clustering statistics use standard JSON null.
        return None
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return value


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def write_library_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Keep input decisions available even when a later build stage fails."""

    _write_csv(path, rows, _MANIFEST_FIELDS)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (tuple, list, set)):
        return ";".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, np.ndarray):
        return ";".join(str(item) for item in value.tolist())
    return value


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _basis_view(
    pattern: PxrdPattern, *, theta_min: float, theta_max: float, n_points: int
) -> PxrdPattern:
    """Return a basis pattern on the configured unmix range."""

    target = np.round(
        np.linspace(float(theta_min), float(theta_max), int(n_points), dtype=np.float64),
        5,
    )
    x = np.asarray(pattern.x, dtype=np.float64)
    y = np.asarray(pattern.y, dtype=np.float64)
    if float(x[0]) > float(target[0]) + 1e-6 or float(x[-1]) < float(target[-1]) - 1e-6:
        raise ValueError(
            f"Pattern {pattern.source_id!r} does not cover basis theta range "
            f"[{theta_min}, {theta_max}]"
        )
    values = np.interp(target.astype(np.float64), x, y)
    return PxrdPattern(
        source_id=pattern.source_id, x=target, y=values, source_path=pattern.source_path
    )


def _settings_metadata(settings: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(settings)
    for key in ("filter", "dedup", "clustering", "model"):
        output[key] = _plain(output.get(key, {}))
    return _plain(output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_sha256(source_ids: Iterable[str]) -> str:
    """Hash a newline-delimited source-ID sequence for manifest comparisons."""

    payload = "".join(f"{str(source_id)}\n" for source_id in source_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def persist_simulated_patterns(
    records: Sequence[CifMetadata],
    patterns: Mapping[str, PxrdPattern],
    *,
    output_dir: Path,
    rows_by_id: dict[str, dict[str, Any]],
) -> list[CifMetadata]:
    """Persist full-range simulation outputs and mark cluster eligibility."""

    eligible: list[CifMetadata] = []
    for record in records:
        row = rows_by_id[record.source_id]
        if record.source_id not in patterns:
            row.update(simulation_status="missing_output", cluster_eligible=False)
            continue
        path = output_dir / f"{safe_filename(record.source_id)}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        pattern = patterns[record.source_id]
        np.savez_compressed(
            path,
            x=np.asarray(pattern.x, dtype=np.float64),
            y=np.asarray(pattern.y, dtype=np.float64),
            source_id=record.source_id,
        )
        row.update(
            simulation_status="success",
            cluster_eligible=True,
            simulated_pattern_path=str(path.resolve()),
        )
        eligible.append(record)
    return eligible


def export_library(context: LibraryExportContext) -> LibraryArtifact:
    """Materialise basis files, manifests, arrays, and metadata."""

    artifact_root = Path(context.artifact_root)
    basis_dir = artifact_root / "basis_patterns"
    cifs_dir = artifact_root / "selected_cifs"
    basis_simulated_dir = artifact_root / "basis_simulated_patterns"
    representative_indices = set(
        int(index) for index in context.clustering_result.representative_indices.tolist()
    )
    assignments: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []

    for index, record in enumerate(context.eligible_records):
        cluster_label = int(context.clustering_result.labels[index])
        representative_index = int(
            context.clustering_result.assignment_representative_indices[index]
        )
        distance = float(context.clustering_result.distances_to_representative[index])
        assignments.append(
            {
                "source_id": record.source_id,
                "cluster_label": cluster_label,
                "representative_id": context.eligible_records[representative_index].source_id,
                "distance_to_medoid": distance,
            }
        )
        context.rows_by_id[record.source_id].update(
            cluster_label=cluster_label,
            representative=bool(index in representative_indices),
            pattern_path=(
                str((basis_dir / f"{safe_filename(record.source_id)}.npz").resolve())
                if index in representative_indices
                else ""
            ),
        )

    for cluster_label in sorted(
        set(int(label) for label in context.clustering_result.labels.tolist())
    ):
        members = [
            index
            for index, label in enumerate(context.clustering_result.labels.tolist())
            if int(label) == cluster_label
        ]
        medoid_index = int(context.clustering_result.assignment_representative_indices[members[0]])
        medoid = context.eligible_records[medoid_index]
        representative_rows.append(
            {
                "cluster_label": cluster_label,
                "cluster_size": len(members),
                "phase_id": medoid.source_id,
                "source_id": medoid.source_id,
                "label": context.labels_by_source_id.get(medoid.source_id, medoid.source_id),
                "source_path": str(medoid.source_path),
                "basis_pattern_path": str(
                    (basis_dir / f"{safe_filename(medoid.source_id)}.npz").resolve()
                ),
                "formula": medoid.formula,
                "elements": medoid.elements,
                "medoid_index": medoid_index,
                "avg_intra_cluster_distance": float(
                    np.mean(
                        [
                            context.clustering_result.distances_to_representative[index]
                            for index in members
                        ]
                    )
                ),
                "max_intra_cluster_distance": float(
                    np.max(
                        [
                            context.clustering_result.distances_to_representative[index]
                            for index in members
                        ]
                    )
                ),
            }
        )

    representative_records = [
        context.eligible_records[index] for index in sorted(representative_indices)
    ]
    for record in representative_records:
        basis_simulated_path = basis_simulated_dir / f"{safe_filename(record.source_id)}.npz"
        basis_pattern = context.basis_patterns[record.source_id]
        np.savez_compressed(
            basis_simulated_path,
            x=np.asarray(basis_pattern.x, dtype=np.float64),
            y=np.asarray(basis_pattern.y, dtype=np.float64),
            source_id=record.source_id,
        )
        context.rows_by_id[record.source_id]["basis_simulated_pattern_path"] = str(
            basis_simulated_path.resolve()
        )

    for index in sorted(representative_indices):
        record = context.eligible_records[index]
        pattern = context.basis_patterns[record.source_id]
        basis_pattern = _basis_view(
            pattern,
            theta_min=context.settings["basis_theta_min"],
            theta_max=context.settings["basis_theta_max"],
            n_points=context.settings["basis_n_points"],
        )
        path = basis_dir / f"{safe_filename(record.source_id)}.npz"
        np.savez_compressed(
            path,
            x=basis_pattern.x.astype(np.float64),
            y=basis_pattern.y.astype(np.float64),
            source_id=record.source_id,
        )
        destination_cif = cifs_dir / f"{safe_filename(record.source_id)}.cif"
        original_cif = Path(context.rows_by_id[record.source_id]["source_path"])
        try:
            shutil.copy2(original_cif, destination_cif)
        except OSError as exc:
            raise OSError(
                f"Could not copy selected CIF for {record.source_id!r}: "
                f"{original_cif} -> {destination_cif}"
            ) from exc

    manifest_path = artifact_root / "manifest.csv"
    representatives_path = artifact_root / "representatives.csv"
    assignments_path = artifact_root / "cluster_assignments.csv"
    write_library_manifest(
        manifest_path,
        (context.rows_by_id[source_id] for _, source_id in context.entries),
    )
    _write_csv(
        representatives_path,
        representative_rows,
        (
            "cluster_label",
            "cluster_size",
            "phase_id",
            "source_id",
            "label",
            "source_path",
            "basis_pattern_path",
            "formula",
            "elements",
            "medoid_index",
            "avg_intra_cluster_distance",
            "max_intra_cluster_distance",
        ),
    )
    _write_csv(
        assignments_path,
        assignments,
        ("source_id", "cluster_label", "representative_id", "distance_to_medoid"),
    )

    np.save(artifact_root / "embeddings.npy", context.embeddings.astype(np.float32))
    distance_matrix = compute_cosine_distance_matrix(context.embeddings)
    np.save(artifact_root / "distance_matrix.npy", distance_matrix.astype(np.float32))
    np.save(
        artifact_root / "linkage.npy", context.clustering_result.linkage_matrix.astype(np.float64)
    )
    np.save(artifact_root / "labels.npy", context.clustering_result.labels.astype(np.int32))
    np.savez_compressed(
        artifact_root / "patterns.npz",
        theta=context.theta.astype(np.float32),
        sample_ids=np.asarray([record.source_id for record in context.eligible_records], dtype=str),
        intensities=np.stack(
            [context.patterns[record.source_id].y for record in context.eligible_records],
            axis=0,
        ).astype(np.float32),
    )

    metadata = {
        "schema_version": 1,
        "artifact_type": "library",
        "input_path": str(context.input_path),
        "theta": context.embedding_theta.tolist(),
        "simulation_theta": context.theta.tolist(),
        "n_input_cifs": len(context.entries),
        "n_domain_pass": int(context.stage_counts["domain_pass"]),
        "n_conversion_success": int(context.stage_counts["conversion_success"]),
        "n_conversion_skipped": int(context.stage_counts["conversion_skipped"]),
        "n_conversion_failed": int(context.stage_counts["conversion_failed"]),
        "n_validated": int(context.validated_count),
        "n_smact_pass": int(context.stage_counts["smact_pass"]),
        "n_smact_drop": int(context.stage_counts["smact_drop"]),
        "n_deduplicated": int(context.deduplicated_count),
        "n_simulated": len(context.eligible_records),
        "n_representatives": len(representative_rows),
        "basis_simulation_theta_min": float(context.settings["basis_theta_min"]),
        "basis_simulation_theta_max": float(context.settings["basis_theta_max"]),
        "basis_simulation_n_points": int(context.settings["basis_n_points"]),
        "candidate_count": len(context.eligible_records),
        "candidate_id_sha256_row_order": _id_sha256(
            record.source_id for record in context.eligible_records
        ),
        "candidate_id_sha256_sorted": _id_sha256(
            sorted(record.source_id for record in context.eligible_records)
        ),
        "cluster_selection": str(context.cluster_selection),
        "cluster_target": context.cluster_target,
        "selected_threshold": context.clustering_result.threshold,
        "cluster_metrics": _plain(context.clustering_result.metrics),
        "threshold_scan": _plain(context.clustering_result.threshold_scan),
        "settings": _settings_metadata(context.settings),
        "input_fingerprints": {
            str(path): _sha256(path) for path, _ in context.entries if path.is_file()
        },
    }
    metadata_path = artifact_root / "metadata.json"
    _json_dump(metadata_path, metadata)
    return LibraryArtifact(
        root=artifact_root,
        manifest_path=manifest_path,
        representatives_path=representatives_path,
        basis_patterns_dir=basis_dir,
        metadata_path=metadata_path,
        metadata=metadata,
    )


__all__ = [
    "LibraryExportContext",
    "export_library",
    "persist_simulated_patterns",
    "write_library_manifest",
]

"""Candidate validation, conversion, filtering, and de-duplication stage."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import safe_filename
from .adapters import call_rewriter, call_validator
from .artifact import CifMetadata
from .config import section_value
from .filtering import deduplicate_records, evaluate_domain_filter
from .preselection import evaluate_smact
from .validation import OversizedCifError

_RAW_METADATA_KEYS = (
    # CifWriter may change formula notation, symmetry settings and atom labels.
    # Retain the original declarations separately from the simulated structure.
    "source_formula",
    "source_formula_sum",
    "source_space_group",
    "source_space_group_number",
    "source_accession",
    "source_block",
    "source_digest",
    # CifWriter does not retain these source audit/ambient tags.  Keep them for
    # manifest provenance and for any downstream policy that reports them.
    "audit_creation_date",
    "ambient_temperature_raw",
    "ambient_temperature_kelvin",
    "ambient_pressure_raw",
    "ambient_pressure_gpa",
    "metadata_phase_values",
)

# Parser and adapter failures that describe one malformed CIF remain visible in
# that row. Runtime, import, and filesystem errors indicate a broken execution
# environment and should reach the workflow with their original traceback.
_ENVIRONMENT_ERRORS = (ImportError, OSError, RuntimeError)


@dataclass(frozen=True)
class CandidatePreparation:
    """Validated records and manifest rows ready for numerical stages."""

    records: list[CifMetadata]
    rows_by_id: dict[str, dict[str, Any]]
    stage_counts: dict[str, int]


def _domain_profile_without_size(profile: Any) -> Any:
    """Leave site-count handling to the conversion stage, as in old pipeline."""

    if isinstance(profile, Mapping):
        result = dict(profile)
        result["max_sites"] = None
        return result
    return replace(profile, max_sites=None)


def _downstream_record(
    raw_record: CifMetadata, converted_record: CifMetadata, converted_path: Path
) -> CifMetadata:
    """Combine converted structural fields with raw-only provenance metadata."""

    raw_metadata = dict(raw_record.metadata or {})
    converted_metadata = dict(converted_record.metadata or {})
    raw_composition = raw_metadata.get("composition")
    converted_composition = converted_metadata.get("composition")
    if raw_composition and converted_composition:
        raw_total = sum(raw_composition.values())
        converted_total = sum(converted_composition.values())
        elements = raw_composition.keys() | converted_composition.keys()
        if any(
            abs(
                raw_composition.get(element, 0) / raw_total
                - converted_composition.get(element, 0) / converted_total
            )
            > 1e-6
            for element in elements
        ):
            raise ValueError(
                "CIF conversion changed elemental ratios: "
                f"{raw_record.formula} -> {converted_record.formula}"
            )
    for key in _RAW_METADATA_KEYS:
        if key in raw_metadata:
            converted_metadata[key] = raw_metadata[key]
    # CifWriter declares P1 for its expanded simulation cell. If symmetry
    # analysis fails, use the original structure's symmetry rather than that
    # generated declaration.
    symmetry = (
        converted_record if converted_metadata.get("space_group_analyzed", True) else raw_record
    )
    return replace(
        converted_record,
        source_id=raw_record.source_id,
        source_path=Path(converted_path),
        space_group=symmetry.space_group,
        space_group_number=symmetry.space_group_number,
        metadata=converted_metadata,
    )


def _manifest_row(
    record: CifMetadata,
    *,
    path: Path,
    label: str,
    accepted: bool,
    rejection_stage: str,
    rejection_reason: str,
    smact_status: str,
    conversion_status: str,
    conversion_reason: str,
    converted_path: Path | None,
) -> dict[str, Any]:
    metadata = record.metadata or {}
    return {
        # ``phase_id`` is the stable cross-stage identity.  Keep ``source_id``
        # as an explicit alias for consumers of the older library contract.
        "phase_id": record.source_id,
        "source_id": record.source_id,
        "label": label,
        "source_path": str(path),
        "gsasii_cif_path": "" if converted_path is None else str(converted_path),
        "conversion_status": conversion_status,
        "conversion_reason": conversion_reason,
        "parse_status": record.parse_status,
        "audit_creation_date": metadata.get("audit_creation_date", ""),
        "formula": record.formula,
        "source_formula": metadata.get("source_formula", ""),
        "source_formula_sum": metadata.get("source_formula_sum", ""),
        "elements": record.elements,
        "space_group": record.space_group,
        "space_group_number": record.space_group_number,
        "source_space_group": metadata.get("source_space_group", ""),
        "source_space_group_number": metadata.get("source_space_group_number"),
        "source_accession": metadata.get("source_accession", ""),
        "source_block": metadata.get("source_block", ""),
        "cell_volume": record.cell_volume if record.cell_volume is not None else "",
        "n_sites": record.n_sites if record.n_sites is not None else "",
        "accepted": bool(accepted),
        "smact_status": smact_status,
        "rejection_stage": rejection_stage,
        "rejection_reason": rejection_reason if not accepted else "",
        "dedup_status": "not_run",
        "dedup_reason": "",
        "dedup_canonical_source_id": "",
        "dedup_group_size": "",
        "duplicate_group": "",
        "simulation_status": "not_selected",
        "pattern_path": "",
        "basis_simulated_pattern_path": "",
        "simulated_pattern_path": "",
        "cluster_eligible": False,
        "cluster_label": "",
        "representative": False,
    }


def prepare_candidates(
    entries: Sequence[tuple[Path, str]],
    *,
    labels_by_source_id: Mapping[str, str],
    filter_profile: Any,
    validator: Any,
    rewriter: Any | None,
    gsasii_dir: Path,
) -> CandidatePreparation:
    """Run the candidate stages while keeping each failure at its owning seam."""

    records: list[CifMetadata] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    domain_profile = _domain_profile_without_size(filter_profile)
    stage_counts = {
        "domain_pass": 0,
        "conversion_success": 0,
        "conversion_skipped": 0,
        "conversion_failed": 0,
        "smact_pass": 0,
        "smact_drop": 0,
    }
    max_sites = section_value(filter_profile, "max_sites", None)

    for index, (path, source_id) in enumerate(entries, 1):
        smact_status = "not_run"
        conversion_status = "not_run"
        conversion_reason = ""
        converted_path: Path | None = None

        # A malformed validator result is a per-CIF validation failure.  The
        # later stages intentionally stay outside this catch so environment or
        # configuration errors reach the workflow with their real cause.
        try:
            record = call_validator(validator, path, source_id)
        except _ENVIRONMENT_ERRORS:
            raise
        except Exception as exc:
            record = CifMetadata(source_id=source_id, source_path=path, parse_status="fail")
            accepted, rejection_stage, rejection_reason = (
                False,
                "validation",
                f"validator_failure:{exc}",
            )
        else:
            accepted, rejection_stage, rejection_reason = evaluate_domain_filter(
                record, domain_profile
            )
            if accepted:
                stage_counts["domain_pass"] += 1
                if (
                    max_sites is not None
                    and record.n_sites is not None
                    and int(record.n_sites) > int(max_sites)
                ):
                    accepted = False
                    rejection_stage = "conversion"
                    rejection_reason = (
                        f"skipped_oversized:{record.n_sites} sites > {int(max_sites)}"
                    )
                    conversion_status = "skipped"
                    conversion_reason = rejection_reason
                    stage_counts["conversion_skipped"] += 1
                elif rewriter is not None:
                    target_path = gsasii_dir / f"{safe_filename(source_id)}.cif"
                    try:
                        converted_path = call_rewriter(rewriter, path, target_path)
                        converted_record = call_validator(validator, converted_path, source_id)
                        if str(converted_record.parse_status).lower() not in {
                            "pass",
                            "ok",
                            "success",
                        }:
                            raise ValueError(
                                f"converted_parser_status:{converted_record.parse_status}"
                            )
                        if (
                            max_sites is not None
                            and converted_record.n_sites is not None
                            and int(converted_record.n_sites) > int(max_sites)
                        ):
                            raise OversizedCifError(
                                f"{converted_record.n_sites} sites > {max_sites}"
                            )
                        record = _downstream_record(record, converted_record, converted_path)
                        conversion_status = "success"
                        conversion_reason = "success"
                        stage_counts["conversion_success"] += 1
                    except OversizedCifError as exc:
                        accepted = False
                        rejection_stage = "conversion"
                        rejection_reason = f"skipped_oversized:{exc}"
                        conversion_status = "skipped"
                        conversion_reason = rejection_reason
                        # A failed conversion must not leave a path that looks
                        # usable in the manifest.  The generated file, if any,
                        # is managed output and is cleared at the next build.
                        converted_path = None
                        stage_counts["conversion_skipped"] += 1
                    except _ENVIRONMENT_ERRORS:
                        raise
                    except Exception as exc:
                        accepted = False
                        rejection_stage = "conversion"
                        rejection_reason = f"conversion_failed:{exc}"
                        conversion_status = "failed"
                        conversion_reason = str(exc)
                        # Do not expose a partially written or rejected CIF as
                        # the downstream GSAS-II input for this row.
                        converted_path = None
                        stage_counts["conversion_failed"] += 1

            if accepted and bool(section_value(filter_profile, "smact_enabled", False)):
                try:
                    smact_ok, smact_status = evaluate_smact(record, filter_profile)
                except _ENVIRONMENT_ERRORS:
                    # An enabled chemistry profile must not silently widen the
                    # candidate set when its required environment is absent.
                    raise
                except Exception as exc:
                    smact_ok, smact_status = False, f"error:{exc}"
                if not smact_ok:
                    accepted, rejection_stage, rejection_reason = (
                        False,
                        "preselection",
                        f"smact_invalid:{smact_status}",
                    )
                    stage_counts["smact_drop"] += 1
                else:
                    stage_counts["smact_pass"] += 1

        rows_by_id[source_id] = _manifest_row(
            record,
            path=path,
            label=labels_by_source_id.get(source_id, source_id),
            accepted=accepted,
            rejection_stage=rejection_stage,
            rejection_reason=rejection_reason,
            smact_status=smact_status,
            conversion_status=conversion_status,
            conversion_reason=conversion_reason,
            converted_path=converted_path,
        )
        if accepted:
            records.append(record)
        if index % 50 == 0 or index == len(entries):
            print(
                f"[library] CIFs checked: {index}/{len(entries)}",
                file=sys.stderr,
                flush=True,
            )

    return CandidatePreparation(records, rows_by_id, stage_counts)


def apply_deduplication(
    preparation: CandidatePreparation,
    dedup_section: Any,
) -> tuple[list[CifMetadata], CandidatePreparation]:
    """Apply the configured duplicate policy and update manifest rows."""

    dedup_enabled = bool(section_value(dedup_section, "enabled", True))
    dedup_tolerance = float(section_value(dedup_section, "volume_relative_tolerance", 0.05))
    deduped_records, dedup_status = deduplicate_records(
        preparation.records,
        enabled=dedup_enabled,
        relative_volume_tolerance=dedup_tolerance,
    )
    for source_id, status in dedup_status.items():
        preparation.rows_by_id[source_id].update(status)
        if int(status.get("dedup_group_size", 1) or 1) > 1:
            preparation.rows_by_id[source_id]["duplicate_group"] = str(
                status.get("dedup_canonical_source_id") or source_id
            )
        if status.get("dedup_status") == "duplicate_drop":
            preparation.rows_by_id[source_id].update(
                accepted=False,
                rejection_stage="dedup",
                rejection_reason=status.get("dedup_reason", "duplicate"),
            )
    return deduped_records, preparation


__all__ = ["CandidatePreparation", "apply_deduplication", "prepare_candidates"]

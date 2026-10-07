"""Domain filtering and deterministic CIF de-duplication."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Iterable, Mapping

from .artifact import CifMetadata
from .config import FilterProfile


def _value(profile: Any, key: str, default: Any = None) -> Any:
    if isinstance(profile, Mapping):
        return profile.get(key, default)
    return getattr(profile, key, default)


def _as_elements(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(sorted(set(re.findall(r"[A-Z][a-z]?", value))))
    return tuple(sorted({str(item) for item in value}))


def formula_key(formula: str) -> str:
    """Normalise whitespace while preserving exact formula notation."""

    return re.sub(r"\s+", "", str(formula or ""))


def evaluate_domain_filter(
    record: CifMetadata, profile: FilterProfile | Mapping[str, Any] | None
) -> tuple[bool, str, str]:
    """Return ``(accepted, stage, reason)`` for one parsed CIF record."""

    if profile is None:
        return True, "", "accepted"
    if str(record.parse_status).lower() not in {"pass", "ok", "success"}:
        return False, "validation", f"parser_status:{record.parse_status}"

    audit_cutoff = _value(profile, "audit_creation_date_cutoff")
    audit_date = (record.metadata or {}).get("audit_creation_date")
    if audit_cutoff and audit_date:
        try:
            if date.fromisoformat(str(audit_date)[:10]) > date.fromisoformat(
                str(audit_cutoff)[:10]
            ):
                return False, "audit_filter", f"audit_creation_date_after_cutoff:{audit_date}"
        except ValueError:
            # An unparseable audit tag should not cause an otherwise valid CIF
            # to disappear; the raw value remains in the manifest metadata.
            pass

    allowed = _as_elements(_value(profile, "allowed_elements"))
    present = set(record.elements)
    if allowed and not present.issubset(set(allowed)):
        disallowed = ",".join(sorted(present - set(allowed)))
        return False, "domain_filter", f"element_not_allowed:{disallowed}"

    required = _as_elements(_value(profile, "required_elements", ()))
    if required and not set(required).issubset(present):
        missing = ",".join(sorted(set(required) - present))
        return False, "domain_filter", f"required_element_missing:{missing}"

    requested_formula = _value(profile, "formula")
    if requested_formula and formula_key(record.formula) != formula_key(str(requested_formula)):
        return False, "domain_filter", f"formula_mismatch:{record.formula}!={requested_formula}"

    formula_regex = _value(profile, "formula_regex")
    if formula_regex and re.search(str(formula_regex), str(record.formula)) is None:
        return False, "domain_filter", f"formula_regex_mismatch:{record.formula}"

    max_sites = _value(profile, "max_sites")
    if (
        max_sites is not None
        and record.n_sites is not None
        and int(record.n_sites) > int(max_sites)
    ):
        return False, "domain_filter", f"structure_too_large:{record.n_sites}>{int(max_sites)}"

    if bool(_value(profile, "metadata_phase_filter_enabled", True)):
        fields = _value(
            profile,
            "metadata_phase_fields",
            ("_chemical_name_common", "_chemical_name_systematic"),
        )
        keywords = tuple(
            str(item).lower()
            for item in _value(
                profile,
                "metadata_phase_hard_drop_keywords",
                ("amorphous", "x-ray amorphous", "noncrystalline", "non-crystalline"),
            )
        )
        phase_values = record.metadata.get("metadata_phase_values", {}) if record.metadata else {}
        for field in fields:
            raw = str(phase_values.get(str(field).lower(), phase_values.get(str(field), "")))
            for keyword in keywords:
                if keyword and keyword in raw.lower():
                    return False, "metadata_filter", f"metadata_phase_hard_drop:{field}:{keyword}"

    if bool(_value(profile, "ambient_condition_filter_enabled", False)) and record.metadata:
        temp = record.metadata.get("ambient_temperature_kelvin")
        temp_limit = _value(profile, "max_ambient_temperature_kelvin")
        if temp_limit is not None and temp is not None and float(temp) > float(temp_limit):
            return (
                False,
                "ambient_filter",
                f"ambient_temperature_above_threshold:{temp}>{temp_limit}K",
            )
        pressure = record.metadata.get("ambient_pressure_gpa")
        pressure_limit = _value(profile, "max_ambient_pressure_gpa")
        if (
            pressure_limit is not None
            and pressure is not None
            and float(pressure) >= float(pressure_limit)
        ):
            return (
                False,
                "ambient_filter",
                f"ambient_pressure_above_threshold:{pressure}>={pressure_limit}GPa",
            )

    return True, "", "accepted"


def _group_key(record: CifMetadata) -> tuple[str, str]:
    metadata = record.metadata or {}
    formula = str(metadata.get("source_formula") or record.formula)
    space_group = str(metadata.get("source_space_group") or record.space_group or "")
    return formula_key(formula), re.sub(r"\s+", "", space_group)


def deduplicate_records(
    records: Iterable[CifMetadata],
    *,
    relative_volume_tolerance: float = 0.05,
    enabled: bool = True,
) -> tuple[list[CifMetadata], dict[str, dict[str, Any]]]:
    """Choose one canonical record for each formula/space-group/volume group.

    The selected record is closest to the group's mean cell volume. For a
    two-member group, distance differences below 1e-12 cubic angstroms are
    treated as a tie and resolved by stable source order. This prevents
    roundoff in the mean from choosing between mathematically tied members.
    """

    ordered = sorted(records, key=lambda item: item.sort_key)
    status: dict[str, dict[str, Any]] = {
        item.source_id: {
            "dedup_status": "disabled" if not enabled else "not_applicable",
            "dedup_reason": "disabled_by_config" if not enabled else "",
            "dedup_canonical_source_id": "",
            "dedup_group_size": 1,
        }
        for item in ordered
    }
    if not enabled:
        return ordered, status

    groups: dict[tuple[str, str], list[CifMetadata]] = {}
    for item in ordered:
        group_formula, group_space_group = _group_key(item)
        if item.cell_volume is None or not group_formula or not group_space_group:
            status[item.source_id].update(
                dedup_status="not_applicable",
                dedup_reason="missing_formula_space_group_or_volume",
            )
            continue
        groups.setdefault(_group_key(item), []).append(item)

    kept: set[str] = {item.source_id for item in ordered}
    for group_key, group in groups.items():
        # Sort by volume before making contiguous groups, as in the source
        # source pipeline.  A volume spread crossing tolerance starts a new
        # duplicate group rather than chaining indefinitely.
        group = sorted(group, key=lambda item: (float(item.cell_volume), item.sort_key))
        current: list[CifMetadata] = []
        current_volumes: list[float] = []
        groups_by_volume: list[list[CifMetadata]] = []
        for item in group:
            value = float(item.cell_volume)
            candidate = [*current_volumes, value]
            mean = sum(candidate) / len(candidate)
            spread = max(abs(volume - mean) / mean for volume in candidate) if mean else 0.0
            if current and spread > float(relative_volume_tolerance):
                groups_by_volume.append(current)
                current = [item]
                current_volumes = [value]
            else:
                current.append(item)
                current_volumes.append(value)
        if current:
            groups_by_volume.append(current)

        for volume_group in groups_by_volume:
            mean = sum(float(item.cell_volume) for item in volume_group) / len(volume_group)  # type: ignore[arg-type]
            canonical = min(
                volume_group,
                key=lambda item: (abs(float(item.cell_volume) - mean), item.sort_key),  # type: ignore[arg-type]
            )
            size = len(volume_group)
            if size == 2:
                distances = [abs(float(item.cell_volume) - mean) for item in volume_group]
                if abs(distances[0] - distances[1]) < 1e-12:
                    canonical = min(volume_group, key=lambda item: item.sort_key)
            key_text = f"{group_key[0]}|sg={group_key[1]}"
            for item in volume_group:
                row = status[item.source_id]
                row["dedup_canonical_source_id"] = canonical.source_id
                row["dedup_group_size"] = size
                if size == 1:
                    row.update(
                        dedup_status="unique_keep",
                        dedup_reason=f"unique_formula_space_group_volume_group:{key_text}",
                    )
                elif item.source_id == canonical.source_id:
                    row.update(
                        dedup_status="canonical_keep",
                        dedup_reason=f"canonical_formula_space_group_volume_group:{key_text}",
                    )
                else:
                    row.update(
                        dedup_status="duplicate_drop",
                        dedup_reason=f"duplicate_formula_space_group_volume_group:{key_text}:canonical={canonical.source_id}",
                    )
                    kept.discard(item.source_id)

    return [item for item in ordered if item.source_id in kept], status

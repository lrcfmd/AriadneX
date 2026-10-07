"""Prepare reproducible local CIF inputs for the bundled paper examples."""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_SPECS = {
    "controlled_leave_one_out": {
        "database_code": "PCD",
        "catalog": "examples/controlled_leave_one_out/entries.csv",
        "config": "configs/controlled_leave_one_out.yaml",
    },
    "open_world_phase_discovery": {
        "database_code": "ICSD",
        "catalog": "examples/open_world_phase_discovery/entries.csv",
        "config": "configs/open_world_phase_discovery.yaml",
    },
}

class ExampleInputError(ValueError):
    """Raised when a supplied CIF directory cannot be matched safely."""


@dataclass(frozen=True)
class ExampleEntry:
    phase_id: str
    label: str
    database_id: str


@dataclass(frozen=True)
class PreparedExample:
    example: str
    output_dir: Path
    manifest_path: Path
    config_path: Path
    entry_count: int
    backup_dir: Path | None = None


def _spec_path(relative_path: str) -> Path:
    path = _REPOSITORY_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"bundled example file is missing: {path}. "
            "Use an editable AriadneX checkout with its examples/ and configs/ directories."
        )
    return path


def _load_catalog(path: Path) -> list[ExampleEntry]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"phase_id", "label", "database_id"}
        fields = {name.strip().lower() for name in (reader.fieldnames or [])}
        if not required.issubset(fields):
            raise ExampleInputError(
                f"example catalog {path} must contain columns phase_id, label, database_id"
            )
        field_map = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        rows = list(reader)

    if not rows:
        raise ExampleInputError(f"example catalog is empty: {path}")

    entries: list[ExampleEntry] = []
    seen_phase_ids: set[str] = set()
    seen_database_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        phase_id = str(row.get(field_map["phase_id"], "") or "").strip()
        label = str(row.get(field_map["label"], "") or "").strip()
        database_id = str(row.get(field_map["database_id"], "") or "").strip()
        if not phase_id or not label or not database_id:
            raise ExampleInputError(
                f"example catalog row {row_number} has an empty phase_id, label, or database_id: {path}"
            )
        if not re.fullmatch(r"\d+", database_id):
            raise ExampleInputError(
                f"example catalog row {row_number} has a nonnumeric database_id "
                f"{database_id!r}: {path}"
            )
        if phase_id in seen_phase_ids:
            raise ExampleInputError(f"duplicate phase_id {phase_id!r} in example catalog: {path}")
        if database_id in seen_database_ids:
            raise ExampleInputError(
                f"duplicate database_id {database_id!r} in example catalog: {path}"
            )
        seen_phase_ids.add(phase_id)
        seen_database_ids.add(database_id)
        entries.append(ExampleEntry(phase_id, label, database_id))
    return entries


def _remove_unquoted_comment(text: str) -> str:
    quote: str | None = None
    for index, character in enumerate(text):
        if quote is not None:
            if character == quote:
                quote = None
        elif character in ("'", '"'):
            quote = character
        elif character == "#":
            return text[:index]
    return text


def _scalar_token(text: str) -> str | None:
    value = _remove_unquoted_comment(text).strip()
    if not value:
        return None
    if value[0] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        if end < 0:
            raise ExampleInputError(f"unterminated quoted CIF value: {text.strip()!r}")
        token = value[1:end]
        remainder = _remove_unquoted_comment(value[end + 1 :]).strip()
        if remainder:
            raise ExampleInputError(f"unexpected text after CIF value: {text.strip()!r}")
        return token
    token = value.split(None, 1)[0]
    remainder = value[len(token) :].strip()
    if remainder:
        raise ExampleInputError(f"unexpected text after CIF value: {text.strip()!r}")
    return token


def _read_cif_database_id(path: Path, database_code: str) -> str | None:
    """Read one source ID from CIF metadata without trusting the filename."""

    tag = f"_database_code_{database_code}".casefold()
    active_values: list[str] = []
    commented_pcd_values: list[str] = []
    data_values: list[str] = []
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    in_text_field = False

    for index, line in enumerate(lines):
        if in_text_field:
            if line.startswith(";"):
                in_text_field = False
            continue
        if line.startswith(";"):
            in_text_field = True
            continue

        stripped = line.strip()
        if not stripped:
            continue

        data_match = re.match(r"(?i)^data_(\d+)(?:[-_].*)?$", stripped)
        if database_code == "PCD" and data_match:
            data_values.append(data_match.group(1))

        is_comment = stripped.startswith("#")
        if is_comment and database_code != "PCD":
            continue
        candidate = stripped[1:].lstrip() if is_comment else stripped
        match = re.match(r"(?i)^(_database_code_[A-Za-z0-9_]+)(?:\s+(.*))?$", candidate)
        if match is None or match.group(1).casefold() != tag:
            continue

        raw_value = match.group(2) or ""
        if raw_value.strip():
            value = _scalar_token(raw_value)
        elif is_comment and database_code == "PCD":
            # PCD's CIF export comments out this metadata. Only same-line
            # commented tags are accepted; a nearby ordinary comment is not.
            value = None
        else:
            value = None
            for following in lines[index + 1 :]:
                following = following.strip()
                if not following or following.startswith("#"):
                    continue
                if following.startswith(("_", "loop_", "data_", "save_", "stop_", ";")):
                    break
                value = _scalar_token(following)
                break

        if value in (None, "", ".", "?"):
            continue
        if not re.fullmatch(r"\d+", value):
            raise ExampleInputError(f"invalid _database_code_{database_code} value {value!r} in {path}")
        if is_comment:
            if database_code == "PCD":
                commented_pcd_values.append(value)
        else:
            active_values.append(value)

    if database_code == "PCD" and commented_pcd_values:
        data_id_set = set(data_values)
        unverified = sorted(set(commented_pcd_values) - data_id_set)
        if unverified:
            raise ExampleInputError(
                f"commented _database_code_PCD value(s) {', '.join(unverified)} in {path} "
                "do not match a data_<id> record; refusing to treat an ordinary comment as metadata"
            )
        active_values.extend(commented_pcd_values)

    distinct = sorted(set(active_values))
    if len(distinct) > 1:
        raise ExampleInputError(
            f"conflicting _database_code_{database_code} values in {path}: {', '.join(distinct)}"
        )
    return distinct[0] if distinct else None


def _match_catalog_to_cifs(
    cif_dir: Path,
    database_code: str,
    entries: list[ExampleEntry],
) -> list[tuple[ExampleEntry, Path]]:
    if not cif_dir.is_dir():
        raise ExampleInputError(f"CIF directory does not exist or is not a directory: {cif_dir}")
    cifs = sorted(
        (path.resolve() for path in cif_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".cif"),
        key=lambda item: str(item),
    )
    if not cifs:
        raise ExampleInputError(f"no .cif files found recursively under: {cif_dir}")

    paths_by_id: dict[str, list[Path]] = defaultdict(list)
    for path in cifs:
        database_id = _read_cif_database_id(path, database_code)
        if database_id is not None:
            paths_by_id[database_id].append(path)

    catalog_ids = {entry.database_id for entry in entries}
    duplicates = {
        key: paths
        for key, paths in paths_by_id.items()
        if key in catalog_ids and len(paths) > 1
    }
    if duplicates:
        detail = "; ".join(
            f"{database_code} {database_id}: {', '.join(str(path) for path in paths)}"
            for database_id, paths in sorted(duplicates.items())
        )
        raise ExampleInputError(
            f"duplicate {database_code} database IDs in {cif_dir}: {detail}. "
            "Keep exactly one CIF export for each database ID."
        )
    matched: list[tuple[ExampleEntry, Path]] = []
    missing_catalog_ids: list[str] = []
    for entry in entries:
        paths = paths_by_id.get(entry.database_id)
        if paths is None:
            missing_catalog_ids.append(entry.database_id)
        else:
            matched.append((entry, paths[0]))
    if missing_catalog_ids:
        preview = ", ".join(missing_catalog_ids[:12])
        suffix = "" if len(missing_catalog_ids) <= 12 else f" (and {len(missing_catalog_ids) - 12} more)"
        raise ExampleInputError(
            f"{len(missing_catalog_ids)} catalog {database_code} ID(s) were not found under {cif_dir}: "
            f"{preview}{suffix}. Check that --cif-dir contains a CIF for every required entry."
        )
    return matched


def _manifest_text(matched: list[tuple[ExampleEntry, Path]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("phase_id", "cif_path", "label"))
    for entry, path in matched:
        writer.writerow((entry.phase_id, str(path), entry.label))
    return output.getvalue()


def prepare_example(
    example: str,
    cif_dir: str | Path,
) -> PreparedExample:
    """Match local CIFs and write the manifest used by the example's config."""

    if example not in _EXAMPLE_SPECS:
        supported = ", ".join(sorted(_EXAMPLE_SPECS))
        raise ExampleInputError(f"unknown example {example!r}; choose one of: {supported}")

    spec = _EXAMPLE_SPECS[example]
    catalog_path = _spec_path(spec["catalog"])
    config_path = _spec_path(spec["config"]).resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExampleInputError(f"could not read {config_path}: {exc}") from exc
    inputs = config.get("inputs") if isinstance(config, dict) else None
    manifest_value = inputs.get("phase_manifest") if isinstance(inputs, dict) else None
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ExampleInputError(f"set inputs.phase_manifest in {config_path}")
    manifest_path = Path(os.path.expandvars(manifest_value)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = config_path.parent / manifest_path
    # Resolve the parent while preserving the output symlink for the write check below.
    manifest_path = manifest_path.parent.resolve() / manifest_path.name
    input_dir = Path(cif_dir).expanduser().resolve()
    destination = manifest_path.parent

    entries = _load_catalog(catalog_path)
    matched = _match_catalog_to_cifs(input_dir, spec["database_code"], entries)
    manifest_bytes = _manifest_text(matched).encode("utf-8")

    destination.mkdir(parents=True, exist_ok=True)
    backup_dir = None
    if manifest_path.exists() or manifest_path.is_symlink():
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ExampleInputError(f"prepared output path must be a regular file: {manifest_path}")
        if manifest_path.read_bytes() == manifest_bytes:
            return PreparedExample(example, destination, manifest_path, config_path, len(matched))
        backup_dir = Path(tempfile.mkdtemp(prefix="previous-", dir=destination))
        shutil.copy2(manifest_path, backup_dir / manifest_path.name)

    manifest_path.write_bytes(manifest_bytes)
    return PreparedExample(example, destination, manifest_path, config_path, len(matched), backup_dir)


__all__ = ["ExampleInputError", "PreparedExample", "prepare_example"]

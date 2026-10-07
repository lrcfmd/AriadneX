"""Persistence, hashing, and cache bookkeeping for workflow artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .contracts import StageStatus


def safe_filename(value: str) -> str:
    """Convert a stable ID into a portable artifact filename."""

    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )


def ensure_unique_sanitized_ids(
    values: Iterable[object],
    sanitizer: Callable[[str], str],
    *,
    context: str = "artifact",
) -> None:
    """Reject distinct IDs that map to the same generated name.

    Several workflow stages persist files or directories from user-facing IDs.
    Sanitising those IDs is necessary for portable paths, but silently allowing
    two distinct IDs to share a target would make one artifact overwrite the
    other.  Callers run this check before creating or clearing any output.
    """

    targets: dict[str, list[str]] = {}
    for value in values:
        original = str(value)
        target = str(sanitizer(original))
        originals = targets.setdefault(target, [])
        if original not in originals:
            originals.append(original)
    collisions = {target: originals for target, originals in targets.items() if len(originals) > 1}
    if not collisions:
        return
    details = "; ".join(
        f"{original!r} -> {target!r}"
        for target, originals in sorted(collisions.items())
        for original in originals
    )
    raise ValueError(f"{context} ID collision after sanitization: {details}")


def ensure_managed_output_directory(
    path: str | Path,
    *,
    context: str = "workflow output",
) -> Path:
    """Create an output directory and reject symlinks in its managed tree.

    Workflow stages replace files below their output directory during a
    rebuild.  Following a pre-existing symlink there could overwrite or
    delete data outside the run, so managed output trees never accept links.
    Parent directories may still be symlinks, which keeps ordinary mounted or
    linked workspaces usable.
    """

    directory = Path(path).expanduser()
    if directory.is_symlink():
        raise ValueError(f"{context} directory must not be a symbolic link: {directory}")
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f"{context} path is not a directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    link = next(
        (child for child in sorted(directory.rglob("*"), key=str) if child.is_symlink()), None
    )
    if link is not None:
        raise ValueError(f"{context} contains a symbolic link: {link}")
    return directory


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_fingerprint(path: Path) -> dict[str, Any]:
    if path.is_symlink() and not path.exists():
        return {"path": str(path), "broken_symlink": True}
    if path.is_file():
        return {"path": str(path), "sha256": file_sha256(path)}
    if path.is_dir():
        entries = []
        for child in sorted(path.rglob("*")):
            if child.is_file() or (child.is_symlink() and not child.exists()):
                entries.append(_path_fingerprint(child))
        return {"path": str(path), "entries": entries}
    return {"path": str(path), "missing": True}


def fingerprint_paths(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Describe paths for cache/provenance by their contents."""

    ordered = sorted({Path(item).expanduser().resolve() for item in paths}, key=str)
    return [_path_fingerprint(path) for path in ordered]


def cache_key(
    stage: str,
    payload: Any,
    paths: Iterable[str | Path] = (),
) -> str:
    """Build a deterministic key from a stage name, config payload, and files."""

    material = {
        "stage": stage,
        "payload": payload,
        "paths": fingerprint_paths(paths),
    }
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n")
    temporary.replace(destination)
    return destination


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def artifact_descriptor(artifact: Any) -> dict[str, Any]:
    """Convert a path-based artifact dataclass to a JSON descriptor."""

    value = _jsonable(artifact)
    if not isinstance(value, dict):
        raise TypeError("artifact must serialize to a mapping")
    return value


class ArtifactStore:
    """Own the on-disk layout and lifecycle metadata for one run."""

    STAGE_DIRS = {
        "library": "01_library",
        "mixtures": "02_mixtures",
        "experiment": "03_unmix",
        "unmix": "03_unmix",
        "reports": "04_reports",
    }

    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root)
        self.status_path = self.run_root / "status.json"
        self.provenance_path = self.run_root / "provenance.json"

    def ensure_layout(self) -> None:
        ensure_managed_output_directory(self.run_root, context="run output")
        for directory in set(self.STAGE_DIRS.values()):
            ensure_managed_output_directory(
                self.run_root / directory,
                context=f"workflow stage {directory}",
            )

    def stage_dir(self, name: str) -> Path:
        try:
            directory = self.STAGE_DIRS[name]
        except KeyError as exc:
            raise KeyError(f"unknown workflow stage: {name}") from exc
        path = self.run_root / directory
        return ensure_managed_output_directory(path, context=f"workflow stage {name}")

    def read_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {"schema_version": 1, "stages": {}}
        value = read_json(self.status_path)
        if not isinstance(value, dict):
            raise ValueError(f"invalid status file: {self.status_path}")
        value.setdefault("schema_version", 1)
        value.setdefault("stages", {})
        return value

    def write_stage_status(self, status: StageStatus, *, artifact: Any = None) -> Path:
        payload = self.read_status()
        entry = status.to_dict()
        if artifact is not None:
            entry["artifact"] = artifact_descriptor(artifact)
        payload.setdefault("stages", {})[status.name] = entry
        write_json(self.status_path, payload)
        return self.status_path

    def invalidate_after(self, stage: str) -> None:
        """Retire downstream results before an upstream stage is rebuilt."""

        dependents = {
            "library": ("experiment", "reports"),
            "mixtures": ("experiment", "reports"),
            "experiment": ("reports",),
            "reports": (),
        }
        payload = self.read_status()
        for downstream in dependents[stage]:
            payload["stages"].pop(downstream, None)
        write_json(self.status_path, payload)

    def stage_status(self, name: str) -> StageStatus | None:
        stages = self.read_status().get("stages", {})
        entry = stages.get(name)
        if entry is None and name == "unmix":
            entry = stages.get("experiment")
        if not isinstance(entry, Mapping):
            return None
        return StageStatus.from_dict(entry)

    def write_provenance(self, payload: Mapping[str, Any]) -> Path:
        return write_json(self.provenance_path, payload)

    def read_provenance(self) -> dict[str, Any]:
        if not self.provenance_path.exists():
            return {}
        value = read_json(self.provenance_path)
        if not isinstance(value, dict):
            raise ValueError(f"invalid provenance file: {self.provenance_path}")
        return value

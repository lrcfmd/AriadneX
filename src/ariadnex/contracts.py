"""Stable value objects exchanged by the workflow stages.

The computational modules deliberately communicate through these small, path-based
contracts.  A stage may add files under its artifact directory, but consumers should
only rely on the paths represented here and on the metadata written alongside them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _path(value: Path | str) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LibraryArtifact:
    """A simulated and clustered PXRD basis library."""

    root: Path
    manifest_path: Path
    representatives_path: Path
    basis_patterns_dir: Path
    metadata_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "root",
            "manifest_path",
            "representatives_path",
            "basis_patterns_dir",
            "metadata_path",
        ):
            object.__setattr__(self, name, _path(getattr(self, name)))


@dataclass(frozen=True)
class MixtureArtifact:
    """Cleaned experimental diffraction patterns and signatures."""

    root: Path
    manifest_path: Path
    patterns_path: Path
    signatures_path: Path
    metadata_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for name in ("root", "manifest_path", "patterns_path", "signatures_path", "metadata_path"):
            object.__setattr__(self, name, _path(getattr(self, name)))


@dataclass(frozen=True)
class UnmixArtifact:
    """Numerical output of a single standard or leave-one-out experiment."""

    root: Path
    results_pickle: Path
    results_npz: Path
    assignments_csv: Path
    abundance_csv: Path
    sample_summary_csv: Path
    metadata_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "root",
            "results_pickle",
            "results_npz",
            "assignments_csv",
            "abundance_csv",
            "sample_summary_csv",
            "metadata_path",
        ):
            object.__setattr__(self, name, _path(getattr(self, name)))


@dataclass(frozen=True)
class ReportArtifact:
    """Files generated from an :class:`UnmixArtifact` without re-solving."""

    root: Path
    files: tuple[Path, ...]
    metadata_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _path(self.root))
        object.__setattr__(self, "files", tuple(_path(item) for item in self.files))
        object.__setattr__(self, "metadata_path", _path(self.metadata_path))


@dataclass(frozen=True)
class RunArtifact:
    """The complete set of artifacts and bookkeeping for one workflow run."""

    root: Path
    status_path: Path
    provenance_path: Path
    library: LibraryArtifact
    mixtures: MixtureArtifact
    unmix: UnmixArtifact
    reports: ReportArtifact
    run_id: str = ""
    config_path: Path | None = None

    def __post_init__(self) -> None:
        for name in ("root", "status_path", "provenance_path"):
            object.__setattr__(self, name, _path(getattr(self, name)))
        if self.config_path is not None:
            object.__setattr__(self, "config_path", _path(self.config_path))


@dataclass(frozen=True)
class StageStatus:
    """Serializable lifecycle state for one workflow stage."""

    name: str
    state: str
    started_at: str | None = None
    ended_at: str | None = None
    cache_key: str | None = None
    artifact_path: str | None = None
    error: str | None = None

    @classmethod
    def running(cls, name: str, *, cache_key: str | None = None) -> "StageStatus":
        return cls(name=name, state="running", started_at=_utc_now(), cache_key=cache_key)

    @classmethod
    def completed(
        cls,
        name: str,
        *,
        cache_key: str | None = None,
        artifact_path: str | Path | None = None,
        started_at: str | None = None,
    ) -> "StageStatus":
        return cls(
            name=name,
            state="completed",
            started_at=started_at,
            ended_at=_utc_now(),
            cache_key=cache_key,
            artifact_path=None if artifact_path is None else str(artifact_path),
        )

    @classmethod
    def failed(
        cls,
        name: str,
        error: str,
        *,
        cache_key: str | None = None,
        started_at: str | None = None,
    ) -> "StageStatus":
        return cls(
            name=name,
            state="failed",
            started_at=started_at,
            ended_at=_utc_now(),
            cache_key=cache_key,
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "cache_key": self.cache_key,
            "artifact_path": self.artifact_path,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageStatus":
        return cls(
            name=str(payload["name"]),
            state=str(payload["state"]),
            started_at=payload.get("started_at"),
            ended_at=payload.get("ended_at"),
            cache_key=payload.get("cache_key"),
            artifact_path=payload.get("artifact_path"),
            error=payload.get("error"),
        )

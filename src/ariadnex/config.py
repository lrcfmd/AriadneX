"""Configuration loading and path resolution for the workflow CLI."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from .defaults import custom_defaults


class ConfigError(ValueError):
    """Raised when a workflow YAML cannot be loaded or validated."""


_PATH_KEYS = {
    "cifs",
    "mixtures",
    "instrument",
    "ground_truth",
    "anomaly_reference",
    "phase_manifest",
    "phase_file",
    "raw_cifs",
    "mixed_patterns",
    "basis_patterns",
    "manifest",
    "instrument_file",
    "checkpoint_file",
    "reference",
    "checkpoint",
    "output_root",
    "output_dir",
    "input_dir",
    "input_folder",
    "output_folder",
    "data_root",
    "sample_description_path",
    "basis_prm_path",
    "prm_path",
}


def _looks_like_path(key: str) -> bool:
    key = key.lower()
    return key in _PATH_KEYS or key.endswith(("_path", "_dir", "_folder", "_file", "_manifest"))


def _resolve_value(value: Any, base_dir: Path, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(k): _resolve_value(v, base_dir, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, base_dir, key) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_value(item, base_dir, key) for item in value)
    if not isinstance(value, str) or not _looks_like_path(key):
        return value
    expanded = os.path.expanduser(os.path.expandvars(value))
    if not expanded or expanded.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", expanded):
        return str(Path(expanded)) if expanded else expanded
    # URLs and other URI values are not filesystem paths.
    if "://" in expanded:
        return expanded
    return str((base_dir / expanded).resolve())


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _merge_defaults(defaults: Mapping[str, Any], explicit: Mapping[str, Any]) -> dict[str, Any]:
    values = deepcopy(dict(defaults))
    for key, value in explicit.items():
        if isinstance(value, Mapping) and isinstance(values.get(key), Mapping):
            values[key] = _merge_defaults(values[key], value)
        else:
            values[key] = deepcopy(value)
    return values


def _validate_custom_analysis(analysis: Mapping[str, Any]) -> None:
    """Validate the only numerical inputs exposed by the custom template."""

    unknown = sorted(set(analysis) - {"theta_min", "theta_max"})
    if unknown:
        raise ConfigError(
            "analysis has unknown field(s): " + ", ".join(str(value) for value in unknown)
        )
    lower = analysis.get("theta_min", 15.0)
    upper = analysis.get("theta_max", 75.0)
    if isinstance(lower, bool) or isinstance(upper, bool):
        raise ConfigError("analysis.theta_min and analysis.theta_max must be numbers")
    try:
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError) as exc:
        raise ConfigError("analysis.theta_min and analysis.theta_max must be numbers") from exc
    if not all(math.isfinite(value) for value in (lower, upper)):
        raise ConfigError("analysis.theta_min and analysis.theta_max must be finite numbers")
    if upper <= lower:
        raise ConfigError("analysis.theta_max must be greater than analysis.theta_min")


@dataclass(frozen=True)
class ProjectConfig(Mapping[str, Any]):
    """Resolved configuration shared by all stages, with dotted-key access."""

    source_path: Path
    base_dir: Path
    resolved: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        object.__setattr__(self, "base_dir", Path(self.base_dir).resolve())
        object.__setattr__(self, "resolved", _plain(deepcopy(dict(self.resolved))))

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
        source_path: str | Path | None = None,
    ) -> ProjectConfig:
        """Resolve paths and validate a configuration once at its entry point."""

        if not isinstance(values, Mapping):
            raise ConfigError("configuration root must be a mapping")
        source = Path(source_path).expanduser().resolve() if source_path is not None else None
        base = (
            Path(base_dir).expanduser().resolve()
            if base_dir is not None
            else (source.parent if source is not None else Path.cwd())
        )
        try:
            version = int(values.get("schema_version", 1))
        except (TypeError, ValueError) as exc:
            raise ConfigError("schema_version must be an integer") from exc
        if version != 1:
            raise ConfigError(f"unsupported schema_version {version}; expected 1")
        profile = values.get("profile")
        if profile is not None:
            if profile != "custom":
                raise ConfigError(f"unknown configuration profile: {profile}")
            analysis = values.get("analysis", {})
            if not isinstance(analysis, Mapping):
                raise ConfigError("analysis must be a mapping")
            _validate_custom_analysis(analysis)
            values = _merge_defaults(custom_defaults(analysis), values)
        resolved = _resolve_value(_plain(values), base)
        resolved["schema_version"] = version
        config = cls(source_path=source or base / "config.yaml", base_dir=base, resolved=resolved)
        config.section("project")
        output_root = config.get("project.output_root")
        if output_root is not None and not isinstance(output_root, str):
            raise ConfigError("project.output_root must be a path string")
        return config

    @property
    def values(self) -> Mapping[str, Any]:
        """Alias used by callers that treat a config as a generic mapping."""

        return self.resolved

    @property
    def schema_version(self) -> int:
        return int(self.resolved.get("schema_version", 1))

    @property
    def project_name(self) -> str:
        project = self.resolved.get("project", {})
        if not isinstance(project, Mapping):
            raise ConfigError("project must be a mapping")
        return str(project.get("name", self.source_path.stem))

    @property
    def output_root(self) -> Path:
        value = self.get("project.output_root", "runs")
        path = Path(str(value))
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def config_dir(self) -> Path:
        return self.base_dir

    def __getitem__(self, key: str) -> Any:
        return self.resolved[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.resolved)

    def __len__(self) -> int:
        return len(self.resolved)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.resolved
        for part in dotted_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.get(name, {})
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ConfigError(f"{name} must be a mapping")
        return value

    @property
    def project(self) -> Mapping[str, Any]:
        return self.section("project")

    @property
    def inputs(self) -> Mapping[str, Any]:
        return self.section("inputs")

    @property
    def model(self) -> Mapping[str, Any]:
        return self.section("model")

    @property
    def library(self) -> Mapping[str, Any]:
        return self.section("library")

    @property
    def unmix(self) -> Mapping[str, Any]:
        return self.section("unmix")

    @property
    def experiment(self) -> Mapping[str, Any]:
        return self.section("experiment")

    @property
    def reports(self) -> Mapping[str, Any]:
        return self.section("reports")

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.resolved))


Config = ProjectConfig


def load_config(path: str | Path) -> ProjectConfig:
    """Load and resolve one YAML configuration.

    All configured filesystem paths are made absolute against the directory that
    contains the YAML file.  The input file itself is never rewritten.
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigError(f"configuration file does not exist: {source_path}")
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read configuration {source_path}: {exc}") from exc
    if raw is None:
        raw = {}
    return ProjectConfig.from_mapping(raw, source_path=source_path)


def resolve_config(path: str | Path) -> ProjectConfig:
    """Explicit alias for :func:`load_config`."""

    return load_config(path)


def config_for_run(config: ProjectConfig, run_root: str | Path) -> dict[str, Any]:
    """Serialize project-local paths relative to a run directory.

    A run can then be moved with its source tree and still reload its config.
    Paths outside the project tree stay absolute so a missing external asset
    remains an explicit report or preflight error.
    """

    destination = Path(run_root).expanduser().resolve()
    project_root = config.base_dir.parent.resolve()

    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): visit(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(item) for item in value]
        if isinstance(value, str) and "://" not in value:
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                try:
                    candidate.relative_to(project_root)
                except ValueError:
                    return value
                return os.path.relpath(candidate, destination)
        return value

    return visit(config.as_dict())

"""Library-stage configuration profiles and normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ariadnex.config import ProjectConfig

from .selection import DEFAULT_MIN_ARI


@dataclass(frozen=True)
class FilterProfile:
    """Chemistry/domain policy applied after structural parsing."""

    allowed_elements: tuple[str, ...] | None = None
    required_elements: tuple[str, ...] = ()
    formula: str | None = None
    formula_regex: str | None = None
    metadata_phase_fields: tuple[str, ...] = (
        "_chemical_name_common",
        "_chemical_name_systematic",
    )
    metadata_phase_hard_drop_keywords: tuple[str, ...] = (
        "amorphous",
        "x-ray amorphous",
        "noncrystalline",
        "non-crystalline",
    )
    metadata_phase_filter_enabled: bool = True
    max_ambient_temperature_kelvin: float | None = None
    max_ambient_pressure_gpa: float | None = None
    ambient_condition_filter_enabled: bool = False
    max_sites: int | None = 2048
    audit_creation_date_cutoff: str | None = None
    smact_enabled: bool = False
    smact_mixed_valence: bool = True
    smact_use_pauling_test: bool = False
    smact_include_alloys: bool = False
    smact_fractional_max_denominator: int = 32
    smact_fractional_tolerance: float = 1e-3
    integer_snap_tolerance: float = 2e-3
    simple_ratio_snap_max_coefficient: int = 8
    simple_ratio_snap_relative_tolerance: float = 5e-3
    smact_skip_unstable_fractional_formula: bool = True


_DISCOVERY_FILTER: dict[str, Any] = {
    "max_sites": 2048,
    "metadata_phase_filter_enabled": True,
    "ambient_condition_filter_enabled": True,
    "max_ambient_temperature_kelvin": 1300.0,
    "max_ambient_pressure_gpa": 1.0,
    "smact_enabled": True,
}

_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "custom": {
        "filter": {
            **_DISCOVERY_FILTER,
            "smact_enabled": False,
            "ambient_condition_filter_enabled": False,
            "metadata_phase_filter_enabled": False,
        },
        "dedup": {"enabled": True, "volume_relative_tolerance": 0.01},
    },
    # The controlled source set contains four known phases.
    "controlled_leave_one_out": {
        "filter": {
            "max_sites": 2048,
            "metadata_phase_filter_enabled": True,
            "ambient_condition_filter_enabled": True,
            "max_ambient_temperature_kelvin": 1300.0,
            "max_ambient_pressure_gpa": 1.0,
        },
        "dedup": {"enabled": True, "volume_relative_tolerance": 0.05},
    },
    # The open-world library uses an audit cutoff, ambient hard drops, SMACT
    # validity, and a 1% volume window before clustering.
    "open_world_phase_discovery": {
        "filter": {
            **_DISCOVERY_FILTER,
            "audit_creation_date_cutoff": "2024-01-01",
        },
        "dedup": {"enabled": True, "volume_relative_tolerance": 0.01},
    },
}


@dataclass(frozen=True)
class DedupProfile:
    """Deterministic duplicate policy for formula/space-group/volume groups."""

    enabled: bool = True
    volume_relative_tolerance: float = 0.05


@dataclass(frozen=True)
class ClusteringProfile:
    """Hierarchical clustering selection policy."""

    selection: str = "stable_auto"
    target_clusters: int | None = None
    min_ari: float = DEFAULT_MIN_ARI
    optimal_ordering: bool = True
    threshold_lower_quantile: float = 0.10
    threshold_upper_quantile: float = 0.90
    n_thresholds: int = 60
    min_clusters: int = 1
    max_clusters: int | None = None


def section_value(section_source: Any, key: str, default: Any = None) -> Any:
    """Read one field from a domain profile or its normalized mapping.

    Stage configuration is normalized before it reaches the library.  This
    small compatibility helper remains for filtering code that also accepts
    the public profile dataclasses directly.
    """

    if isinstance(section_source, Mapping):
        return section_source.get(key, default)
    return getattr(section_source, key, default)


def as_tuple(value: Any, *, default: tuple[Any, ...] = ()) -> tuple[Any, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _first(values: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first configured value, preserving explicit falsey values."""

    for key in keys:
        value = values.get(key)
        if value is not None:
            return value
    return default


def _path(value: Any, default: Path | None = None) -> Path | None:
    if value is None or value == "":
        return default
    return Path(value).expanduser()


def _mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("library stage sections must be mappings")
    return value


def _as_int(value: Any, *, name: str) -> int:
    """Convert an integer setting without silently truncating decimals."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
        result = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or numeric != result:
        raise ValueError(f"{name} must be an integer")
    return result


def _merge_section(explicit: Any, defaults: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(defaults)
    values.update(_mapping(explicit))
    return values


def _validate_grid(
    *,
    name: str,
    theta_min: float,
    theta_max: float,
    n_points: int,
) -> None:
    """Reject a grid that would otherwise fail or silently mis-simulate."""

    if not all(math.isfinite(value) for value in (theta_min, theta_max)):
        raise ValueError(f"{name} theta_min and theta_max must be finite numbers")
    if theta_max <= theta_min:
        raise ValueError(f"{name} theta_max must be greater than theta_min")
    if n_points < 2:
        raise ValueError(f"{name} n_points must be at least two")


def resolve_library_settings(
    config: ProjectConfig, output_root: Path | None = None
) -> dict[str, Any]:
    """Normalize one project config into the library stage settings.

    The project loader has already resolved all paths against the YAML
    directory.  This function only applies the library aliases and profile
    defaults that belong to the stage.
    """

    if not isinstance(config, ProjectConfig):
        raise TypeError("library configuration must be a ProjectConfig")

    library = config.library
    model = config.model
    simulation = _mapping(library.get("simulation"))
    basis_simulation = _mapping(library.get("basis_simulation"))

    profile_name = (
        str(_first(library, "profile", default=config.get("library_profile", "")))
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if profile_name and profile_name not in _PROFILE_DEFAULTS:
        raise ValueError(f"unknown library profile: {profile_name}")
    profile_defaults = _PROFILE_DEFAULTS.get(profile_name, {})
    filter_section = _merge_section(
        _first(
            library,
            "filter",
            "filter_profile",
            "domain_filter",
            default=_first(config, "filter", "filter_profile", default={}),
        ),
        profile_defaults.get("filter", {}),
    )
    dedup = _merge_section(
        _first(
            library,
            "dedup",
            "deduplication",
            default=_first(config, "dedup", "deduplication", default={}),
        ),
        profile_defaults.get("dedup", {}),
    )
    clustering = dict(_mapping(_first(library, "clustering", default=config.get("clustering", {}))))

    input_path = _path(
        _first(
            config,
            "input_path",
            "input_dir",
            "cifs",
            "cif_dir",
            "inputs.cifs",
            "inputs.input_path",
        )
    )
    if input_path is None:
        input_path = _path(_first(library, "input_path", "input_dir", "cifs"))

    phase_manifest = _path(
        _first(config, "phase_manifest", "inputs.phase_manifest", "library.phase_manifest")
    )

    instrument_path = _path(
        _first(
            config,
            "instrument_path",
            "instrument",
            "inputs.instrument",
            "inputs.instrument_path",
        )
    )
    if instrument_path is None:
        instrument_path = _path(_first(simulation, "instrument_path", "instrument"))

    # Candidate simulation and selected-basis regeneration may use different
    # instruments.  Both keep the historical inputs.instrument fallback.
    simulation_instrument_path = _path(
        _first(config, "simulation_instrument_path", "simulation_instrument")
    )
    if simulation_instrument_path is None:
        simulation_instrument_path = _path(
            _first(simulation, "instrument", "instrument_path"), instrument_path
        )
    basis_instrument_path = _path(_first(config, "basis_instrument_path", "basis_instrument"))
    if basis_instrument_path is None:
        basis_instrument_path = _path(
            _first(basis_simulation, "instrument", "instrument_path"), instrument_path
        )

    checkpoint_path = _path(
        _first(
            config,
            "checkpoint_path",
            "checkpoint",
            "model.checkpoint",
            "model.checkpoint_path",
            "inputs.checkpoint",
        )
    )
    if checkpoint_path is None:
        checkpoint_path = _path(_first(model, "checkpoint", "checkpoint_path"))

    if output_root is None:
        output_root = _path(
            _first(
                config,
                "output_root",
                "output_dir",
                "library_root",
                "project.output_root",
                "outputs.library",
            )
        )
    if output_root is None:
        output_root = config.output_root
    if output_root is None:
        output_root = _path(_first(library, "output_root", "output_dir"))

    theta_min = float(
        _first(
            config,
            "theta_min",
            "theta.min",
            "simulation.theta_min",
            default=_first(simulation, "theta_min", default=5.0),
        )
    )
    theta_max = float(
        _first(
            config,
            "theta_max",
            "theta.max",
            "simulation.theta_max",
            default=_first(simulation, "theta_max", default=90.0),
        )
    )
    n_points = _as_int(
        _first(
            config,
            "n_points",
            "gridpoints",
            "simulation.n_points",
            "simulation.gridpoints",
            default=_first(simulation, "n_points", "gridpoints", default=4096),
        ),
        name="simulation n_points",
    )
    basis_theta_min = float(
        _first(
            config,
            "basis_theta_min",
            "basis.theta_min",
            "library.basis_simulation.theta_min",
            default=_first(basis_simulation, "theta_min", default=theta_min),
        )
    )
    basis_theta_max = float(
        _first(
            config,
            "basis_theta_max",
            "basis.theta_max",
            "library.basis_simulation.theta_max",
            default=_first(basis_simulation, "theta_max", default=theta_max),
        )
    )
    basis_n_points = _as_int(
        _first(
            config,
            "basis_n_points",
            "basis_gridpoints",
            "library.basis_simulation.n_points",
            "library.basis_simulation.gridpoints",
            default=_first(basis_simulation, "n_points", "gridpoints", default=n_points),
        ),
        name="basis_simulation n_points",
    )
    _validate_grid(
        name="simulation",
        theta_min=theta_min,
        theta_max=theta_max,
        n_points=n_points,
    )
    if "step" in basis_simulation:
        step = float(basis_simulation["step"])
        if not all(math.isfinite(value) for value in (step, basis_theta_min, basis_theta_max)):
            raise ValueError("basis_simulation range and step must be finite numbers")
        if step <= 0 or basis_theta_max <= basis_theta_min:
            raise ValueError("basis_simulation needs a positive step and theta_max > theta_min")
        # Keep the requested bounds exact. The nominal step determines the
        # nearest number of uniformly spaced samples, including both endpoints.
        intervals = (basis_theta_max - basis_theta_min) / step
        basis_n_points = round(intervals) + 1
    _validate_grid(
        name="basis_simulation",
        theta_min=basis_theta_min,
        theta_max=basis_theta_max,
        n_points=basis_n_points,
    )

    batch_size = _as_int(
        _first(
            config,
            "batch_size",
            "model.batch_size",
            "library.embedding.batch_size",
            "embedding.batch_size",
            default=32,
        ),
        name="embedding batch_size",
    )
    if batch_size < 1:
        raise ValueError("embedding batch_size must be at least one")

    return {
        "input_path": input_path,
        "phase_manifest": phase_manifest,
        "output_root": output_root,
        "instrument_path": instrument_path,
        "simulation_instrument_path": simulation_instrument_path,
        "basis_instrument_path": basis_instrument_path,
        "checkpoint_path": checkpoint_path,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "n_points": n_points,
        "basis_theta_min": basis_theta_min,
        "basis_theta_max": basis_theta_max,
        "basis_n_points": basis_n_points,
        "batch_size": batch_size,
        "device": str(_first(config, "device", "model.device", default="cpu")),
        "filter": filter_section,
        "dedup": dedup,
        "clustering": clustering,
        "model": model,
        "library_profile": profile_name,
    }

"""Portable PXRD library construction and anomaly-aware unmixing workflow."""

from ._version import __version__
from .config import Config, ConfigError, ProjectConfig, load_config, resolve_config
from .contracts import (
    LibraryArtifact,
    MixtureArtifact,
    ReportArtifact,
    RunArtifact,
    StageStatus,
    UnmixArtifact,
)
from .runtime import CondaRunner, EncoderRunner, RuntimeSpec, launcher_command
from .workflow import DoctorReport, WorkflowError, doctor, report_run, run_project

__all__ = [
    "__version__",
    "Config",
    "ConfigError",
    "ProjectConfig",
    "load_config",
    "resolve_config",
    "LibraryArtifact",
    "MixtureArtifact",
    "ReportArtifact",
    "RunArtifact",
    "StageStatus",
    "UnmixArtifact",
    "DoctorReport",
    "WorkflowError",
    "doctor",
    "report_run",
    "run_project",
    "CondaRunner",
    "RuntimeSpec",
    "EncoderRunner",
    "launcher_command",
]

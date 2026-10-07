"""Small data objects and shared constants for report generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_ANOMALY_NAMES = {
    "anomaly",
    "anomalous",
    "anomalousarea",
    "anomalousfraction",
    "anomalousareapct",
    "anomalousareapercent",
    "unknown",
    "unidentified",
    "unidentifiedarea",
    "unknowncomponent",
    "__anomaly__",
    "__unknown__",
}
_PHASE_REPORT_FILES = ("phase_recovery_matrix.png",)
_ANOMALY_REPORT_FILES = (
    "anomaly_profiles.npz",
    "anomaly_similarity.csv",
    "anomaly_summary.png",
    "anomaly_ranking_labels.csv",
    "anomaly_ranking_metrics.csv",
    "anomaly_precision_at_k.csv",
    "anomaly_reference_similarity.csv",
)
_SAMPLE_RECONSTRUCTION_DIR = "sample_reconstructions"
_OPEN_WORLD_SAMPLE_ID = re.compile(r"(NH\d+)", re.IGNORECASE)
_RASTER_DPI = 600
_OBSERVED_COLOR = "#262524"
_KNOWN_COLOR = "#A9A5A0"
_ANOMALY_COLOR = "#D55E00"
_REFERENCE_COLOR = "#7B6A9E"
_BASELINE_FILL_COLOR = "#F0EDE8"
_BASELINE_LINE_COLOR = "#C8C1B9"
_PHASE_COLORS = ("#B7791F", "#009E73", "#CC79A7", "#7B6A9E", "#4C78A8", "#C95B86")


@dataclass(frozen=True)
class _BasisProfile:
    x: np.ndarray
    y: np.ndarray
    positions: np.ndarray
    weights: np.ndarray


@dataclass
class _Sample:
    sample_id: str
    abundances: dict[str, float] = field(default_factory=dict)
    selected_phase_ids: list[str] = field(default_factory=list)
    anomaly_pct: float | None = None
    x: np.ndarray | None = None
    mixture: np.ndarray | None = None
    raw_x: np.ndarray | None = None
    raw_y: np.ndarray | None = None
    baseline: np.ndarray | None = None
    known: np.ndarray | None = None
    anomaly: np.ndarray | None = None
    total: np.ndarray | None = None
    signature_positions: np.ndarray | None = None
    signature_weights: np.ndarray | None = None
    known_signature: np.ndarray | None = None
    anomaly_signature: np.ndarray | None = None
    total_signature: np.ndarray | None = None
    component_signatures: dict[str, np.ndarray] = field(default_factory=dict)
    phase_fingerprints: dict[str, np.ndarray] = field(default_factory=dict)
    anomaly_fingerprint: np.ndarray | None = None
    reconstruction_sigma: float | None = None
    fingerprint_sigma: float | None = None
    objective: float | None = None
    solver_status: str = ""
    warnings: str = ""
    summary_fields: dict[str, str] = field(default_factory=dict)


@dataclass
class _ResultData:
    samples: dict[str, _Sample]
    phase_order: list[str]
    labels: dict[str, str] = field(default_factory=dict)
    phase_aliases: dict[tuple[tuple[str, float], ...], str] = field(default_factory=dict)
    phase_formula_keys: dict[str, tuple[tuple[str, float], ...]] = field(default_factory=dict)
    basis_profiles: dict[str, _BasisProfile] = field(default_factory=dict)

"""Numerical defaults for user-supplied CIFs and measurements.

The paper configurations keep their explicit experimental settings. Custom runs
use the open-world numerical settings with a general-purpose library policy.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

_CUSTOM_DEFAULTS = {
    "library": {
        "profile": "custom",
        "simulation": {"theta_min": 5.0, "theta_max": 90.0, "gridpoints": 4096},
        "basis_simulation": {"theta_min": 15.0, "theta_max": 75.0, "step": 0.02},
        "embedding": {
            "theta_min": 5.0,
            "theta_max": 90.0,
            "gridpoints": 4096,
            "feature_space": "raw_cls",
            "distance": "cosine",
            "normalization": "l2",
            "batch_size": 64,
        },
        "clustering": {
            "selection": "stable_auto",
            "min_ari": 0.96,
            "stable_regime": {"interval_selection": "penultimate"},
            "stability": {"repeats": 50, "sample_fraction": 0.8, "random_seed": 42},
        },
    },
    "unmix": {
        "allow_anomaly": True,
        "element_prior": {"enabled": False},
        "preprocessing": {
            "background": {"basis": "psalsa", "mixture": "none"},
            "theta_min": 15.0,
            "theta_max": 75.0,
            "gridpoints": 0,
            "signature_points": 72,
            "signature_reduce_mode": "dp_sse",
            "center": False,
            "smooth": False,
            "denoise": {"basis": "savgol", "mixture": "savgol", "window": 7, "polyorder": 2},
            "signature_cleaning": {
                "basis": "none",
                "mixture": {
                    "mode": "adaptive_noise",
                    "keep_top_k": 16,
                    "min_components": 40,
                    "unsupported_strength_mult": 1.2,
                    "strong_quantile_low_snr": 0.85,
                    "strong_quantile_high_snr": 0.8,
                },
            },
        },
        "solver": {
            "formulation": "milp",
            "max_base": "auto",
            "resolution": 0.02,
            "beta": 2.5,
            "soft_sparse_penalty": 0.125,
            "missing_element_penalty": 0.0,
            "width_prior_mode": "inst_width_tau_v1",
            "delta_tau": 1.5,
            "delta_floor": 0.03,
            "width_floor_abs": 0.045,
            "width_budget_mode": "global_avg",
            "strict_solver": True,
            "threads": 32,
            "max_base_extra": 0,
        },
    },
    "experiment": {"type": "standard"},
    "reports": {
        "profile": "final_required_v1",
        "phase_recovery": True,
        "anomaly": True,
        "reconstruction_sigma": "auto",
        "fingerprint_sigma": "auto",
    },
}


def custom_defaults(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Expand the small user-facing analysis range into stage settings."""

    values = deepcopy(_CUSTOM_DEFAULTS)
    values["library"]["simulation"]["instrument"] = str(Path(__file__).with_name("encoder.instprm"))
    for key in ("theta_min", "theta_max"):
        if key in analysis:
            values["library"]["basis_simulation"][key] = analysis[key]
            values["unmix"]["preprocessing"][key] = analysis[key]
    return values

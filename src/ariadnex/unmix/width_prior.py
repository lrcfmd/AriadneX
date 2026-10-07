"""Instrument-aware width estimates and local transport budgets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import signal

_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def load_instprm(path: str | Path) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = (item.strip() for item in line.split(":", 1))
        values[key] = float(value) if _FLOAT_RE.fullmatch(value) else value
    return values


def caglioti_fwhm_proxy(two_theta, params: Mapping[str, Any]):
    two_theta = np.asarray(two_theta, dtype=float)
    theta = np.deg2rad(two_theta / 2.0)
    tan_theta = np.tan(theta)
    cos_theta = np.clip(np.cos(theta), 1e-6, None)
    gaussian = np.clip(
        float(params.get("U", 0.0)) * tan_theta**2
        + float(params.get("V", 0.0)) * tan_theta
        + float(params.get("W", 0.0)),
        1e-12,
        None,
    )
    lorentz = np.clip(
        float(params.get("X", 0.0)) / cos_theta
        + float(params.get("Y", 0.0)) * tan_theta
        + float(params.get("Z", 0.0)),
        0.0,
        None,
    )
    return np.sqrt(gaussian) + lorentz


def _default_prominence(y):
    y = np.asarray(y, dtype=float)
    if not y.size:
        return 0.0
    upper = float(np.percentile(y, 95))
    return max((upper - float(np.median(y))) * 0.08, upper * 0.015, 1e-8)


def estimate_peak_envelope(x, y, *, width_floor_abs: float = 0.045, rel_height: float = 0.5):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 5:
        return np.array([]), np.array([])
    peaks, _ = signal.find_peaks(y, prominence=_default_prominence(y))
    if not peaks.size:
        return np.array([]), np.array([])
    widths = signal.peak_widths(y, peaks, rel_height=float(rel_height))
    grid_step = float(np.median(np.diff(x))) if x.size > 1 else float(width_floor_abs)
    return x[peaks], np.maximum(widths[0] * grid_step, float(width_floor_abs))


def widths_at_positions(
    x, y, positions, *, width_floor_abs: float = 0.045, rel_height: float = 0.5
):
    positions = np.asarray(positions, dtype=float)
    if not positions.size:
        return np.array([])
    peaks, widths = estimate_peak_envelope(
        x, y, width_floor_abs=width_floor_abs, rel_height=rel_height
    )
    if not peaks.size:
        return np.full(positions.size, float(width_floor_abs))
    nearest = np.abs(positions[:, None] - peaks[None, :]).argmin(axis=1)
    return np.maximum(widths[nearest], float(width_floor_abs))


def signature_widths_from_pattern(
    x, y, positions, *, width_floor_abs: float = 0.045, rel_height: float = 0.5
):
    return widths_at_positions(
        x, y, positions, width_floor_abs=width_floor_abs, rel_height=rel_height
    ), {
        "source": "pattern",
        "width_floor_abs": float(width_floor_abs),
        "rel_height": float(rel_height),
    }


def signature_widths_from_prm(
    x,
    y,
    positions,
    *,
    prm_path: str | Path,
    width_floor_abs: float = 0.045,
    rel_height: float = 0.5,
):
    positions = np.asarray(positions, dtype=float)
    params = load_instprm(prm_path)
    empirical_positions, empirical_widths = estimate_peak_envelope(
        x, y, width_floor_abs=width_floor_abs, rel_height=rel_height
    )
    proxy = caglioti_fwhm_proxy(positions, params)
    if empirical_positions.size:
        denominator = float(np.median(caglioti_fwhm_proxy(empirical_positions, params)))
        scale = float(np.median(empirical_widths) / denominator) if denominator > 0 else 1.0
    else:
        denominator = float(np.median(proxy)) if proxy.size else 0.0
        scale = float(width_floor_abs / denominator) if denominator > 0 else 1.0
    return np.maximum(proxy * scale, float(width_floor_abs)), {
        "source": "prm",
        "prm_path": str(prm_path),
        "width_floor_abs": float(width_floor_abs),
        "rel_height": float(rel_height),
        "prm_scale": float(scale),
    }


def coerce_widths(widths, positions, width_floor_abs: float = 0.045):
    positions = np.asarray(positions, dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError("basis positions must contain only finite values")
    if not np.isfinite(float(width_floor_abs)) or float(width_floor_abs) < 0.0:
        raise ValueError("width_floor_abs must be finite and non-negative")
    if widths is None:
        return np.full(positions.size, float(width_floor_abs))
    values = np.asarray(widths, dtype=float).reshape(-1)
    if values.size != positions.size:
        raise ValueError(
            f"signature widths length {values.size} does not match positions length {positions.size}"
        )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("signature widths must be finite and non-negative")
    return np.maximum(values, float(width_floor_abs))


def build_local_disp_budgets(
    basis_positions,
    basis_widths=None,
    *,
    delta_tau: float = 1.0,
    delta_floor: float = 0.0,
    width_floor_abs: float = 0.045,
):
    positions = np.asarray(basis_positions, dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError("basis positions must contain only finite values")
    for name, value in (("delta_tau", delta_tau), ("delta_floor", delta_floor)):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if positions.size == 0:
        return np.array([]), np.array([])
    widths = coerce_widths(basis_widths, positions, width_floor_abs)
    budgets = float(delta_floor) + float(delta_tau) * widths
    return np.maximum(budgets, float(delta_floor)).astype(float), widths.astype(float)


__all__ = [
    "load_instprm",
    "caglioti_fwhm_proxy",
    "estimate_peak_envelope",
    "widths_at_positions",
    "signature_widths_from_pattern",
    "signature_widths_from_prm",
    "coerce_widths",
    "build_local_disp_budgets",
]

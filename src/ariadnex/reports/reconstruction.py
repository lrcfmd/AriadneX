"""Continuous curve reconstruction used by the report layer."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from ..config import ProjectConfig
from .common import _float
from .models import _ResultData, _Sample


def _rebuild_signature(
    x: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Rebuild a continuous PXRD curve from a discrete signature.

    This is the reflected-edge, mass-preserving Gaussian reconstruction used
    for the manuscript figures. Reflection prevents intensity loss for peaks
    close to either edge of the measured two-theta window.
    """

    grid = np.asarray(x, dtype=float).reshape(-1)
    centres = np.asarray(positions, dtype=float).reshape(-1)
    masses = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    if not math.isfinite(float(sigma)) or sigma <= 0.0:
        raise ValueError("reconstruction sigma must be finite and positive")
    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(centres)):
        raise ValueError("reconstruction axis or peak positions contain non-finite values")
    if not np.all(np.isfinite(masses)):
        raise ValueError("reconstruction weights contain non-finite values")
    if grid.size == 0 or centres.size != masses.size:
        return np.zeros_like(grid)
    curve = np.zeros_like(grid)
    for centre, mass in zip(centres, masses):
        if mass <= 0.0:
            continue
        gaussian = np.exp(-0.5 * np.square((grid - centre) / sigma))
        gaussian += np.exp(-0.5 * np.square((grid - (2.0 * grid[0] - centre)) / sigma))
        gaussian += np.exp(-0.5 * np.square((grid - (2.0 * grid[-1] - centre)) / sigma))
        normalizer = float(gaussian.sum())
        if normalizer > 0.0:
            curve += mass * gaussian / normalizer
    return curve


def _sigma_fit_records(
    samples: Sequence[_Sample],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for sample in samples:
        if sample.x is None or sample.mixture is None or sample.signature_positions is None:
            continue
        weights = sample.signature_weights
        if weights is None:
            continue
        x = np.asarray(sample.x, dtype=float).reshape(-1)
        observed = np.maximum(np.asarray(sample.mixture, dtype=float).reshape(-1), 0.0)
        positions = np.asarray(sample.signature_positions, dtype=float).reshape(-1)
        weights = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
        if x.size < 3 or x.size != observed.size or positions.size != weights.size:
            continue
        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(observed))
            and np.all(np.isfinite(positions))
            and np.all(np.isfinite(weights))
        ):
            raise ValueError(f"sample {sample.sample_id!r} contains non-finite reconstruction data")
        observed_mass = float(observed.sum())
        signature_mass = float(weights.sum())
        if observed_mass <= 0.0 or signature_mass <= 0.0:
            continue
        records.append((x, observed / observed_mass, positions, weights / signature_mass))
    return records


def _basis_sigma_fit_records(
    data: _ResultData,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    records = _sigma_fit_records(list(data.samples.values()))
    for profile in data.basis_profiles.values():
        observed_mass = float(profile.y.sum())
        signature_mass = float(profile.weights.sum())
        if not (
            np.all(np.isfinite(profile.x))
            and np.all(np.isfinite(profile.y))
            and np.all(np.isfinite(profile.positions))
            and np.all(np.isfinite(profile.weights))
        ):
            raise ValueError("basis profile contains non-finite reconstruction data")
        if observed_mass <= 0.0 or signature_mass <= 0.0:
            continue
        records.append(
            (
                profile.x,
                profile.y / observed_mass,
                profile.positions,
                profile.weights / signature_mass,
            )
        )
    return records


def _fit_sigma_records(
    records: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    sample_limit: int | None = None,
) -> tuple[float | None, float | None]:
    if not records:
        return None, None
    selected = list(records)
    if sample_limit is not None and len(selected) > sample_limit:
        random = np.random.RandomState(0)
        indices = random.choice(len(selected), size=sample_limit, replace=False)
        selected = [selected[int(index)] for index in indices]
    bound_records = selected[:10] if sample_limit is not None else selected
    spacings = np.concatenate([np.diff(x) for x, _observed, _positions, _weights in bound_records])
    spacings = spacings[np.isfinite(spacings) & (spacings > 0.0)]
    if spacings.size == 0:
        return None, None
    q1, q3 = np.percentile(spacings, [25.0, 75.0])
    lower = max(float(q1) / 10.0, 1e-6)
    upper = max(float(q3) * 10.0, lower * 10.0)

    def cost(sigma: float) -> float:
        error = 0.0
        for x, observed, positions, weights in selected:
            rebuilt = _rebuild_signature(x, positions, weights, sigma)
            error += float(np.square(observed - rebuilt).sum())
        return error

    result = minimize_scalar(
        cost,
        bounds=(lower, upper),
        method="bounded",
        options={
            "xatol": 1e-4 if sample_limit is not None else max(lower * 1e-3, 1e-7),
            "maxiter": 100,
        },
    )
    sigma = float(result.x)
    rmse = math.sqrt(float(result.fun) / sum(record[0].size for record in selected))
    return sigma, rmse


def _fit_reconstruction_sigma(samples: Sequence[_Sample]) -> tuple[float | None, float | None]:
    return _fit_sigma_records(_sigma_fit_records(samples))


def _rebuild_result_profiles(
    data: _ResultData,
    config: ProjectConfig,
) -> tuple[float | None, float | None, float | None, float | None, str]:
    configured = config.reports.get("reconstruction_sigma", "auto")
    if configured is None or str(configured).strip().lower() == "auto":
        sigma, rmse = _fit_reconstruction_sigma(list(data.samples.values()))
    else:
        parsed = _float(configured)
        if parsed is None or parsed <= 0.0:
            raise ValueError("reports.reconstruction_sigma must be 'auto' or a positive number")
        sigma, rmse = float(parsed), None
    if sigma is None:
        return None, rmse, None, None, "result_grid"

    fingerprint_setting = config.reports.get("fingerprint_sigma", "auto")
    if fingerprint_setting is None or str(fingerprint_setting).strip().lower() == "auto":
        fingerprint_sigma, fingerprint_rmse = _fit_sigma_records(
            _basis_sigma_fit_records(data),
            sample_limit=50,
        )
    else:
        parsed = _float(fingerprint_setting)
        if parsed is None or parsed <= 0.0:
            raise ValueError("reports.fingerprint_sigma must be 'auto' or a positive number")
        fingerprint_sigma, fingerprint_rmse = float(parsed), None
    if fingerprint_sigma is None:
        fingerprint_sigma = sigma

    rebuilt_count = 0
    for sample in data.samples.values():
        if sample.x is None or sample.signature_positions is None:
            continue
        x = np.asarray(sample.x, dtype=float).reshape(-1)
        positions = np.asarray(sample.signature_positions, dtype=float).reshape(-1)
        if sample.known_signature is not None:
            sample.known = _rebuild_signature(x, positions, sample.known_signature, sigma)
        if sample.anomaly_signature is not None:
            sample.anomaly = _rebuild_signature(x, positions, sample.anomaly_signature, sigma)
        if sample.total_signature is not None:
            sample.total = _rebuild_signature(x, positions, sample.total_signature, sigma)
        elif sample.known is not None and sample.anomaly is not None:
            sample.total = sample.known + sample.anomaly
        sample.phase_fingerprints = {}
        for phase_id, component in sample.component_signatures.items():
            basis = data.basis_profiles.get(phase_id)
            if basis is not None:
                curve = _rebuild_signature(
                    x,
                    basis.positions,
                    basis.weights,
                    fingerprint_sigma,
                )
            else:
                curve = _rebuild_signature(x, positions, component, fingerprint_sigma)
            sample.phase_fingerprints[phase_id] = curve
        if sample.anomaly_signature is not None:
            sample.anomaly_fingerprint = _rebuild_signature(
                x,
                positions,
                sample.anomaly_signature,
                fingerprint_sigma,
            )
        sample.reconstruction_sigma = sigma
        sample.fingerprint_sigma = fingerprint_sigma
        rebuilt_count += 1
    grid_name = "experimental_pxrd" if rebuilt_count else "result_grid"
    return sigma, rmse, fingerprint_sigma, fingerprint_rmse, grid_name

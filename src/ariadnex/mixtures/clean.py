"""Deterministic experimental PXRD preprocessing.

The pipeline crops or resamples the signal, subtracts the background, applies
optional denoising, normalises nonnegative intensity, reduces it to a sparse
signature, cleans the signature, and estimates instrument-aware peak widths.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import ndimage as scipy_ndimage
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from pybaselines import Baseline
from skimage.restoration import denoise_wavelet, estimate_sigma

DEFAULT_SIG_CLEAN_MODE = "adaptive_noise"
DEFAULT_SIG_CLEAN_MIN_THRESH = 2.5e-4
DEFAULT_SIG_CLEAN_MAX_THRESH = 3.0e-3
DEFAULT_SIG_CLEAN_SNR_REF = 8.0
DEFAULT_SIG_CLEAN_PEAK_SIGMA = 3.0
DEFAULT_SIG_CLEAN_KEEP_TOP_K = 10
DEFAULT_SIG_CLEAN_MIN_COMPONENTS = 8
DEFAULT_SIG_CLEAN_SMOOTH_FRACTION = 0.02
DEFAULT_SIG_CLEAN_WIDTH_REL_HEIGHT = 0.8
DEFAULT_SIG_CLEAN_STRONG_QUANTILE_LOW_SNR = 0.88
DEFAULT_SIG_CLEAN_STRONG_QUANTILE_HIGH_SNR = 0.72
DEFAULT_SIG_CLEAN_SUPPORT_GRID_MULT = 3.0
DEFAULT_SIG_CLEAN_SUPPORT_WIDTH_MULT = 0.40
DEFAULT_SIG_CLEAN_UNSUPPORTED_STRENGTH_MULT = 1.75
THETA_COVERAGE_TOLERANCE = 1e-5
DEFAULT_DENOISE_METHOD = "none"
DEFAULT_DENOISE_WINDOW = 7
DEFAULT_DENOISE_POLYORDER = 2
DEFAULT_WIDTH_FLOOR_ABS = 0.045
DEFAULT_SIG_REDUCE_MODE = "merge"


def _odd_window(
    length: int, fraction: float = DEFAULT_SIG_CLEAN_SMOOTH_FRACTION, minimum: int = 9
) -> int:
    window = max(int(minimum), int(length * fraction))
    if window % 2 == 0:
        window += 1
    if length <= 3:
        return max(1, length)
    if window >= length:
        window = length - 1
        if window % 2 == 0:
            window -= 1
    return max(window, 3)


def smooth_y(y, window: int = 18):
    """Historical centred moving average used by the SOTU preprocessing path."""

    y = np.asarray(y, dtype=float)
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window >= y.size:
        window = y.size - 1
        if window % 2 == 0:
            window -= 1
    if window < 1:
        return y.copy()
    # ``min_periods=1`` is significant at the two ends of the pattern.  Use
    # pandas exactly as the historical implementation did; a padded
    # convolution gives different edge values.
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def fit_background_min(x, y):
    floor = max(float(np.nanmin(y)), 0.0)
    return np.full(np.asarray(y).shape, floor, dtype=float)


def fit_background_rb(
    x,
    y,
    n: int | None = None,
    ry: float | None = None,
    xratio: float = 0.025,
    yratio: float = 0.0125,
):
    """A NumPy implementation of the rolling-ball lower-envelope fitter."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y.copy()
    if n is None:
        n = max(1, int(xratio * x.size))
    n = max(1, int(n))
    if n % 2 == 0:
        n += 1
    m = n // 2
    if ry is None:
        ry = float(yratio) * float(np.nanmax(y) - np.nanmin(y))
    u = np.linspace(-1.0, 1.0, n)
    c = float(ry) * (np.sin(np.arccos(np.clip(u, -1.0, 1.0))) - 1.0)
    result = np.empty_like(y, dtype=float)
    for i in range(y.size):
        left = max(0, i - m)
        right = min(y.size, i + m + 1)
        c_left = m - i if i <= m else 0
        c_right = c_left + (right - left)
        result[i] = np.min(y[left:right] - c[c_left:c_right])
    return result


def _cycle_clockwise(vertices):
    vertices = list(vertices)
    while vertices and vertices[0] != 0:
        vertices.append(vertices.pop(0))
    return vertices


def fit_background_cv(x, y):
    """Historical lower-convex-hull baseline used by SOTU."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    points = np.array([x, y]).T
    hull = ConvexHull(points)
    vertices = _cycle_clockwise(hull.vertices)
    end = vertices.index(y.size - 1)
    line = points[vertices[: end + 1]]
    return interp1d(line[:, 0], line[:, 1])(x)


def fit_background_cvrb(x, y):
    try:
        convex = fit_background_cv(x, y)
        rolling = fit_background_rb(x, np.asarray(y) - convex)
        return convex + rolling
    except Exception:
        return fit_background_min(x, y)


def fit_background_psalsa(x, y):
    """Fit the pybaselines psalsa baseline with its default parameters."""

    return Baseline(np.asarray(x, dtype=float)).psalsa(np.asarray(y, dtype=float))[0]


def background_function(name: Any):
    if callable(name):
        return name
    key = "" if name is None else str(name).strip().lower()
    if key in {"", "none", "null"}:
        return None
    if key in {"min_const", "min", "constant"}:
        return fit_background_min
    if key in {"lowerconvexhull", "convex_hull", "cv"}:
        return fit_background_cv
    if key in {"rollingball", "rolling_ball", "rb"}:
        return fit_background_rb
    if key in {"cv+rb", "cvrb", "convex_rolling"}:
        return fit_background_cvrb
    if key == "psalsa":
        return fit_background_psalsa
    raise ValueError(f"Unsupported background method: {name!r}")


def denoise_signal(
    y,
    method: Any = DEFAULT_DENOISE_METHOD,
    *,
    window: int = DEFAULT_DENOISE_WINDOW,
    polyorder: int = DEFAULT_DENOISE_POLYORDER,
    wavelet: str = "sym8",
    wavelet_method: str = "BayesShrink",
    wavelet_mode: str = "soft",
    wavelet_levels: int | None = None,
    rescale_sigma: bool = True,
    sigma_scale: float = 1.0,
):
    signal = np.asarray(y, dtype=float)
    mode = "none" if method is None else str(method).lower()
    if mode in {"none", ""}:
        return signal.copy()
    if mode in {"savgol", "savitzky_golay"}:
        if signal.size < 3:
            return signal.copy()
        width = max(3, int(window))
        if width % 2 == 0:
            width += 1
        width = min(width, signal.size if signal.size % 2 else signal.size - 1)
        if width < 3:
            return signal.copy()
        result = scipy_signal.savgol_filter(signal, width, min(int(polyorder), width - 1))
        return np.maximum(np.asarray(result, dtype=float), 0.0)
    if mode == "wavelet":
        sigma = float(estimate_sigma(signal, channel_axis=None, average_sigmas=True)) * max(
            float(sigma_scale), 0.0
        )
        result = denoise_wavelet(
            signal,
            sigma=sigma or None,
            wavelet=wavelet,
            method=wavelet_method,
            mode=wavelet_mode,
            wavelet_levels=wavelet_levels,
            rescale_sigma=bool(rescale_sigma),
            channel_axis=None,
        )
        return np.maximum(np.asarray(result, dtype=float), 0.0)
    raise ValueError(f"Unsupported denoise method: {method!r}")


def _merge_adjacent(positions, weights, target: int):
    positions = np.asarray(positions, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if positions.size != weights.size:
        raise ValueError("Signature positions and weights must have the same length")
    if positions.size <= target:
        return positions, weights / max(float(weights.sum()), 1e-30)
    order = np.argsort(positions, kind="mergesort")
    positions, weights = positions[order], weights[order]
    while positions.size > target:
        gaps = np.diff(positions)
        idx = int(np.argmin(gaps))
        left, right = idx, idx + 1
        mass = weights[left] + weights[right]
        if mass <= 0:
            merged_position = (positions[left] + positions[right]) / 2.0
        else:
            merged_position = (
                positions[left] * weights[left] + positions[right] * weights[right]
            ) / mass
        positions[left] = merged_position
        weights[left] = mass
        positions = np.delete(positions, right)
        weights = np.delete(weights, right)
    weights = np.maximum(weights, 0.0)
    return positions, weights / max(float(weights.sum()), 1e-30)


def _prepare_sig_for_fixed_k(positions, weights, target: int, epsilon: float = 1e-19):
    """Prepare a signature exactly as ``SOT.reduce_sig_dp_sse`` does."""

    positions = np.asarray(positions, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if positions.size != weights.size:
        raise ValueError("Signature positions and weights must have the same length")
    keep = weights <= float(epsilon)
    # The old implementation deliberately keeps zero-weight samples when
    # removing them would leave fewer than Nδ support points.
    if keep.size - int(np.sum(keep)) < int(target):
        keep = np.zeros(positions.size, dtype=bool)
    positions, weights = positions[~keep], weights[~keep]
    if positions.size == 0:
        return positions, weights
    order = np.argsort(positions)
    return positions[order], weights[order]


def _weighted_interval_prefixes(positions, weights):
    return (
        np.concatenate([[0.0], np.cumsum(weights)]),
        np.concatenate([[0.0], np.cumsum(weights * positions)]),
        np.concatenate([[0.0], np.cumsum(weights * positions * positions)]),
    )


def _interval_sse_cost(prefix_w, prefix_wx, prefix_wx2, start: int, end: int):
    total_weight = prefix_w[end + 1] - prefix_w[start]
    if total_weight <= 0:
        return 0.0
    total_wx = prefix_wx[end + 1] - prefix_wx[start]
    total_wx2 = prefix_wx2[end + 1] - prefix_wx2[start]
    return float(max(total_wx2 - (total_wx * total_wx) / total_weight, 0.0))


def _interval_weighted_centroid(prefix_w, prefix_wx, start: int, end: int):
    total_weight = prefix_w[end + 1] - prefix_w[start]
    if total_weight <= 0:
        return 0.0, 0.0
    total_wx = prefix_wx[end + 1] - prefix_wx[start]
    return float(total_wx / total_weight), float(total_weight)


def _dp_optimal_partition(n_points: int, n_clusters: int, cost_fn):
    n_clusters = min(int(n_clusters), int(n_points))
    previous = np.full(n_points + 1, np.inf, dtype=float)
    previous[0] = 0.0
    back = np.full((n_clusters + 1, n_points + 1), -1, dtype=np.int32)

    for cluster_index in range(1, n_clusters + 1):
        current = np.full(n_points + 1, np.inf, dtype=float)

        def solve(left, right, opt_left, opt_right):
            if left > right:
                return
            middle = (left + right) // 2
            start = max(cluster_index - 1, opt_left)
            end = min(middle - 1, opt_right)
            best_cost = np.inf
            best_split = -1
            for split in range(start, end + 1):
                candidate = previous[split] + cost_fn(split, middle - 1)
                if candidate < best_cost:
                    best_cost = candidate
                    best_split = split
            current[middle] = best_cost
            back[cluster_index, middle] = best_split
            solve(left, middle - 1, opt_left, best_split)
            solve(middle + 1, right, best_split, opt_right)

        solve(cluster_index, n_points, cluster_index - 1, n_points - 1)
        previous = current

    intervals = []
    end = n_points
    for cluster_index in range(n_clusters, 0, -1):
        start = int(back[cluster_index, end])
        intervals.append((start, end - 1))
        end = start
    intervals.reverse()
    return intervals


def _reduce_sig_dp_sse(positions, weights, target: int, epsilon: float = 1e-19):
    positions, weights = _prepare_sig_for_fixed_k(positions, weights, target, epsilon)
    if positions.size <= int(target):
        return positions, weights
    prefix_w, prefix_wx, prefix_wx2 = _weighted_interval_prefixes(positions, weights)
    intervals = _dp_optimal_partition(
        positions.size,
        int(target),
        lambda start, end: _interval_sse_cost(prefix_w, prefix_wx, prefix_wx2, start, end),
    )
    reduced_positions = []
    reduced_weights = []
    for start, end in intervals:
        position, weight = _interval_weighted_centroid(prefix_w, prefix_wx, start, end)
        reduced_positions.append(position)
        reduced_weights.append(weight)
    return np.asarray(reduced_positions, dtype=float), np.asarray(reduced_weights, dtype=float)


def reduce_signature(
    positions,
    weights,
    *,
    points: int | None = None,
    sig_tol: float | None = None,
    sig_max_sticks: int | None = None,
    mode: str = DEFAULT_SIG_REDUCE_MODE,
):
    """Reduce a non-negative signature with the selected SOTU reducer."""

    positions = np.asarray(positions, dtype=float).reshape(-1)
    weights = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    if positions.size == 0:
        return positions, weights
    order = np.argsort(positions, kind="mergesort")
    positions, weights = positions[order], weights[order]
    reduce_mode = str(mode).lower()
    total = float(weights.sum())
    if total <= 0:
        weights = np.full(positions.size, 1.0 / positions.size)
    elif reduce_mode != "dp_sse":
        weights = weights / total
    if sig_tol is not None:
        tolerance = max(float(sig_tol), 0.0)
        while positions.size > 1 and float(np.min(np.diff(positions))) <= tolerance:
            idx = int(np.argmin(np.diff(positions)))
            mass = weights[idx] + weights[idx + 1]
            positions[idx] = (
                positions[idx] * weights[idx] + positions[idx + 1] * weights[idx + 1]
            ) / max(mass, 1e-30)
            weights[idx] = mass
            positions = np.delete(positions, idx + 1)
            weights = np.delete(weights, idx + 1)
    limit = points
    if sig_max_sticks is not None:
        limit = int(sig_max_sticks) if limit is None else min(int(limit), int(sig_max_sticks))
    if limit is not None and int(limit) > 0:
        if reduce_mode == "dp_sse":
            positions, weights = _reduce_sig_dp_sse(positions, weights, int(limit))
        elif reduce_mode in {"merge", "none", ""}:
            positions, weights = _merge_adjacent(positions, weights, int(limit))
        else:
            raise ValueError(f"Unsupported signature reduction mode: {mode!r}")
    if reduce_mode == "dp_sse":
        # The historical reducer receives the dense curve after preprocessing
        # has normalized it.  It preserves that mass through the DP and does
        # not apply another normalization pass here.
        return positions, weights
    return positions, weights / max(float(weights.sum()), 1e-30)


def _smooth_for_noise(y, window: int):
    values = np.asarray(y, dtype=float)
    if values.size < 5:
        return values.copy()
    width = _odd_window(values.size, fraction=float(window) / max(values.size, 1), minimum=window)
    polyorder = max(1, min(3, width - 2))
    try:
        return scipy_signal.savgol_filter(values, width, polyorder)
    except Exception:
        sigma = max(width / 6.0, 1.0)
        return scipy_ndimage.gaussian_filter1d(values, sigma=sigma)


def estimate_sig_clean_stats(
    x,
    y,
    *,
    peak_sigma: float = DEFAULT_SIG_CLEAN_PEAK_SIGMA,
    smooth_fraction: float = DEFAULT_SIG_CLEAN_SMOOTH_FRACTION,
    width_rel_height: float = DEFAULT_SIG_CLEAN_WIDTH_REL_HEIGHT,
):
    signal = scipy_signal
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return {
            "noise_sigma": 0.0,
            "signal_level": 0.0,
            "noise_to_signal": 0.0,
            "snr": float("inf"),
            "peak_positions": np.array([]),
            "peak_left": np.array([]),
            "peak_right": np.array([]),
            "peak_count": 0,
        }
    window = _odd_window(y.size, fraction=smooth_fraction)
    residual = y - _smooth_for_noise(y, window)
    residual -= np.median(residual)
    noise_sigma = 1.4826 * float(np.median(np.abs(residual)))
    signal_level = float(np.percentile(y, 95) - np.percentile(y, 50))
    noise_to_signal = noise_sigma / (signal_level + 1e-12)
    snr = signal_level / (noise_sigma + 1e-12)
    prominence = max(noise_sigma * float(peak_sigma), float(np.percentile(y, 75)) * 0.05)
    peaks, _ = signal.find_peaks(y, prominence=prominence)
    if peaks.size:
        widths = signal.peak_widths(y, peaks, rel_height=float(width_rel_height))
        idx = np.arange(x.size)
        left = np.interp(widths[2], idx, x)
        right = np.interp(widths[3], idx, x)
        peak_positions = x[peaks]
    else:
        left = right = peak_positions = np.array([])
    return {
        "noise_sigma": float(noise_sigma),
        "signal_level": float(signal_level),
        "noise_to_signal": float(noise_to_signal),
        "snr": float(snr),
        "peak_positions": peak_positions,
        "peak_left": left,
        "peak_right": right,
        "peak_count": int(peaks.size),
    }


def adaptive_clean_signature(
    x,
    y,
    positions,
    weights,
    *,
    keep_top_k: int = DEFAULT_SIG_CLEAN_KEEP_TOP_K,
    min_components: int = DEFAULT_SIG_CLEAN_MIN_COMPONENTS,
    unsupported_strength_mult: float = DEFAULT_SIG_CLEAN_UNSUPPORTED_STRENGTH_MULT,
    strong_quantile_low_snr: float = DEFAULT_SIG_CLEAN_STRONG_QUANTILE_LOW_SNR,
    strong_quantile_high_snr: float = DEFAULT_SIG_CLEAN_STRONG_QUANTILE_HIGH_SNR,
):
    """Drop weak signature sticks that no detected peak supports.

    The keyword arguments are the ``signature_cleaning.mixture`` settings.
    """

    positions = np.asarray(positions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if positions.size == 0:
        return positions, weights
    stats = estimate_sig_clean_stats(
        x,
        y,
        peak_sigma=DEFAULT_SIG_CLEAN_PEAK_SIGMA,
        smooth_fraction=DEFAULT_SIG_CLEAN_SMOOTH_FRACTION,
        width_rel_height=DEFAULT_SIG_CLEAN_WIDTH_REL_HEIGHT,
    )
    min_thresh = DEFAULT_SIG_CLEAN_MIN_THRESH
    max_thresh = DEFAULT_SIG_CLEAN_MAX_THRESH
    snr_ref = max(DEFAULT_SIG_CLEAN_SNR_REF, 1e-12)
    snr_factor = 1.0 / (1.0 + max(stats["snr"], 0.0) / snr_ref)
    noise_factor = float(np.clip(stats["noise_to_signal"] * 4.0, 0.0, 1.0))
    threshold = min_thresh + (max_thresh - min_thresh) * max(snr_factor, noise_factor)
    grid_step = float(np.median(np.diff(np.sort(x)))) if len(x) > 1 else 0.0
    support = np.zeros(positions.size, dtype=bool)
    for left, right in zip(stats["peak_left"], stats["peak_right"]):
        margin = max(
            grid_step * DEFAULT_SIG_CLEAN_SUPPORT_GRID_MULT,
            (right - left) * DEFAULT_SIG_CLEAN_SUPPORT_WIDTH_MULT,
        )
        support |= (positions >= left - margin) & (positions <= right + margin)
    keep = support.copy()
    unsupported_floor = threshold * max(float(unsupported_strength_mult), 1.0)
    keep |= (~support) & (weights >= unsupported_floor)
    quantile = float(strong_quantile_low_snr)
    high_quantile = float(strong_quantile_high_snr)
    ratio = max(stats["snr"], 0.0) / (max(stats["snr"], 0.0) + snr_ref)
    quantile = float(
        np.clip(quantile - (quantile - high_quantile) * ratio, high_quantile, quantile)
    )
    keep |= weights >= max(threshold, float(np.quantile(weights, quantile)))
    top_k = int(keep_top_k)
    if top_k > 0:
        keep[np.argsort(weights)[-min(top_k, weights.size) :]] = True
    min_components = int(min_components)
    if keep.sum() < min(min_components, weights.size):
        keep[np.argsort(weights)[-min(min_components, weights.size) :]] = True
    cleaned = np.where(keep, weights, 0.0)
    if float(cleaned.sum()) <= 0:
        cleaned[np.argmax(weights)] = float(np.max(weights))
    return positions, cleaned / float(cleaned.sum())


_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def load_instprm(path: str | Path) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        values[key] = float(value) if _FLOAT_RE.fullmatch(value) else value
    return values


def caglioti_fwhm_proxy(two_theta, params: Mapping[str, Any]):
    theta = np.deg2rad(np.asarray(two_theta, dtype=float) / 2.0)
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


def estimate_peak_envelope(
    x, y, *, width_floor_abs: float = DEFAULT_WIDTH_FLOOR_ABS, rel_height: float = 0.5
):
    signal = scipy_signal
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 5:
        return np.array([]), np.array([])
    prominence = max(
        (float(np.percentile(y, 95)) - float(np.median(y))) * 0.08,
        float(np.percentile(y, 95)) * 0.015,
        1e-8,
    )
    peaks, _ = signal.find_peaks(y, prominence=prominence)
    if not peaks.size:
        return np.array([]), np.array([])
    widths = signal.peak_widths(y, peaks, rel_height=float(rel_height))
    step = float(np.median(np.diff(x))) if x.size > 1 else width_floor_abs
    return x[peaks], np.maximum(widths[0] * step, float(width_floor_abs))


def signature_widths(
    x,
    y,
    positions,
    *,
    instrument: str | Path | None = None,
    width_floor_abs: float = DEFAULT_WIDTH_FLOOR_ABS,
    rel_height: float = 0.5,
):
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.array([]), {"source": "none"}
    peak_positions, peak_widths = estimate_peak_envelope(
        x, y, width_floor_abs=width_floor_abs, rel_height=rel_height
    )
    if instrument is None:
        if not peak_positions.size:
            return np.full(positions.size, width_floor_abs), {
                "source": "pattern",
                "width_floor_abs": float(width_floor_abs),
            }
        idx = np.abs(positions[:, None] - peak_positions[None, :]).argmin(axis=1)
        return np.maximum(peak_widths[idx], width_floor_abs), {
            "source": "pattern",
            "width_floor_abs": float(width_floor_abs),
            "rel_height": float(rel_height),
        }
    params = load_instprm(instrument)
    proxy = caglioti_fwhm_proxy(positions, params)
    if peak_positions.size:
        denom = float(np.median(caglioti_fwhm_proxy(peak_positions, params)))
        scale = float(np.median(peak_widths) / denom) if denom > 0 else 1.0
    else:
        denom = float(np.median(proxy)) if proxy.size else 0.0
        scale = float(width_floor_abs / denom) if denom > 0 else 1.0
    return np.maximum(scale * proxy, width_floor_abs), {
        "source": "prm",
        "prm_path": str(instrument),
        "width_floor_abs": float(width_floor_abs),
        "rel_height": float(rel_height),
        "prm_scale": float(scale),
    }


def preprocess_pattern(
    record: Mapping[str, Any],
    *,
    theta_min: float | None = None,
    theta_max: float | None = None,
    gridpoints: int | None = None,
    background: Any = None,
    center: bool = False,
    epsilon: float = 1e-19,
    smooth: bool = False,
    window: int = 18,
    denoise: Any = DEFAULT_DENOISE_METHOD,
    denoise_window: int = DEFAULT_DENOISE_WINDOW,
    denoise_polyorder: int = DEFAULT_DENOISE_POLYORDER,
    signature_points: int | None = None,
    signature_reduce_mode: str = DEFAULT_SIG_REDUCE_MODE,
    signature_cleaning: str | Mapping[str, Any] | None = DEFAULT_SIG_CLEAN_MODE,
    instrument: str | Path | None = None,
    width_floor_abs: float = DEFAULT_WIDTH_FLOOR_ABS,
    width_rel_height: float = 0.5,
) -> dict[str, Any]:
    """Preprocess one loader record and return a fresh serialisable mapping.

    Parameter names follow ``unmix.preprocessing`` in the YAML schema.
    ``signature_cleaning`` is either a mode name, as for the basis branch, or
    a mapping with ``mode`` and the adaptive cleaning settings, as for the
    mixture branch.
    """

    raw_x = np.asarray(record["x"], dtype=float).reshape(-1).copy()
    raw_y = np.asarray(record["y"], dtype=float).reshape(-1).copy()
    if raw_x.size != raw_y.size:
        raise ValueError(
            f"Sample {record.get('sample_id', '')!r} has x/y lengths {raw_x.size}/{raw_y.size}"
        )
    if raw_x.size < 2:
        raise ValueError(f"Sample {record.get('sample_id', '')!r} must contain at least two points")
    finite = np.isfinite(raw_x) & np.isfinite(raw_y)
    if not finite.all():
        bad = np.flatnonzero(~finite) + 1
        shown = ", ".join(str(int(row)) for row in bad[:8])
        suffix = " ..." if bad.size > 8 else ""
        raise ValueError(
            f"Sample {record.get('sample_id', '')!r} contains non-finite x/y values "
            f"on row(s) {shown}{suffix}"
        )
    if gridpoints is not None:
        if isinstance(gridpoints, bool):
            raise ValueError("gridpoints must be zero (native grid) or an integer at least two")
        try:
            numeric_gridpoints = float(gridpoints)
            integer_gridpoints = int(numeric_gridpoints)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "gridpoints must be zero (native grid) or an integer at least two"
            ) from exc
        if not math.isfinite(numeric_gridpoints) or numeric_gridpoints != integer_gridpoints:
            raise ValueError("gridpoints must be zero (native grid) or an integer at least two")
        gridpoints = integer_gridpoints
        if gridpoints < 0 or gridpoints == 1:
            raise ValueError("gridpoints must be zero (native grid) or an integer at least two")
    x = raw_x.copy()
    y = raw_y.copy()
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    left = float(np.min(x)) if theta_min is None else float(theta_min)
    right = float(np.max(x)) if theta_max is None else float(theta_max)
    if not math.isfinite(left) or not math.isfinite(right) or right <= left:
        raise ValueError(
            f"Sample {record.get('sample_id', '')!r} needs a finite theta range with max > min"
        )
    raw_min, raw_max = float(x[0]), float(x[-1])
    native_grid = gridpoints is None or gridpoints == 0
    if right <= raw_min or left >= raw_max:
        raise ValueError(
            f"Sample {record.get('sample_id', '')!r} requested theta range "
            f"[{left:g}, {right:g}] with no overlap with measured range "
            f"[{raw_min:g}, {raw_max:g}]"
        )
    coverage_issues: list[str] = []
    if theta_min is not None and left < raw_min - THETA_COVERAGE_TOLERANCE:
        coverage_issues.append(f"theta_min={left:g} < measured minimum {raw_min:g}")
    if theta_max is not None and right > raw_max + THETA_COVERAGE_TOLERANCE:
        coverage_issues.append(f"theta_max={right:g} > measured maximum {raw_max:g}")
    if coverage_issues and not native_grid:
        raise ValueError(
            f"Sample {record.get('sample_id', '')!r} requested theta range "
            f"[{left:g}, {right:g}] is outside measured range "
            f"[{raw_min:g}, {raw_max:g}] for interpolated grid: " + "; ".join(coverage_issues)
        )
    # The source ``default_preprocess_func`` drops non-positive samples
    # when operating on the native grid, then keeps theta_min
    # and excludes theta_max.  For explicit interpolation, it retains all finite
    # samples as the old ``interp1d`` branch did.
    positive = np.ones(x.size, dtype=bool)
    if gridpoints is None or gridpoints < 2:
        positive = y > 0.0
    mask = (x >= left) & (x < right) & positive
    if mask.sum() < 2:
        raise ValueError(
            f"Sample {record.get('sample_id', '')!r} has no points inside [{left}, {right}]"
        )
    x, y = x[mask], y[mask]
    if gridpoints is not None and gridpoints >= 2:
        grid = np.linspace(left, right, gridpoints)
        y = np.interp(grid, x, y)
        x = grid
    fitter = background_function(background)
    baseline = None
    if fitter is not None:
        baseline = np.asarray(fitter(x, y), dtype=float)
        baseline = np.minimum(baseline, y)
        y = y - baseline
        # Historical ``default_preprocess_func`` replaces exact zeros before
        # denoising and normalisation.  This matters for sparse simulated
        # patterns where a broad baseline can touch the signal floor.
        y[y == 0] = float(epsilon)
    y = denoise_signal(y, denoise, window=denoise_window, polyorder=denoise_polyorder)
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    if smooth:
        y = smooth_y(y, window=window)
    total = float(y.sum())
    if not math.isfinite(total) or total <= 0:
        y = np.full(x.size, 1.0 / x.size, dtype=float)
    else:
        y /= total
    offset = None
    if center:
        offset = float(np.sum(x * y))
        x = x - offset
    # A normalised dense curve is itself a valid signature.
    target = 256 if signature_points is None else signature_points
    pos, weights = reduce_signature(x, y, points=int(target), mode=signature_reduce_mode)
    if isinstance(signature_cleaning, Mapping):
        cleaning_settings = dict(signature_cleaning)
        cleaning_mode = cleaning_settings.pop("mode", DEFAULT_SIG_CLEAN_MODE)
    else:
        cleaning_settings, cleaning_mode = {}, signature_cleaning
    clean_info: dict[str, Any] = {}
    if cleaning_mode not in (None, "none"):
        if str(cleaning_mode) != DEFAULT_SIG_CLEAN_MODE:
            raise ValueError(f"Unsupported signature_cleaning mode: {cleaning_mode!r}")
        pos, weights = adaptive_clean_signature(x, y, pos, weights, **cleaning_settings)
        clean_info = {"mode": DEFAULT_SIG_CLEAN_MODE, "kept_components": int(weights.size)}
    widths, width_info = signature_widths(
        x,
        y,
        pos,
        instrument=instrument,
        width_floor_abs=width_floor_abs,
        rel_height=width_rel_height,
    )
    output = dict(record)
    # ``x``/``y`` are the public raw pattern fields used by callers that load
    # a prepared record.  Keep the transformed dense curve in the explicit
    # ``clean_x``/``clean_y`` fields so cropping, centering, background removal
    # and normalisation cannot silently change the raw payload's meaning.
    output.update(
        {
            "x": raw_x,
            "y": raw_y,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "clean_x": x,
            "clean_y": y,
            "baseline": baseline,
            "signature_positions": pos,
            "signature_weights": weights,
            "signature_widths": widths,
            "signature_width_info": width_info,
            "sig_clean_info": clean_info,
            "offset": offset,
            "x_min": float(x.min()),
            "x_max": float(x.max()),
        }
    )
    return output


__all__ = [
    "DEFAULT_SIG_CLEAN_MODE",
    "background_function",
    "fit_background_psalsa",
    "reduce_signature",
    "adaptive_clean_signature",
    "signature_widths",
    "preprocess_pattern",
]

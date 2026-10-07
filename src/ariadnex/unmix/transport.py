"""One-dimensional transport primitives used by the MILP and CPU fallback."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import nnls


def _vector(value, *, name: str, allow_empty: bool = True) -> np.ndarray:
    """Return a finite flattened float vector.

    Solver inputs are scientific data, so replacing a malformed value with a
    clipped or empty array would change the experiment without telling the
    caller.  Keep this check at the numerical seam where every solver path
    passes through it.
    """

    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if not allow_empty and result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _signature(positions, weights, *, name: str, allow_empty: bool = False):
    positions = _vector(positions, name=f"{name} positions", allow_empty=allow_empty)
    weights = _vector(weights, name=f"{name} weights", allow_empty=allow_empty)
    if positions.size != weights.size:
        raise ValueError(
            f"{name} positions and weights must have matching lengths "
            f"({positions.size} and {weights.size})"
        )
    if np.any(weights < 0.0):
        raise ValueError(f"{name} weights must be non-negative")
    if positions.size and float(weights.sum()) <= 0.0:
        raise ValueError(f"{name} weights must have positive total mass")
    return positions, weights


def pairwise_cost(source_positions, target_positions, p: float = 2.0):
    if not np.isfinite(float(p)) or float(p) <= 0.0:
        raise ValueError(f"transport exponent p must be finite and positive, got {p!r}")
    source = _vector(source_positions, name="source positions")[:, None]
    target = _vector(target_positions, name="target positions")[None, :]
    return np.abs(source - target) ** float(p)


def monotone_transport(source_positions, source_weights, target_positions, target_weights):
    """Return the exact monotone 1-D transport matrix for two signatures."""

    sp, sw = _signature(source_positions, source_weights, name="source signature")
    tp, tw = _signature(target_positions, target_weights, name="target signature")
    if not sp.size or not tp.size:
        return np.zeros((sp.size, tp.size), dtype=float)
    so, to = np.argsort(sp, kind="mergesort"), np.argsort(tp, kind="mergesort")
    sp, sw, tp, tw = sp[so], sw[so], tp[to], tw[to]
    source_total, target_total = float(sw.sum()), float(tw.sum())
    if source_total <= 0 or target_total <= 0:
        return np.zeros((sp.size, tp.size), dtype=float)
    sw = sw / source_total
    tw = tw / target_total
    flow_sorted = np.zeros((sp.size, tp.size), dtype=float)
    i = j = 0
    left_s, left_t = float(sw[0]), float(tw[0])
    while i < sp.size and j < tp.size:
        amount = min(left_s, left_t)
        flow_sorted[i, j] += amount
        left_s -= amount
        left_t -= amount
        if left_s <= 1e-15:
            i += 1
            if i < sp.size:
                left_s = float(sw[i])
        if left_t <= 1e-15:
            j += 1
            if j < tp.size:
                left_t = float(tw[j])
    flow = np.zeros_like(flow_sorted)
    flow[np.ix_(so, to)] = flow_sorted
    return flow


def transport_cost(
    flow, source_positions, target_positions, p: float = 2.0, *, cost_matrix=None
) -> float:
    matrix = (
        pairwise_cost(source_positions, target_positions, p=p)
        if cost_matrix is None
        else np.asarray(cost_matrix, dtype=float)
    )
    return float(np.sum(np.asarray(flow, dtype=float) * matrix))


def broaden_support(mask, radius: int = 1):
    result = np.asarray(mask, dtype=bool).copy()
    radius = max(int(radius), 0)
    if radius == 0:
        return result
    for _ in range(radius):
        expanded = result.copy()
        expanded[1:, :] |= result[:-1, :]
        expanded[:-1, :] |= result[1:, :]
        expanded[:, 1:] |= result[:, :-1]
        expanded[:, :-1] |= result[:, 1:]
        result = expanded
    return result


def monotone_support_mask(
    source_positions, source_weights, target_positions, target_weights, *, radius: int = 1
):
    return broaden_support(
        monotone_transport(source_positions, source_weights, target_positions, target_weights) > 0,
        radius=radius,
    )


def signature_on_grid(positions, weights, grid):
    """Project a sparse signature onto a target grid by linear mass splitting."""

    p, w = _signature(positions, weights, name="basis signature")
    g = _vector(grid, name="target grid", allow_empty=False)
    output = np.zeros(g.size, dtype=float)
    if not p.size:
        return output
    total = float(w.sum())
    w = w / total
    order = np.argsort(g, kind="mergesort")
    sorted_grid = g[order]
    for position, weight in zip(p, w):
        if position <= sorted_grid[0]:
            output[order[0]] += weight
        elif position >= sorted_grid[-1]:
            output[order[-1]] += weight
        else:
            right = int(np.searchsorted(sorted_grid, position, side="left"))
            left = right - 1
            span = sorted_grid[right] - sorted_grid[left]
            fraction = 0.0 if span <= 0 else (position - sorted_grid[left]) / span
            output[order[left]] += weight * (1.0 - fraction)
            output[order[right]] += weight * fraction
    return output / max(float(output.sum()), 1e-30)


def cpu_anomaly_solution(
    basis_signatures: Sequence[tuple[Any, Any]],
    target_signature: tuple[Any, Any],
    *,
    max_base: int | None = None,
    anomaly_penalty: float | Sequence[float] = 1.0,
    p: float = 2.0,
    selection_threshold: float = 1e-9,
):
    """Deterministic non-MILP solve used when Gurobi is unavailable.

    Basis signatures are projected onto the target support and solved with
    NNLS.  A cardinality cap keeps the same sparse intent as the MILP; the
    remaining mass is represented explicitly by the anomaly channel.
    """

    if not np.isfinite(float(p)) or float(p) <= 0.0:
        raise ValueError(f"transport exponent p must be finite and positive, got {p!r}")
    if max_base is not None and int(max_base) < 0:
        raise ValueError(f"max_base must be non-negative, got {max_base!r}")
    if not np.isfinite(float(selection_threshold)) or float(selection_threshold) < 0.0:
        raise ValueError(
            f"selection_threshold must be finite and non-negative, got {selection_threshold!r}"
        )

    target_positions, target_weights = _signature(
        target_signature[0], target_signature[1], name="target signature"
    )
    target_total = float(target_weights.sum())
    if target_total <= 0:
        empty_components = [np.zeros(target_positions.size, dtype=float) for _ in basis_signatures]
        return (
            empty_components,
            np.zeros(target_positions.size, dtype=float),
            {"solver": "cpu", "selected_indices": [], "objective": 0.0, "max_base": max_base},
        )
    target_weights = target_weights / target_total
    validated_basis = [
        _signature(item[0], item[1], name=f"basis signature {index}")
        for index, item in enumerate(basis_signatures)
    ]
    vectors = (
        np.vstack(
            [signature_on_grid(pos, weights, target_positions) for pos, weights in validated_basis]
        )
        if basis_signatures
        else np.zeros((0, target_positions.size))
    )
    if vectors.size == 0:
        return (
            [],
            np.asarray(target_weights, dtype=float).copy(),
            {"solver": "cpu", "selected_indices": [], "objective": float(np.sum(target_weights))},
        )
    if np.ndim(anomaly_penalty) != 0:
        penalty = _vector(anomaly_penalty, name="anomaly_penalty", allow_empty=False)
        if penalty.shape != target_weights.shape:
            raise ValueError("anomaly_penalty vector must have the same length as target weights")
        if np.any(penalty < 0.0):
            raise ValueError("anomaly_penalty values must be non-negative")
    elif not np.isfinite(float(anomaly_penalty)) or float(anomaly_penalty) < 0.0:
        raise ValueError("anomaly_penalty must be finite and non-negative")
    coefficients, _ = nnls(vectors.T, target_weights)
    coefficients = np.maximum(np.asarray(coefficients, dtype=float), 0.0)
    if (
        max_base is not None
        and int(max_base) >= 0
        and int(np.count_nonzero(coefficients > selection_threshold)) > int(max_base)
    ):
        keep = np.argsort(-coefficients, kind="mergesort")[: int(max_base)]
        mask = np.zeros(coefficients.size, dtype=bool)
        mask[keep] = True
        coefficients[~mask] = 0.0
        if mask.any():
            coefficients[mask], _ = nnls(vectors[mask].T, target_weights)
            expanded = np.zeros_like(coefficients)
            expanded[mask] = np.maximum(coefficients[mask], 0.0)
            coefficients = expanded
    reconstruction = coefficients @ vectors
    # Enforce the same mass-conservation invariant as the MILP: no target bin
    # may receive more basis mass than observed.  Ignore exactly-zero target
    # bins when choosing a global scale (real PXRD signatures are sparse, so a
    # basis can legitimately have interpolation tails there), then apply a
    # per-bin projection below for any remaining overshoot.
    positive = reconstruction > 1e-15
    supported = positive & (target_weights > 1e-15)
    if np.any(supported):
        ratio = float(np.min(target_weights[supported] / reconstruction[supported]))
        if ratio < 1.0:
            coefficients *= max(ratio, 0.0)
            reconstruction = coefficients @ vectors
    coeff_total = float(coefficients.sum())
    if coeff_total > 1.0:
        coefficients /= coeff_total
        reconstruction = coefficients @ vectors
    reconstruction = np.maximum(reconstruction, 0.0)
    # Project every component on the observed mass envelope.  This keeps the
    # returned component sums and anomaly profile exactly consistent even when
    # sparse target bins are zero.
    bin_scale = np.ones(target_weights.size, dtype=float)
    positive = reconstruction > 1e-15
    bin_scale[positive] = np.minimum(1.0, target_weights[positive] / reconstruction[positive])
    if np.any(bin_scale < 1.0):
        vectors = vectors * bin_scale[None, :]
        reconstruction = coefficients @ vectors
    anomaly = np.maximum(target_weights - reconstruction, 0.0)
    components = [float(coefficients[i]) * vectors[i] for i in range(vectors.shape[0])]
    objective = float(
        np.sum(anomaly)
        * (
            float(anomaly_penalty)
            if np.ndim(anomaly_penalty) == 0
            else float(np.mean(anomaly_penalty))
        )
    )
    objective += float(np.sum(np.maximum(reconstruction - target_weights, 0.0)))
    return (
        components,
        anomaly,
        {
            "solver": "cpu",
            "selected_indices": [
                int(i) for i, value in enumerate(coefficients) if value > selection_threshold
            ],
            "coefficients": coefficients,
            "objective": objective,
            "max_base": max_base,
        },
    )


__all__ = [
    "pairwise_cost",
    "monotone_transport",
    "transport_cost",
    "broaden_support",
    "monotone_support_mask",
    "signature_on_grid",
    "cpu_anomaly_solution",
]

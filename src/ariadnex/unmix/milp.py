"""Gurobi transport MILP for anomaly-aware PXRD unmixing.

Only this module contains the Gurobi formulation.
"""

from __future__ import annotations

from typing import Any, Sequence

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.signal import convolve2d

from .transport import _signature, monotone_transport, pairwise_cost


def _gen_spherical_kernel(n=3, cutoff=1.2):
    axis = np.linspace(-1.0, 1.0, int(n))
    xx, yy = np.meshgrid(axis, axis)
    return xx**2 + yy**2 < float(cutoff)


def _fill_diag(max_support):
    rows, columns = max_support.shape
    mask = np.zeros((rows, columns), dtype=float)
    for column in range(columns):
        row_target = int((rows / columns) * column)
        row_peak = int(np.asarray(max_support[:, column]).argmax())
        for row in range(min(row_peak, row_target), max(row_peak, row_target) + 1):
            mask[row, column] = 1.0
    return mask


def _support_mask(
    source_positions,
    source_weights,
    target_positions,
    target_weights,
    cost_matrix,
    *,
    kernel_size,
):
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    if cost_matrix.ndim != 2 or not np.isfinite(cost_matrix).all():
        raise ValueError("Transport cost matrix must be a finite two-dimensional array")
    # This is the mask construction in ``segmentation.lp_segment``:
    # monotone support, diagonal filling, then spherical convolution.
    source_weights = np.asarray(source_weights, dtype=float).reshape(-1)
    target_weights = np.asarray(target_weights, dtype=float).reshape(-1)
    monotone = monotone_transport(
        np.arange(source_weights.size, dtype=float),
        source_weights,
        np.arange(target_weights.size, dtype=float),
        target_weights,
    )
    mask = _fill_diag(monotone > 0)
    kernel = _gen_spherical_kernel(max(3, int(kernel_size)))
    mask = (convolve2d(mask, kernel, "same") > 0).astype(float)
    if mask.shape != cost_matrix.shape:
        raise ValueError(
            f"Transport support mask shape {mask.shape} does not match cost matrix {cost_matrix.shape}"
        )
    if not np.isfinite(mask).all():
        raise ValueError("Transport support mask must contain only finite values")
    return np.where(mask > 0, 1.0, 0.0)


def _checked_matrices(
    matrices: Sequence[Any], expected_shapes: Sequence[tuple[int, int]], name: str
) -> list[np.ndarray]:
    checked = []
    for index, (matrix, expected_shape) in enumerate(zip(matrices, expected_shapes)):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(
                f"{name}[{index}] must be finite with shape {expected_shape}, got {matrix.shape}"
            )
        if np.any(matrix < 0.0):
            raise ValueError(f"{name}[{index}] must be non-negative")
        checked.append(matrix)
    return checked


def solve_anomaly_milp(
    basis_signatures: Sequence[tuple[Any, Any]],
    target_signature: tuple[Any, Any],
    *,
    cost_mats: Sequence[Any] | None = None,
    disp_mats: Sequence[Any] | None = None,
    max_emd_rows: Sequence[Any] | None,
    anomaly_penalty: float | Sequence[float],
    missing_element_penalty: float,
    soft_sparse_penalty: float,
    present_elements: Sequence[str],
    basis_element_sets: Sequence[set[str]],
    max_base: int | None,
    allow_anomaly: bool,
    resolution: float,
    epsilon: float,
    anomaly_relax: float,
    p: float,
    gurobi_threads: int | None,
    width_budget_mode: str,
    strict_solver: bool,
    kernel_size: int | None = None,
):
    """Solve the anomaly-aware transport problem and return flow matrices.

    Returns ``(flows, cost_matrices, anomaly, diagnostics)``.  ``flows[i]``
    has shape ``(n_basis_sticks_i, n_target_sticks)`` and ``anomaly`` has one
    value per target stick.  The diagnostics mapping contains solver status,
    selected basis indices, and support masks.  ``anomaly_penalty`` and the
    element and sparsity penalties are absolute; the solver scales the
    relative ``beta`` settings before calling.  ``max_emd_rows`` holds one
    displacement budget per basis stick, or ``None`` for no budget.
    """

    normalized_width_budget_mode = str(width_budget_mode).strip().lower()
    if normalized_width_budget_mode != "global_avg":
        raise ValueError(
            f"Unsupported width_budget_mode {width_budget_mode!r}; only 'global_avg' is supported"
        )

    try:
        basis_signatures = list(basis_signatures)
    except TypeError as exc:
        raise ValueError("basis_signatures must be a sequence of position/weight pairs") from exc
    target_positions, target_weights = _signature(
        target_signature[0], target_signature[1], name="target signature"
    )
    target_total = float(target_weights.sum())
    if target_total <= 0:
        raise ValueError("Target signature must have positive total mass")
    # _signature returns a view of the caller's array; do not normalize it in place.
    target_weights = target_weights / target_total
    if not basis_signatures:
        return (
            [],
            [],
            target_weights.copy(),
            {"solver": "gurobi", "status": "empty", "selected_indices": []},
        )

    if cost_mats is None:
        cost_mats = [
            pairwise_cost(positions, target_positions, p=p) for positions, _ in basis_signatures
        ]
    if disp_mats is None:
        disp_mats = [
            np.abs(np.asarray(positions, dtype=float)[:, None] - target_positions[None, :])
            for positions, _ in basis_signatures
        ]
    if len(cost_mats) != len(basis_signatures) or len(disp_mats) != len(basis_signatures):
        raise ValueError("One cost and displacement matrix is required per basis signature")

    validated_basis = [
        _signature(positions, weights, name=f"basis signature {index}")
        for index, (positions, weights) in enumerate(basis_signatures)
    ]
    expected_shapes = [(positions.size, target_positions.size) for positions, _ in validated_basis]
    cost_mats = _checked_matrices(cost_mats, expected_shapes, "cost_mats")
    disp_mats = _checked_matrices(disp_mats, expected_shapes, "disp_mats")

    if not np.isfinite(float(p)) or float(p) <= 0.0:
        raise ValueError(f"transport exponent p must be finite and positive, got {p!r}")
    for name, value in (
        ("resolution", resolution),
        ("epsilon", epsilon),
        ("anomaly_relax", anomaly_relax),
        ("missing_element_penalty", missing_element_penalty),
        ("soft_sparse_penalty", soft_sparse_penalty),
    ):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    if max_emd_rows is not None:
        if len(max_emd_rows) != len(validated_basis):
            raise ValueError("max_emd_rows must contain one row budget per basis signature")
        for index, (budgets, (positions, _)) in enumerate(zip(max_emd_rows, validated_basis)):
            values = np.asarray(budgets, dtype=float).reshape(-1)
            if values.size != positions.size or not np.isfinite(values).all():
                raise ValueError(
                    f"max_emd_rows[{index}] must be finite with {positions.size} values"
                )
            if np.any(values < 0.0):
                raise ValueError(f"max_emd_rows[{index}] must be non-negative")

    mask_kernel_size = max(3, int(kernel_size or max(3, target_positions.size // 5)))
    support_masks = [
        _support_mask(
            positions,
            weights,
            target_positions,
            target_weights,
            cost_mats[index],
            kernel_size=mask_kernel_size,
        )
        for index, (positions, weights) in enumerate(validated_basis)
    ]
    element_sets = [
        set(elements) for elements in (basis_element_sets or [set() for _ in basis_signatures])
    ]
    present = [str(element) for element in (present_elements or [])]
    if len(element_sets) != len(basis_signatures):
        raise ValueError("basis_element_sets must have one set per basis signature")

    model = gp.Model("ariadnex_unmix")
    model.Params.TimeLimit = 1e9
    model.Params.MIPGap = 0.001
    model.Params.OutputFlag = 0
    if gurobi_threads is not None:
        if int(gurobi_threads) < 1:
            raise ValueError("gurobi_threads must be >= 1")
        model.Params.Threads = int(gurobi_threads)

    flow_vars = []
    indicators = []
    objective_terms = []
    marginal = None
    total_flow_ub = float(np.sum(target_weights)) + target_weights.size * float(anomaly_relax)
    for index, ((source_positions, source_weights), matrix, disp, mask) in enumerate(
        zip(basis_signatures, cost_mats, disp_mats, support_masks)
    ):
        source_weights = np.asarray(source_weights, dtype=float).reshape(-1)
        source_total = float(source_weights.sum())
        if source_total <= 0:
            source_weights = np.full(source_weights.size, 1.0 / max(source_weights.size, 1))
        else:
            source_weights = np.maximum(source_weights, 0.0) / source_total
        ub = np.where(mask > 0, GRB.INFINITY, 0.0)
        flow = model.addMVar(np.asarray(matrix).shape, lb=0.0, ub=ub, name=f"flow_{index}")
        indicator = model.addVar(vtype=GRB.BINARY, name=f"use_basis_{index}")
        flow_sum = model.addVar(lb=0.0, ub=total_flow_ub, name=f"flow_sum_{index}")
        model.addConstr(flow_sum == flow.sum(), name=f"flow_sum_def_{index}")
        model.addConstr(
            flow.sum(axis=1) - flow_sum * source_weights <= float(epsilon),
            name=f"source_marginal_upper_{index}",
        )
        model.addConstr(
            flow.sum(axis=1) - flow_sum * source_weights >= 0.0,
            name=f"source_marginal_lower_{index}",
        )
        model.addConstr(flow_sum >= float(resolution) * indicator, name=f"basis_lb_{index}")
        model.addGenConstrIndicator(
            indicator, 0, flow_sum, GRB.EQUAL, 0.0, name=f"basis_off_{index}"
        )
        if max_emd_rows is not None:
            row_budget = np.asarray(max_emd_rows[index], dtype=float).reshape(-1, 1)
            model.addConstr(
                ((np.asarray(disp, dtype=float) - row_budget) * flow).sum() <= 0.0,
                name=f"row_budget_{index}",
            )
        flow_vars.append(flow)
        indicators.append(indicator)
        marginal = flow.sum(axis=0) if marginal is None else marginal + flow.sum(axis=0)
        objective_terms.append((np.asarray(matrix, dtype=float) * flow).sum())

    anomaly_ub = GRB.INFINITY if allow_anomaly else 0.0
    anomaly = model.addMVar(target_weights.shape, lb=0.0, ub=anomaly_ub, name="anomaly")
    penalty = np.asarray(anomaly_penalty, dtype=float)
    if not np.isfinite(penalty).all() or np.any(penalty < 0.0):
        raise ValueError("anomaly_penalty must contain finite non-negative values")
    if penalty.ndim == 0:
        objective_terms.append(float(penalty) * anomaly.sum())
    elif penalty.shape == target_weights.shape:
        objective_terms.append((penalty * anomaly).sum())
    else:
        raise ValueError(
            f"anomaly_penalty vector shape {penalty.shape} does not match target weights "
            f"{target_weights.shape}"
        )
    model.addConstr(
        marginal + anomaly - target_weights <= float(anomaly_relax), name="target_upper"
    )
    model.addConstr(
        marginal + anomaly - target_weights >= -float(anomaly_relax), name="target_lower"
    )

    # Element priors are soft: a candidate containing a required element can
    # be selected, but omitting all candidates for that element is penalised.
    for element in present:
        indices = [index for index, elements in enumerate(element_sets) if element in elements]
        if not indices:
            continue
        missing = model.addVar(lb=0.0, ub=1.0, name=f"missing_{element}")
        model.addConstr(
            missing >= 1.0 - gp.quicksum(indicators[index] for index in indices),
            name=f"missing_def_{element}",
        )
        objective_terms.append(float(missing_element_penalty) * missing)
    if float(soft_sparse_penalty):
        objective_terms.append(float(soft_sparse_penalty) * gp.quicksum(indicators))
    max_base_value = len(basis_signatures) if max_base is None else int(max_base)
    if max_base_value < 0:
        raise ValueError("max_base must be non-negative")
    model.addConstr(gp.quicksum(indicators) <= max_base_value, name="max_base")
    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
    model.optimize()
    status = int(model.Status)
    if model.SolCount <= 0:
        raise RuntimeError(f"Gurobi returned status {status} without a feasible solution")
    status_name = {
        int(GRB.OPTIMAL): "OPTIMAL",
        int(GRB.SUBOPTIMAL): "SUBOPTIMAL",
        int(GRB.TIME_LIMIT): "TIME_LIMIT",
        int(GRB.INTERRUPTED): "INTERRUPTED",
        int(GRB.NODE_LIMIT): "NODE_LIMIT",
        int(GRB.SOLUTION_LIMIT): "SOLUTION_LIMIT",
        int(GRB.MEM_LIMIT): "MEM_LIMIT",
        int(GRB.NUMERIC): "NUMERIC",
        int(GRB.USER_OBJ_LIMIT): "USER_OBJ_LIMIT",
    }.get(status, f"STATUS_{status}")
    if strict_solver and status != int(GRB.OPTIMAL):
        raise RuntimeError(
            f"Gurobi returned {status_name} with a feasible solution; "
            "strict_solver requires an optimal solution"
        )
    flows = [np.asarray(flow.X, dtype=float) * mask for flow, mask in zip(flow_vars, support_masks)]
    anomaly_out = np.asarray(anomaly.X, dtype=float)
    if not np.isfinite(anomaly_out).all() or any(not np.isfinite(flow).all() for flow in flows):
        raise RuntimeError("Gurobi returned non-finite flow or anomaly values")
    selected = [index for index, indicator in enumerate(indicators) if float(indicator.X) > 0.5]
    objective = float(model.ObjVal)
    transport_cost = float(
        sum(
            np.sum(np.asarray(matrix, dtype=float) * flow.X * mask)
            for matrix, flow, mask in zip(cost_mats, flow_vars, support_masks)
        )
    )
    diagnostics = {
        "solver": "gurobi",
        "status": status,
        "status_name": status_name,
        "selected_indices": selected,
        "objective": objective,
        # SOTU's ``segment_anom_milp`` returns W (transport cost) separately
        # from the full MILP objective, which also contains anomaly and prior
        # penalties.  Keep both values for report and parity consumers.
        "transport_cost": transport_cost,
        "basis_masses": [float(flow.sum()) for flow in flows],
        "anomaly_mass": float(anomaly_out.sum()),
        "support_masks": support_masks,
    }
    return flows, list(cost_mats), anomaly_out, diagnostics


__all__ = ["solve_anomaly_milp"]

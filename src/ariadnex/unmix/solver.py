"""Configuration resolution and numerical sample solving.

The mathematical implementations stay together here so moving orchestration
code does not change their construction order, floating-point operations, or
worker entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from ariadnex.config import ProjectConfig

from .element_prior import (
    build_element_prior,
    check_element_coverage,
    filter_basis_indices,
    format_removed_phases,
    normalize_sample_name,
    resolve_sample_max_base,
)
from .milp import solve_anomaly_milp
from .transport import _signature, cpu_anomaly_solution, pairwise_cost
from .width_prior import build_local_disp_budgets

# Gurobi's feasibility tolerance can leave a tiny negative flow after solving
# constraints whose lower bound is zero.  Accepted paper runs contain values
# down to about -1.3e-6; retain those values for numerical parity while still
# rejecting a materially negative solver output.
_SOLVER_NEGATIVE_TOL = 1e-5


def _resolve_config_path(config: ProjectConfig, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config.base_dir / path
    return path.resolve()


# The only defaults for unmix settings.  Public names follow ``unmix.solver``;
# ``p``, ``epsilon``, ``anomaly_relax``, ``selection_threshold``, and ``solver``
# are numerical settings that the YAML schema does not expose.
_SOLVER_DEFAULTS: dict[str, Any] = {
    "allow_anomaly": True,
    "formulation": "milp",
    "solver": "auto",
    "max_base": 3,
    "max_base_extra": 0,
    "resolution": 1e-5,
    "beta": 2.5,
    "missing_element_penalty": 0.0,
    "soft_sparse_penalty": 0.0,
    "width_prior_mode": "none",
    "width_budget_mode": "global_avg",
    "delta_floor": 0.0,
    "delta_tau": None,
    "width_floor_abs": 0.045,
    "strict_solver": False,
    "threads": 1,
    "gurobi_threads": None,
    "phase_order_manifest": None,
    "p": 2,
    "epsilon": 1e-31,
    "anomaly_relax": 1e-5,
    "selection_threshold": 1e-8,
}


def _unmix_options(config: ProjectConfig) -> dict[str, Any]:
    """Return the unmix settings under their YAML names.

    ``unmix.solver`` keys sit at the top level with ``profile``,
    ``allow_anomaly``, ``instrument`` (``inputs.instrument``), and
    ``signature_points``.  ``element_prior`` holds that section, and
    ``basis_preprocessing`` holds the ``preprocess_pattern`` settings for basis
    patterns, or ``None`` when ``preprocessing.background.basis`` is unset.
    """

    unmix = config.section("unmix")
    preprocessing = config.section("unmix.preprocessing")
    options: dict[str, Any] = {**_SOLVER_DEFAULTS, **config.section("unmix.solver")}
    if "profile" in unmix:
        options["profile"] = unmix["profile"]
    options["allow_anomaly"] = bool(unmix.get("allow_anomaly", options["allow_anomaly"]))
    options["instrument"] = config.get("inputs.instrument")
    options["signature_points"] = preprocessing.get("signature_points")
    prior = unmix.get("element_prior")
    prior = prior if isinstance(prior, Mapping) else {}
    options["element_prior"] = {
        "enabled": bool(prior.get("enabled", False)),
        "considered_elements": prior.get("considered_elements"),
        "sample_description_path": prior.get("sample_description_path"),
    }
    options["basis_preprocessing"] = _basis_preprocessing(preprocessing, options)
    return options


def _basis_preprocessing(
    preprocessing: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return the basis-branch ``preprocess_pattern`` settings.

    Unlike the mixture branch, basis patterns keep the function defaults for
    ``center`` and ``smooth`` and use the solver's ``width_floor_abs``.
    """

    def branch(section: str) -> Mapping[str, Any]:
        value = preprocessing.get(section)
        return value if isinstance(value, Mapping) else {}

    background = branch("background").get("basis")
    if background is None:
        return None
    denoise = branch("denoise")
    return {
        "theta_min": preprocessing.get("theta_min"),
        "theta_max": preprocessing.get("theta_max"),
        "gridpoints": preprocessing.get("gridpoints"),
        "background": background,
        "denoise": denoise.get("basis", "none"),
        "denoise_window": int(denoise.get("window", 7)),
        "denoise_polyorder": int(denoise.get("polyorder", 2)),
        "signature_reduce_mode": preprocessing.get("signature_reduce_mode", "merge"),
        "signature_cleaning": branch("signature_cleaning").get("basis", "none"),
        "width_floor_abs": float(options["width_floor_abs"]),
    }


def _prior_for_samples(
    config: ProjectConfig, samples: Sequence[Mapping[str, Any]], options: Mapping[str, Any]
):
    considered = options["element_prior"]["considered_elements"] or None
    source = options["element_prior"]["sample_description_path"]
    if source not in (None, ""):
        try:
            path = _resolve_config_path(config, source)
        except Exception as exc:
            raise ValueError(
                f"Unable to resolve sample description path {source!s}: {exc}"
            ) from exc
        try:
            prior = build_element_prior(path, considered_elements=considered)
        except Exception as exc:
            raise ValueError(
                f"Unable to read or parse sample description path {path}: {exc}"
            ) from exc
        if not prior:
            raise ValueError(f"Sample description path {path} contains no usable sample priors")
        return prior
    rows = [
        {"sample_id": sample.get("sample_id"), "composition": sample.get("composition")}
        for sample in samples
        if sample.get("composition")
    ]
    return build_element_prior(rows, considered_elements=considered)


def _segment_concat_scale(
    basis_signatures: Sequence[tuple[Any, Any]],
    target_signature: tuple[Any, Any],
    epsilon: float = 1e-19,
) -> float:
    """Return the historical ``segment_concat`` transport scale ``W_``.

    ``segmentation.segment_anom_milp`` uses a uniform-abundance monotone
    transport pass before constructing relative parameters.  Keeping this
    calculation at the seam makes beta and sparsity penalties numerically
    equivalent to the SOTU path without carrying over its optimisation or
    plotting machinery.
    """

    target_positions, target_weights = target_signature
    target_positions = np.asarray(target_positions, dtype=float).reshape(-1)
    target_weights = np.asarray(target_weights, dtype=float).reshape(-1)
    if not basis_signatures or target_positions.size == 0:
        return 0.0

    # ``pad_sig`` is applied before ``segment_concat1`` in the historical
    # workflow.  Keep its fixed 1e-19 padding mass and normalization exactly;
    # ``epsilon`` below controls only the optimizer's abundance lower bound.
    pad_length = max(
        [
            target_positions.size,
            *(len(np.asarray(item[0]).reshape(-1)) for item in basis_signatures),
        ]
    )

    def pad_signature(positions, weights):
        positions = np.asarray(positions, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        padded_positions = np.arange(pad_length) * 1e-19
        padded_weights = np.ones(pad_length) * 1e-19
        padded_positions[: positions.size] = positions
        padded_positions[positions.size :] += positions[-1]
        padded_weights[: weights.size] = weights
        padded_weights /= padded_weights.sum()
        return padded_positions, padded_weights

    target_positions, target_weights = pad_signature(target_positions, target_weights)
    padded_basis = [pad_signature(item[0], item[1]) for item in basis_signatures]
    K = len(padded_basis)
    N = len(target_positions)
    XB = np.zeros((K, N))
    B = np.zeros((K, N))
    for i, (xb, b) in enumerate(padded_basis):
        XB[i] = xb
        B[i] = b

    # This is ``segment_cost`` from the old solver, including its concatenated
    # arrays, stable sort and source/basis traversal arithmetic.
    def segment_cost(av):
        xct = np.zeros(N * K)
        ct = np.zeros(N * K)
        Bt = np.zeros((K, N))
        for i in range(K):
            xct[i * N : (i + 1) * N] = XB[i]
            ct[i * N : (i + 1) * N] = av[i] * B[i]
        ids = list(sorted(range(N * K), key=lambda i: xct[i]))
        i = 0
        j = 0
        sc = target_weights[0]
        sct = ct[ids[0]]
        W = 0.0
        for _ in range((K + 1) * N):
            p = ids[j] // N
            if sc < sct:
                sct -= sc
                Bt[p, i] += sc
                W += sc * (target_positions[i] - xct[ids[j]]) ** 2
                if i < N - 1:
                    i += 1
                    sc = target_weights[i]
                else:
                    break
            else:
                sc -= sct
                Bt[p, i] += sct
                W += sct * (target_positions[i] - xct[ids[j]]) ** 2
                if j < N * K - 1:
                    j += 1
                    sct = ct[ids[j]]
                else:
                    break
        return W

    h0 = np.ones(K) / K
    W_ = segment_cost(h0 / h0.sum())
    gamma = 1000 * W_

    def objective(hv):
        sc = segment_cost(hv / hv.sum())
        return sc + gamma * (hv.sum() - 1) ** 2

    result = minimize(objective, h0, bounds=[(epsilon, 1) for _ in range(K)], method=None)
    hv = result.x
    return float(segment_cost(hv / hv.sum()))


def _solve_one(
    sample: Mapping[str, Any],
    library: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any],
    prior_map: Mapping[str, Any],
):
    try:
        target = _signature(
            sample["signature_positions"],
            sample["signature_weights"],
            name=f"sample {sample.get('sample_id', '')!r} signature",
        )
    except KeyError as exc:
        raise ValueError(
            f"Sample {sample.get('sample_id', '')!r} is missing {exc.args[0]}"
        ) from exc
    if not library:
        raise ValueError("Cannot solve a mixture with an empty basis library")
    validated_library: list[Mapping[str, Any]] = []
    names: list[str] = []
    for index, item in enumerate(library):
        phase_id = str(item.get("phase_id", "")).strip()
        if not phase_id:
            raise ValueError(f"Basis record {index + 1} has a blank phase_id")
        try:
            signature = _signature(
                item["signature"][0], item["signature"][1], name=f"basis {phase_id!r} signature"
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Basis {phase_id!r} has no valid signature pair") from exc
        record = dict(item)
        record["signature"] = signature
        validated_library.append(record)
        names.append(phase_id)
    if len(set(names)) != len(names):
        raise ValueError("Basis library contains duplicate phase_id values")
    library = validated_library
    element_sets = [set(item.get("elements", set())) for item in library]
    use_prior = options["element_prior"]["enabled"]
    prior = None
    if use_prior:
        sample_name = str(sample["sample_id"])
        normalized_name = normalize_sample_name(sample_name)
        prior = (
            prior_map.get(sample_name)
            or prior_map.get(sample_name.upper())
            or (prior_map.get(normalized_name) if normalized_name else None)
        )
        if prior is None:
            raise ValueError(f"Element prior is missing for sample_id {sample_name!r}")
    keep, removed = filter_basis_indices(names, element_sets, prior)
    selected_library = [library[index] for index in keep]
    selected_sigs = [item["signature"] for item in selected_library]
    max_base = resolve_sample_max_base(
        options["max_base"],
        prior,
        default_elements=options["element_prior"]["considered_elements"] or [],
        max_base_extra=int(options["max_base_extra"] or 0),
        n_candidates=len(selected_library),
    )
    if not selected_sigs:
        zeros = [np.zeros(target[1].size, dtype=float) for _ in library]
        return {
            "components": zeros,
            "anomaly": target[1].copy(),
            "objective": float(target[1].sum()),
            "solver": "empty",
            "selected_indices": [],
            "selected_global_indices": [],
            "warnings": ["element prior removed every basis"],
            "element_prior": prior or {},
            "removed": format_removed_phases(removed),
        }
    width_mode = str(options["width_prior_mode"] or "none").lower()
    if width_mode not in {"none", "inst_width_tau_v1"}:
        raise ValueError(f"Unsupported width_prior_mode {options['width_prior_mode']!r}")
    row_budgets = None
    if width_mode == "inst_width_tau_v1":
        delta_tau = options["delta_tau"]
        if delta_tau is None:
            raise ValueError("delta_tau is required for width_prior_mode=inst_width_tau_v1")
        delta_floor = options["delta_floor"]
        width_floor_abs = options["width_floor_abs"]
        row_budgets = []
        for item in selected_library:
            try:
                budgets, _widths = build_local_disp_budgets(
                    item["signature"][0],
                    item.get("signature_widths"),
                    delta_tau=delta_tau,
                    delta_floor=delta_floor,
                    width_floor_abs=width_floor_abs,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Basis {item['phase_id']!r} width prior is invalid: {exc}"
                ) from exc
            row_budgets.append(budgets)
    solver_name = str(options["solver"]).lower()
    formulation = str(options["formulation"]).lower()
    if formulation not in {"milp", "cpu", "nnls", "greedy"}:
        raise ValueError(f"Unsupported solver formulation {options['formulation']!r}")
    if solver_name not in {"auto", "milp", "gurobi", "cpu", "nnls", "greedy"}:
        raise ValueError(f"Unsupported solver {options['solver']!r}")
    warnings: list[str] = []
    diagnostics: dict[str, Any]
    local_components: list[Any]
    anomaly: Any
    try_milp = solver_name not in {"cpu", "greedy", "nnls"} and formulation == "milp"
    if try_milp:
        cost_mats = [
            pairwise_cost(sig[0], target[0], p=float(options["p"])) for sig in selected_sigs
        ]
        disp_mats = [
            np.abs(np.asarray(sig[0])[:, None] - target[0][None, :]) for sig in selected_sigs
        ]
        # Relative SOTU penalties are scaled by the uniform-abundance
        # ``segment_concat`` cost before entering the MILP.  This is the
        # quantity used by Param.get_value() in the historical implementation.
        scale = _segment_concat_scale(selected_sigs, target, epsilon=float(options["epsilon"]))
        milp_result = solve_anomaly_milp(
            selected_sigs,
            target,
            cost_mats=cost_mats,
            disp_mats=disp_mats,
            max_emd_rows=row_budgets,
            anomaly_penalty=float(options["beta"]) * scale,
            missing_element_penalty=float(options["missing_element_penalty"]) * scale,
            soft_sparse_penalty=float(options["soft_sparse_penalty"]) * scale,
            present_elements=sorted((prior or {}).get("present", set())),
            basis_element_sets=[element_sets[index] for index in keep],
            max_base=max_base,
            allow_anomaly=bool(options["allow_anomaly"]),
            resolution=float(options["resolution"]),
            epsilon=float(options["epsilon"]),
            anomaly_relax=float(options["anomaly_relax"]),
            p=float(options["p"]),
            gurobi_threads=options["gurobi_threads"],
            width_budget_mode=options["width_budget_mode"],
            strict_solver=bool(options["strict_solver"]),
        )
        flows, _, anomaly, diagnostics = milp_result
        diagnostics["segment_concat_scale"] = float(scale)
        diagnostics["relative_penalties"] = {
            "beta": float(options["beta"]),
            "missing_element_penalty": float(options["missing_element_penalty"]),
            "soft_sparse_penalty": float(options["soft_sparse_penalty"]),
        }
        local_components = [flow.sum(axis=0) for flow in flows]
    else:
        if bool(options["strict_solver"]) and solver_name == "auto" and formulation == "milp":
            raise RuntimeError(
                "strict_solver=true requires solver=milp or an explicit non-MILP formulation"
            )
        local_components, anomaly, diagnostics = cpu_anomaly_solution(
            selected_sigs,
            target,
            max_base=max_base,
            anomaly_penalty=options["beta"],
            p=float(options["p"]),
            selection_threshold=float(options["selection_threshold"]),
        )
    # Both solvers return one finite component per selected basis and a finite
    # anomaly on the target grid; only their sign needs checking here.
    anomaly = np.asarray(anomaly, dtype=float).reshape(-1)
    if np.any(anomaly < -_SOLVER_NEGATIVE_TOL):
        raise RuntimeError("Solver returned a negative anomaly profile")
    for index, component in enumerate(local_components):
        if np.any(np.asarray(component) < -_SOLVER_NEGATIVE_TOL):
            raise RuntimeError(f"Solver returned a negative component at local index {index}")
    status_name = diagnostics.get("status_name")
    if status_name and status_name != "OPTIMAL":
        warnings.append(f"Gurobi status: {status_name}")
    full_components = [np.zeros(target[1].size, dtype=float) for _ in library]
    for local_index, global_index in enumerate(keep):
        full_components[global_index] = np.asarray(local_components[local_index], dtype=float)
    # ``multi_segment.cpu._cfbk`` maps filtered components back to the full
    # library and normalizes the complete basis+anomaly stack.  The MILP
    # tolerances permit a tiny mass residual (typically ~1e-5), so preserve
    # that historical public contract explicitly before abundance export.
    total_mass = float(sum(np.sum(component) for component in full_components) + np.sum(anomaly))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise RuntimeError("Solver returned no finite positive reconstructed mass")
    full_components = [component / total_mass for component in full_components]
    anomaly = np.maximum(anomaly / total_mass, 0.0)
    selected_local = list(diagnostics["selected_indices"])
    selected_global = [keep[index] for index in selected_local]
    covered, missing, violated = check_element_coverage(element_sets, selected_global, prior)
    # The MILP reports its transport cost apart from the penalized objective.
    objective = float(diagnostics.get("transport_cost", diagnostics.get("objective")))
    return {
        "components": full_components,
        "anomaly": anomaly,
        "objective": objective,
        "solver": diagnostics.get("solver", "cpu"),
        "selected_indices": selected_local,
        "selected_global_indices": selected_global,
        "warnings": warnings,
        "element_prior": {
            "present": sorted((prior or {}).get("present", set())),
            "absent": sorted((prior or {}).get("absent", set())),
        },
        "removed": format_removed_phases(removed),
        "element_covered": sorted(covered),
        "element_missing": sorted(missing),
        "element_violated": sorted(violated),
        "diagnostics": diagnostics,
    }


def _solve_one_worker(args):
    """Pickle-safe worker entry point for deterministic sample-level mapping."""

    sample, library, options, prior_map = args
    return _solve_one(sample, library, options, prior_map)

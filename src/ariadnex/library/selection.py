"""Select a clustering threshold using the original stability/compression rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The paper's ARI cutoff; with the open-world library it selects 154 bases.
DEFAULT_MIN_ARI = 0.96


@dataclass(frozen=True)
class StableRegimeConfig:
    """Compression-oriented stable-regime rule set."""

    policy_name: str = "compression"
    min_consecutive_thresholds: int = 3
    interval_selection: str = "right_endpoint"
    min_ari: float = DEFAULT_MIN_ARI
    min_nmi: float = 0.98
    min_medoid_jaccard: float = 0.43
    max_medoid_matched_distance: float = 0.105
    max_repeat_cluster_cv: float = 0.04
    max_neighbor_cluster_change_ratio: float = 0.20
    max_singleton_ratio: float = 0.85
    max_largest_cluster_ratio: float = 0.251
    max_coverage_90: float = 0.14
    max_compactness_separation_ratio: float = 0.125
    fallback_weight_compression: float = 0.28
    fallback_weight_ari: float = 0.18
    fallback_weight_nmi: float = 0.12
    fallback_weight_medoid_jaccard: float = 0.10
    fallback_weight_medoid_distance: float = 0.09
    fallback_weight_singleton_ratio: float = 0.08
    fallback_weight_largest_cluster_ratio: float = 0.05
    fallback_weight_coverage: float = 0.05
    fallback_weight_compactness: float = 0.05


@dataclass
class StableRegimeResult:
    """Stable-regime detection and final-threshold selection result."""

    summary: pd.DataFrame
    stable_intervals: list[tuple[float, float]]
    selected_threshold: float
    selected_index: int
    selected_interval_index: int | None
    selection_reason: str
    criteria: dict[str, float | str]


def _neighbor_cluster_change_ratio(cluster_counts: np.ndarray) -> np.ndarray:
    if cluster_counts.size == 0:
        return np.asarray([], dtype=np.float64)
    ratios = np.zeros(cluster_counts.size, dtype=np.float64)
    for index in range(cluster_counts.size):
        current = max(float(cluster_counts[index]), 1.0)
        neighbors: list[float] = []
        if index > 0:
            neighbors.append(
                abs(float(cluster_counts[index]) - float(cluster_counts[index - 1])) / current
            )
        if index + 1 < cluster_counts.size:
            neighbors.append(
                abs(float(cluster_counts[index + 1]) - float(cluster_counts[index])) / current
            )
        ratios[index] = max(neighbors) if neighbors else 0.0
    return ratios


def _intervals_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        if not value and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(mask) - 1))
    return intervals


def _rank_desirability(values: pd.Series, *, higher_is_better: bool) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    valid_mask = numeric.notna().to_numpy(dtype=bool)
    desirability = np.zeros(len(numeric), dtype=np.float64)
    if valid_mask.sum() == 0:
        return desirability

    valid_values = numeric[valid_mask]
    ranks = valid_values.rank(method="average", pct=True).to_numpy(dtype=np.float64)
    if higher_is_better:
        desirability[valid_mask] = ranks
    else:
        desirability[valid_mask] = 1.0 - ranks + (1.0 / float(valid_values.shape[0]))
    return desirability


def identify_stable_regime(
    threshold_scan: pd.DataFrame,
    stability_summary: pd.DataFrame,
    config: StableRegimeConfig,
) -> StableRegimeResult:
    """Identify stable threshold intervals and select the final operating point."""

    if config.interval_selection not in {"right_endpoint", "penultimate"}:
        raise ValueError("interval_selection must be 'right_endpoint' or 'penultimate'")
    if config.interval_selection == "penultimate" and config.min_consecutive_thresholds < 2:
        raise ValueError("penultimate selection requires at least two consecutive thresholds")

    merged = threshold_scan.merge(
        stability_summary, on="threshold", how="inner", validate="one_to_one"
    )
    merged = merged.sort_values("threshold", kind="stable").reset_index(drop=True)
    thresholds = merged["threshold"].to_numpy(dtype=np.float64)
    cluster_counts = merged["n_clusters"].to_numpy(dtype=np.float64)
    merged["neighbor_cluster_change_ratio"] = _neighbor_cluster_change_ratio(cluster_counts)

    criteria: dict[str, float | str] = {
        "policy_name": config.policy_name,
        "selection_rule": f"{config.interval_selection}_of_rightmost_stable_interval",
        "fallback_rule": "ranked_compression_tradeoff",
        "ari_cutoff": config.min_ari,
        "nmi_cutoff": config.min_nmi,
        "medoid_jaccard_cutoff": config.min_medoid_jaccard,
        "medoid_distance_cutoff": config.max_medoid_matched_distance,
        "repeat_cluster_cv_cutoff": config.max_repeat_cluster_cv,
        "coverage_cutoff": config.max_coverage_90,
        "compactness_cutoff": config.max_compactness_separation_ratio,
    }

    merged["rule_ari"] = merged["ari_mean"] >= criteria["ari_cutoff"]
    merged["rule_nmi"] = merged["nmi_mean"] >= criteria["nmi_cutoff"]
    merged["rule_medoid_jaccard"] = (
        merged["medoid_exact_jaccard_mean"] >= criteria["medoid_jaccard_cutoff"]
    )
    merged["rule_medoid_distance"] = (
        merged["medoid_matched_distance_mean"] <= criteria["medoid_distance_cutoff"]
    )
    merged["rule_repeat_cluster_cv"] = (
        merged["repeat_n_clusters_cv"] <= criteria["repeat_cluster_cv_cutoff"]
    )
    merged["rule_cluster_smoothness"] = (
        merged["neighbor_cluster_change_ratio"] <= config.max_neighbor_cluster_change_ratio
    )
    merged["rule_singleton_ratio"] = merged["singleton_ratio"] <= config.max_singleton_ratio
    merged["rule_largest_cluster_ratio"] = (
        merged["largest_cluster_ratio"] <= config.max_largest_cluster_ratio
    )
    merged["rule_coverage"] = merged["coverage_90"] <= criteria["coverage_cutoff"]
    merged["rule_compactness"] = (
        merged["compactness_separation_ratio"] <= criteria["compactness_cutoff"]
    )

    merged["stable_candidate"] = merged[
        [
            "rule_ari",
            "rule_nmi",
            "rule_medoid_jaccard",
            "rule_medoid_distance",
            "rule_repeat_cluster_cv",
            "rule_cluster_smoothness",
            "rule_singleton_ratio",
            "rule_largest_cluster_ratio",
            "rule_coverage",
            "rule_compactness",
        ]
    ].all(axis=1)

    raw_intervals = _intervals_from_mask(merged["stable_candidate"].to_numpy(dtype=bool))
    filtered_intervals = [
        (start, end)
        for start, end in raw_intervals
        if (end - start + 1) >= int(config.min_consecutive_thresholds)
    ]

    merged["stable_regime_id"] = np.nan
    merged["is_stable_regime"] = False
    for regime_id, (start, end) in enumerate(filtered_intervals):
        merged.loc[start:end, "stable_regime_id"] = float(regime_id)
        merged.loc[start:end, "is_stable_regime"] = True

    selected_interval_index: int | None = None
    selection_reason: str
    if filtered_intervals:
        selected_interval_index = len(filtered_intervals) - 1
        interval_start, interval_end = filtered_intervals[selected_interval_index]
        selected_index = int(interval_end) - (config.interval_selection == "penultimate")
        selection_reason = (
            "Selected the penultimate scanned threshold of the rightmost stable interval."
            if config.interval_selection == "penultimate"
            else "Selected the right endpoint of the rightmost stable interval."
        )
    else:
        merged["score_compression"] = _rank_desirability(
            1.0 - merged["compression_ratio"], higher_is_better=True
        )
        merged["score_ari"] = _rank_desirability(merged["ari_mean"], higher_is_better=True)
        merged["score_nmi"] = _rank_desirability(merged["nmi_mean"], higher_is_better=True)
        merged["score_medoid_jaccard"] = _rank_desirability(
            merged["medoid_exact_jaccard_mean"],
            higher_is_better=True,
        )
        merged["score_medoid_distance"] = _rank_desirability(
            merged["medoid_matched_distance_mean"],
            higher_is_better=False,
        )
        merged["score_singleton_ratio"] = _rank_desirability(
            merged["singleton_ratio"],
            higher_is_better=False,
        )
        merged["score_largest_cluster_ratio"] = _rank_desirability(
            merged["largest_cluster_ratio"],
            higher_is_better=False,
        )
        merged["score_coverage"] = _rank_desirability(
            merged["coverage_90"],
            higher_is_better=False,
        )
        merged["score_compactness"] = _rank_desirability(
            merged["compactness_separation_ratio"],
            higher_is_better=False,
        )
        merged["fallback_score"] = (
            config.fallback_weight_compression * merged["score_compression"]
            + config.fallback_weight_ari * merged["score_ari"]
            + config.fallback_weight_nmi * merged["score_nmi"]
            + config.fallback_weight_medoid_jaccard * merged["score_medoid_jaccard"]
            + config.fallback_weight_medoid_distance * merged["score_medoid_distance"]
            + config.fallback_weight_singleton_ratio * merged["score_singleton_ratio"]
            + config.fallback_weight_largest_cluster_ratio * merged["score_largest_cluster_ratio"]
            + config.fallback_weight_coverage * merged["score_coverage"]
            + config.fallback_weight_compactness * merged["score_compactness"]
        )
        selected_index = int(merged["fallback_score"].idxmax())
        selection_reason = (
            "No interval satisfied every stable-regime rule; selected the threshold with the "
            "best normalized fallback score under the configured compression/stability tradeoff."
        )

    if "fallback_score" not in merged.columns:
        merged["fallback_score"] = np.nan
    merged["selection_score"] = merged["fallback_score"]

    merged["selected_threshold_flag"] = False
    merged.loc[selected_index, "selected_threshold_flag"] = True

    stable_intervals = [
        (float(thresholds[start]), float(thresholds[end])) for start, end in filtered_intervals
    ]

    return StableRegimeResult(
        summary=merged,
        stable_intervals=stable_intervals,
        selected_threshold=float(merged.loc[selected_index, "threshold"]),
        selected_index=selected_index,
        selected_interval_index=selected_interval_index,
        selection_reason=selection_reason,
        criteria=criteria,
    )

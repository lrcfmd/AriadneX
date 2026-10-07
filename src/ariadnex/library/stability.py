"""Repeated subsampling diagnostics for automatic basis-library selection."""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .clustering import cut_linkage_at_threshold, fit_average_linkage


@dataclass(frozen=True)
class StabilityConfig:
    """Repeated subsampling stability settings."""

    repeats: int = 50
    sample_fraction: float = 0.80
    random_seed: int = 42
    medoid_unmatched_penalty_quantile: float = 0.90

    def __post_init__(self) -> None:
        if self.repeats < 2:
            raise ValueError("stability.repeats must be at least 2")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("stability.sample_fraction must be in (0, 1]")
        if not 0 <= self.medoid_unmatched_penalty_quantile <= 1:
            raise ValueError("stability.medoid_unmatched_penalty_quantile must be in [0, 1]")


@dataclass
class RepeatClusteringResult:
    """Cached clustering outputs for one subsampling repeat."""

    sampled_indices: np.ndarray
    labels_by_threshold: list[np.ndarray]
    medoids_by_threshold: list[np.ndarray]
    n_clusters_by_threshold: np.ndarray


@dataclass
class StabilityAnalysisResult:
    """Aggregate stability outputs and raw repeat results."""

    summary: pd.DataFrame
    repeat_results: list[RepeatClusteringResult]
    medoid_unmatched_penalty: float


def _select_medoid_indices(distance_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    medoid_indices: list[int] = []
    for cluster_label in np.unique(labels):
        members = np.flatnonzero(labels == cluster_label)
        if members.size == 1:
            medoid_indices.append(int(members[0]))
            continue
        submatrix = distance_matrix[np.ix_(members, members)].astype(np.float64)
        sums = submatrix.sum(axis=1)
        best = float(sums.min())
        tied = np.flatnonzero(np.isclose(sums, best, atol=1e-12))
        medoid_indices.append(int(members[int(tied[0])]))
    return np.asarray(medoid_indices, dtype=np.int32)


def _exact_medoid_jaccard(medoids_a: np.ndarray, medoids_b: np.ndarray) -> float:
    set_a = set(int(index) for index in medoids_a.tolist())
    set_b = set(int(index) for index in medoids_b.tolist())
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return float(len(set_a & set_b) / len(union))


def _matched_medoid_distance(
    full_distance_matrix: np.ndarray,
    medoids_a: np.ndarray,
    medoids_b: np.ndarray,
    *,
    unmatched_penalty: float,
) -> float:
    if medoids_a.size == 0 and medoids_b.size == 0:
        return 0.0
    if medoids_a.size == 0 or medoids_b.size == 0:
        return float(unmatched_penalty)

    cost = full_distance_matrix[np.ix_(medoids_a, medoids_b)].astype(np.float64)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_sum = float(cost[row_ind, col_ind].sum())
    unmatched = abs(int(medoids_a.size) - int(medoids_b.size))
    denominator = max(int(medoids_a.size), int(medoids_b.size))
    return float((matched_sum + unmatched_penalty * unmatched) / denominator)


def _pair_intersection_positions(
    repeat_results: list[RepeatClusteringResult],
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    pair_cache: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for left in range(len(repeat_results)):
        for right in range(left + 1, len(repeat_results)):
            _, left_pos, right_pos = np.intersect1d(
                repeat_results[left].sampled_indices,
                repeat_results[right].sampled_indices,
                assume_unique=True,
                return_indices=True,
            )
            pair_cache.append((left, right, left_pos.astype(np.int32), right_pos.astype(np.int32)))
    return pair_cache


def run_subsampling_stability_analysis(
    *,
    distance_matrix: np.ndarray,
    thresholds: np.ndarray,
    config: StabilityConfig,
) -> StabilityAnalysisResult:
    """Run repeated subsampling stability analysis over a threshold grid."""

    n_samples = int(distance_matrix.shape[0])
    if n_samples < 2:
        raise ValueError("Need at least two samples for stability analysis.")

    sample_size = max(2, int(round(config.sample_fraction * n_samples)))
    rng = np.random.default_rng(config.random_seed)

    upper_tri = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    unmatched_penalty = float(
        np.quantile(upper_tri.astype(np.float64), config.medoid_unmatched_penalty_quantile)
    )

    repeat_results: list[RepeatClusteringResult] = []
    for repeat in range(int(config.repeats)):
        sampled_indices = np.sort(
            rng.choice(np.arange(n_samples, dtype=np.int32), size=sample_size, replace=False)
        )
        sampled_distance = distance_matrix[np.ix_(sampled_indices, sampled_indices)]
        linkage_artifacts = fit_average_linkage(sampled_distance, optimal_ordering=False)

        labels_by_threshold: list[np.ndarray] = []
        medoids_by_threshold: list[np.ndarray] = []
        n_clusters_by_threshold: list[int] = []

        for threshold in thresholds.tolist():
            labels = cut_linkage_at_threshold(linkage_artifacts.linkage_matrix, float(threshold))
            local_medoids = _select_medoid_indices(sampled_distance, labels)
            global_medoids = sampled_indices[local_medoids]
            labels_by_threshold.append(labels.astype(np.int32))
            medoids_by_threshold.append(global_medoids.astype(np.int32))
            n_clusters_by_threshold.append(int(np.unique(labels).size))

        repeat_results.append(
            RepeatClusteringResult(
                sampled_indices=sampled_indices.astype(np.int32),
                labels_by_threshold=labels_by_threshold,
                medoids_by_threshold=medoids_by_threshold,
                n_clusters_by_threshold=np.asarray(n_clusters_by_threshold, dtype=np.int32),
            )
        )
        if (repeat + 1) % 10 == 0 or repeat + 1 == int(config.repeats):
            print(
                f"[library] clustering subsamples: {repeat + 1}/{config.repeats}",
                file=sys.stderr,
                flush=True,
            )

    pair_cache = _pair_intersection_positions(repeat_results)
    stability_rows: list[dict[str, float | int]] = []

    for threshold_index, threshold in enumerate(thresholds.tolist()):
        ari_values: list[float] = []
        nmi_values: list[float] = []
        medoid_jaccard_values: list[float] = []
        medoid_match_distance_values: list[float] = []
        repeat_cluster_counts = np.asarray(
            [result.n_clusters_by_threshold[threshold_index] for result in repeat_results],
            dtype=np.float64,
        )

        for left, right, left_pos, right_pos in pair_cache:
            if left_pos.size >= 2:
                labels_left = repeat_results[left].labels_by_threshold[threshold_index][left_pos]
                labels_right = repeat_results[right].labels_by_threshold[threshold_index][right_pos]
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="The number of unique classes is greater than 50% of the number of samples.*",
                        category=UserWarning,
                    )
                    ari_values.append(float(adjusted_rand_score(labels_left, labels_right)))
                    nmi_values.append(
                        float(normalized_mutual_info_score(labels_left, labels_right))
                    )

            medoids_left = repeat_results[left].medoids_by_threshold[threshold_index]
            medoids_right = repeat_results[right].medoids_by_threshold[threshold_index]
            medoid_jaccard_values.append(_exact_medoid_jaccard(medoids_left, medoids_right))
            medoid_match_distance_values.append(
                _matched_medoid_distance(
                    distance_matrix,
                    medoids_left,
                    medoids_right,
                    unmatched_penalty=unmatched_penalty,
                )
            )

        cluster_count_mean = float(repeat_cluster_counts.mean())
        cluster_count_std = float(repeat_cluster_counts.std(ddof=0))
        stability_rows.append(
            {
                "threshold": float(threshold),
                "stability_pairs": int(len(pair_cache)),
                "ari_mean": float(np.mean(ari_values)) if ari_values else np.nan,
                "ari_std": float(np.std(ari_values, ddof=0)) if ari_values else np.nan,
                "nmi_mean": float(np.mean(nmi_values)) if nmi_values else np.nan,
                "nmi_std": float(np.std(nmi_values, ddof=0)) if nmi_values else np.nan,
                "medoid_exact_jaccard_mean": float(np.mean(medoid_jaccard_values))
                if medoid_jaccard_values
                else np.nan,
                "medoid_exact_jaccard_std": float(np.std(medoid_jaccard_values, ddof=0))
                if medoid_jaccard_values
                else np.nan,
                "medoid_matched_distance_mean": float(np.mean(medoid_match_distance_values))
                if medoid_match_distance_values
                else np.nan,
                "medoid_matched_distance_std": float(np.std(medoid_match_distance_values, ddof=0))
                if medoid_match_distance_values
                else np.nan,
                "repeat_n_clusters_mean": cluster_count_mean,
                "repeat_n_clusters_std": cluster_count_std,
                "repeat_n_clusters_cv": float(cluster_count_std / cluster_count_mean)
                if cluster_count_mean > 0
                else np.nan,
                "medoid_unmatched_penalty": unmatched_penalty,
            }
        )
        if (threshold_index + 1) % 10 == 0 or threshold_index + 1 == len(thresholds):
            print(
                f"[library] stability thresholds: {threshold_index + 1}/{len(thresholds)}",
                file=sys.stderr,
                flush=True,
            )

    return StabilityAnalysisResult(
        summary=pd.DataFrame(stability_rows),
        repeat_results=repeat_results,
        medoid_unmatched_penalty=unmatched_penalty,
    )

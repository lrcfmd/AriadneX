"""Deterministic average-linkage clustering for PXRD basis selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .artifact import ClusterResult
from .selection import DEFAULT_MIN_ARI


@dataclass(frozen=True)
class HierarchicalClustering:
    distance_matrix: np.ndarray
    linkage_matrix: np.ndarray
    merge_heights: np.ndarray


@dataclass
class MedoidSelection:
    labels: np.ndarray
    medoid_indices: np.ndarray
    assignment_medoid_indices: np.ndarray
    distances_to_medoid: np.ndarray
    representatives: Any


def _distance_array(distance_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distance_matrix contains non-finite values")
    if np.any(matrix < -1e-9):
        raise ValueError("distance_matrix contains negative distances")
    matrix = np.maximum(matrix, 0.0)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def l2_normalize_embeddings(embeddings: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    # Keep float32 arithmetic here to match the checkpoint embedding output and
    # the historical clustering implementation.  Promoting to float64 can
    # perturb very close linkage heights and change a fixed-count cut.
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(eps))


def compute_cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = l2_normalize_embeddings(embeddings)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distance = np.clip(1.0 - similarity, 0.0, 2.0).astype(np.float32)
    np.fill_diagonal(distance, 0.0)
    return distance


def fit_average_linkage(
    distance_matrix: np.ndarray, *, optimal_ordering: bool = True
) -> HierarchicalClustering:
    matrix = _distance_array(distance_matrix)
    n_samples = matrix.shape[0]
    if n_samples < 2:
        return HierarchicalClustering(
            matrix.astype(np.float32),
            np.zeros((0, 4), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    condensed = squareform(matrix, checks=False).astype(np.float64)
    linkage_matrix = linkage(
        condensed, method="average", optimal_ordering=bool(optimal_ordering)
    ).astype(np.float64)
    return HierarchicalClustering(
        distance_matrix=matrix.astype(np.float32),
        linkage_matrix=linkage_matrix,
        merge_heights=linkage_matrix[:, 2].copy(),
    )


def build_threshold_grid(
    merge_heights: np.ndarray,
    *,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
    n_thresholds: int = 60,
) -> np.ndarray:
    heights = np.asarray(merge_heights, dtype=np.float64).reshape(-1)
    if heights.size == 0:
        return np.zeros(0, dtype=np.float64)
    if not (0.0 <= float(lower_quantile) < float(upper_quantile) <= 1.0):
        raise ValueError("Expected 0 <= lower_quantile < upper_quantile <= 1")
    if int(n_thresholds) < 1:
        raise ValueError("n_thresholds must be positive")
    unique = np.unique(heights)
    lower = float(np.quantile(unique, lower_quantile))
    upper = float(np.quantile(unique, upper_quantile))
    candidates = unique[(unique >= lower) & (unique <= upper)]
    if candidates.size == 0:
        candidates = unique
    if candidates.size <= int(n_thresholds):
        return candidates
    positions = np.linspace(0, candidates.size - 1, num=int(n_thresholds))
    return np.unique(candidates[np.round(positions).astype(int)])


def cut_linkage_at_threshold(linkage_matrix: np.ndarray, threshold: float) -> np.ndarray:
    linkage_values = np.asarray(linkage_matrix, dtype=np.float64)
    if linkage_values.ndim != 2 or linkage_values.shape[1] != 4:
        raise ValueError("linkage_matrix must have shape (n-1, 4)")
    if linkage_values.shape[0] == 0:
        return np.zeros(1, dtype=np.int32)
    raw = fcluster(linkage_values, t=float(threshold), criterion="distance")
    return _relabel_labels(raw)


def cut_linkage_at_count(
    linkage_matrix: np.ndarray, n_samples: int, target_clusters: int
) -> np.ndarray:
    """Cut exactly ``target_clusters`` from a linkage matrix.

    ``fcluster(..., criterion='maxclust')`` can return fewer clusters when merge
    heights tie.  Applying the first ``n-target`` merges directly guarantees the
    fixed-count contract and remains deterministic under ties.
    """

    n = int(n_samples)
    target = int(target_clusters)
    if n < 1:
        raise ValueError("n_samples must be positive")
    if target < 1 or target > n:
        raise ValueError(f"target_clusters must be in [1, {n}], got {target}")
    matrix = np.asarray(linkage_matrix, dtype=np.float64)
    if matrix.shape != (max(0, n - 1), 4):
        raise ValueError(f"linkage_matrix must have shape ({max(0, n - 1)}, 4)")
    merge_count = n - target
    members: list[list[int]] = [[index] for index in range(n)]
    for row_index, row in enumerate(matrix[:merge_count]):
        left, right = int(row[0]), int(row[1])
        merged = sorted([*members[left], *members[right]])
        members.append(merged)
    # Recompute labels from the active forest using linkage-node members.
    labels = np.empty(n, dtype=np.int32)
    active_nodes = list(range(n + merge_count))
    merged_nodes = {int(row[0]) for row in matrix[:merge_count]} | {
        int(row[1]) for row in matrix[:merge_count]
    }
    # Nodes participating in a merge are represented by the newly-created node.
    child_nodes = set(merged_nodes)
    final_nodes = [node for node in active_nodes if node not in child_nodes]
    final_nodes.sort(key=lambda node: (members[node][0], node))
    for label, node in enumerate(final_nodes):
        labels[members[node]] = label
    return labels


def _relabel_labels(raw_labels: Sequence[int]) -> np.ndarray:
    labels = np.asarray(raw_labels)
    mapping: dict[int, int] = {}
    output = np.empty(labels.shape[0], dtype=np.int32)
    next_label = 0
    for index, raw in enumerate(labels.tolist()):
        key = int(raw)
        if key not in mapping:
            mapping[key] = next_label
            next_label += 1
        output[index] = mapping[key]
    return output


def select_cluster_medoids(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    sample_ids: Sequence[str],
    *,
    source_order_ties: bool = False,
) -> MedoidSelection:
    matrix = _distance_array(distance_matrix)
    labels_array = np.asarray(labels, dtype=np.int32).reshape(-1)
    ids = [str(item) for item in sample_ids]
    n = matrix.shape[0]
    if labels_array.shape[0] != n or len(ids) != n:
        raise ValueError("distance_matrix, labels, and sample_ids must have matching sample counts")
    assignments = np.empty(n, dtype=np.int32)
    distances = np.empty(n, dtype=np.float32)
    medoids: list[int] = []
    rows: list[dict[str, Any]] = []
    for cluster_label in sorted(int(value) for value in np.unique(labels_array)):
        members = np.flatnonzero(labels_array == cluster_label)
        submatrix = matrix[np.ix_(members, members)]
        sums = submatrix.sum(axis=1)
        best = float(np.min(sums))
        candidates = [
            int(member)
            for member, value in zip(members.tolist(), sums.tolist())
            if np.isclose(value, best, atol=1e-12, rtol=1e-5 if source_order_ties else 0.0)
        ]
        # Automatic selection follows the original subsampling pipeline's
        # tolerance and first-row tie rule; fixed-count runs retain ID ties.
        medoid = (
            min(candidates)
            if source_order_ties
            else min(candidates, key=lambda index: (ids[index], index))
        )
        member_distances = matrix[members, medoid].astype(np.float32)
        medoids.append(medoid)
        assignments[members] = medoid
        distances[members] = member_distances
        rows.append(
            {
                "cluster_label": cluster_label,
                "cluster_size": int(members.size),
                "medoid_id": ids[medoid],
                "medoid_index": medoid,
                "avg_intra_cluster_distance": float(member_distances.mean()),
                "max_intra_cluster_distance": float(member_distances.max()),
            }
        )
    representatives = (
        pd.DataFrame(rows)
        .sort_values(["cluster_label", "medoid_index"], kind="stable")
        .reset_index(drop=True)
    )
    return MedoidSelection(
        labels=labels_array,
        medoid_indices=np.asarray(medoids, dtype=np.int32),
        assignment_medoid_indices=assignments,
        distances_to_medoid=distances,
        representatives=representatives,
    )


def _cluster_metrics(distance_matrix: np.ndarray, medoids: MedoidSelection) -> dict[str, Any]:
    labels = medoids.labels
    n = labels.size
    n_clusters = int(np.unique(labels).size)
    mean_intra = float(np.mean(medoids.distances_to_medoid)) if n else 0.0
    metrics: dict[str, Any] = {
        "n_clusters": n_clusters,
        "sub_lib_size": n_clusters,
        "compression_ratio": float(n_clusters / n) if n else np.nan,
        "mean_intra_cluster_distance_to_medoid": mean_intra,
        "median_intra_cluster_distance_to_medoid": float(np.median(medoids.distances_to_medoid))
        if n
        else np.nan,
        "coverage_90": float(np.quantile(medoids.distances_to_medoid, 0.90)) if n else np.nan,
    }
    sizes = np.asarray(
        [int(np.sum(labels == label)) for label in sorted(np.unique(labels))], dtype=np.int32
    )
    metrics.update(
        n_singletons=int(np.sum(sizes == 1)),
        singleton_ratio=float(np.mean(sizes == 1)) if sizes.size else np.nan,
        largest_cluster_size=int(np.max(sizes)) if sizes.size else 0,
        largest_cluster_ratio=float(np.max(sizes) / n) if sizes.size and n else np.nan,
    )
    if medoids.medoid_indices.size > 1:
        medoid_distances = distance_matrix[
            np.ix_(medoids.medoid_indices, medoids.medoid_indices)
        ].copy()
        np.fill_diagonal(medoid_distances, np.inf)
        nearest = np.min(medoid_distances, axis=1)
        separation = float(np.mean(nearest))
        metrics["mean_nearest_other_medoid_distance"] = separation
        metrics["compactness_separation_ratio"] = (
            float(mean_intra / separation) if separation > 0 else np.nan
        )
    else:
        metrics["mean_nearest_other_medoid_distance"] = np.nan
        metrics["compactness_separation_ratio"] = np.nan
    return metrics


def _stable_auto_choice(
    hierarchy: HierarchicalClustering,
    sample_ids: Sequence[str],
    *,
    lower_quantile: float,
    upper_quantile: float,
    n_thresholds: int,
    min_clusters: int,
    max_clusters: int | None,
    min_ari: float,
    stability: Mapping[str, Any] | None,
    stable_regime: Mapping[str, Any] | None,
) -> tuple[int, float, tuple[Mapping[str, Any], ...]]:
    from .selection import StableRegimeConfig, identify_stable_regime
    from .stability import StabilityConfig, run_subsampling_stability_analysis

    thresholds = build_threshold_grid(
        hierarchy.merge_heights,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_thresholds=n_thresholds,
    )
    if thresholds.size == 0:
        return 0, 0.0, ()
    scans: list[dict[str, Any]] = []
    for threshold in thresholds.tolist():
        labels = cut_linkage_at_threshold(hierarchy.linkage_matrix, threshold)
        medoids = select_cluster_medoids(
            hierarchy.distance_matrix, labels, sample_ids, source_order_ties=True
        )
        metrics = _cluster_metrics(hierarchy.distance_matrix, medoids)
        scans.append({"threshold": float(threshold), **metrics})
    scan = pd.DataFrame(scans)
    eligible = scan["n_clusters"] >= int(min_clusters)
    if max_clusters is not None:
        eligible &= scan["n_clusters"] <= int(max_clusters)
    scan = scan.loc[eligible].reset_index(drop=True)
    if scan.empty:
        raise ValueError("No scanned clustering satisfies min_clusters/max_clusters")
    diagnostics = run_subsampling_stability_analysis(
        distance_matrix=hierarchy.distance_matrix,
        thresholds=scan["threshold"].to_numpy(dtype=np.float64),
        config=StabilityConfig(**dict(stability or {})),
    )
    rules = {"min_ari": min_ari, **dict(stable_regime or {})}
    selected = identify_stable_regime(scan, diagnostics.summary, StableRegimeConfig(**rules))
    return (
        selected.selected_index,
        selected.selected_threshold,
        tuple(selected.summary.to_dict(orient="records")),
    )


def cluster_embeddings(
    embeddings: np.ndarray,
    sample_ids: Sequence[str],
    *,
    selection: str = "stable_auto",
    target_clusters: int | None = None,
    optimal_ordering: bool = True,
    min_ari: float = DEFAULT_MIN_ARI,
    threshold_lower_quantile: float = 0.10,
    threshold_upper_quantile: float = 0.90,
    n_thresholds: int = 60,
    min_clusters: int = 1,
    max_clusters: int | None = None,
    stability: Mapping[str, Any] | None = None,
    stable_regime: Mapping[str, Any] | None = None,
) -> Any:
    """Cluster embeddings and choose deterministic medoids."""

    features = np.asarray(embeddings, dtype=np.float32)
    ids = [str(item) for item in sample_ids]
    if features.ndim != 2 or features.shape[0] != len(ids) or not ids:
        raise ValueError("embeddings and sample_ids must describe a non-empty matching dataset")
    distance = compute_cosine_distance_matrix(features)
    hierarchy = fit_average_linkage(distance, optimal_ordering=optimal_ordering)
    mode = str(selection).lower()
    if mode == "fixed_count":
        if target_clusters is None:
            raise ValueError("target_clusters is required when selection='fixed_count'")
        labels = cut_linkage_at_count(hierarchy.linkage_matrix, len(ids), int(target_clusters))
        representative = select_cluster_medoids(distance, labels, ids)
        threshold = (
            float(hierarchy.merge_heights[len(ids) - int(target_clusters) - 1])
            if int(target_clusters) < len(ids)
            else 0.0
        )
        scan: tuple[Mapping[str, Any], ...] = ()
    elif mode == "stable_auto":
        _selected_index, threshold, scan = _stable_auto_choice(
            hierarchy,
            ids,
            lower_quantile=threshold_lower_quantile,
            upper_quantile=threshold_upper_quantile,
            n_thresholds=n_thresholds,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            min_ari=min_ari,
            stability=stability,
            stable_regime=stable_regime,
        )
        labels = (
            cut_linkage_at_threshold(
                hierarchy.linkage_matrix, float(np.nextafter(threshold, np.inf))
            )
            if len(ids) > 1
            else np.zeros(1, dtype=np.int32)
        )
        representative = select_cluster_medoids(distance, labels, ids, source_order_ties=True)
    else:
        raise ValueError("selection must be 'fixed_count' or 'stable_auto'")
    return ClusterResult(
        labels=representative.labels,
        representative_indices=representative.medoid_indices,
        assignment_representative_indices=representative.assignment_medoid_indices,
        distances_to_representative=representative.distances_to_medoid,
        threshold=threshold,
        selection=mode,
        linkage_matrix=hierarchy.linkage_matrix,
        threshold_scan=scan,
        metrics={"min_ari": float(min_ari), **_cluster_metrics(distance, representative)},
    )


__all__ = [
    "HierarchicalClustering",
    "MedoidSelection",
    "build_threshold_grid",
    "cluster_embeddings",
    "compute_cosine_distance_matrix",
    "cut_linkage_at_count",
    "cut_linkage_at_threshold",
    "fit_average_linkage",
    "l2_normalize_embeddings",
    "select_cluster_medoids",
]

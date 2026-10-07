"""Report metrics and stable CSV/NPZ artifact writers."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .common import (
    _first,
    _float,
    _is_anomaly_phase,
    _ordered_curve,
    _read_rows,
    _truth_for_sample,
)
from .models import _ResultData, _Sample
from .phases import phase_formula_key, truth_phase_amount


def _phase_matrix(
    data: _ResultData,
    truth: Mapping[str, Mapping[str, float]],
    truth_labels: Mapping[str, str],
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray]:
    sample_ids = list(data.samples)
    phase_order: list[str] = []

    def canonical_phase(phase_id: str) -> str:
        key = phase_formula_key(data, phase_id)
        return data.phase_aliases.get(key, phase_id) if key is not None else phase_id

    data_phase_order: list[str] = []
    for phase_id in data.phase_order:
        canonical = canonical_phase(phase_id)
        if not _is_anomaly_phase(canonical) and canonical not in data_phase_order:
            data_phase_order.append(canonical)
    aligned_truth: dict[str, dict[str, float]] = {}
    aligned_truth_labels: dict[str, str] = {}
    truth_phase_order: list[str] = []
    for sample_id, sample_truth in truth.items():
        aligned_row = aligned_truth.setdefault(sample_id, {})
        for phase_id, amount in sample_truth.items():
            if _is_anomaly_phase(phase_id):
                continue
            canonical = canonical_phase(phase_id)
            aligned_row[canonical] = aligned_row.get(canonical, 0.0) + float(amount)
            aligned_truth_labels.setdefault(canonical, str(truth_labels.get(phase_id, phase_id)))
            if canonical not in truth_phase_order:
                truth_phase_order.append(canonical)
    phase_order = truth_phase_order + [
        phase for phase in data_phase_order if phase not in truth_phase_order
    ]

    predicted_rows: list[list[float]] = []
    for sample_id in sample_ids:
        sample = data.samples[sample_id]
        by_phase: dict[str, float] = {}
        # Only the phases the MILP selected; unselected ones can carry ~1e-7 solver residue.
        for phase_id in sample.selected_phase_ids:
            canonical = canonical_phase(phase_id)
            if _is_anomaly_phase(canonical):
                continue
            amount = sample.abundances.get(phase_id, 0.0)
            by_phase[canonical] = by_phase.get(canonical, 0.0) + float(amount)
        predicted_rows.append([by_phase.get(phase, 0.0) for phase in phase_order])
    predicted = np.asarray(predicted_rows, dtype=float)
    expert = np.asarray(
        [
            [float(sample_truth.get(phase, 0.0)) for phase in phase_order]
            for sample_truth in (_truth_for_sample(aligned_truth, sample) for sample in sample_ids)
        ],
        dtype=float,
    )
    labels = [
        str(aligned_truth_labels.get(phase, data.labels.get(phase, phase))) for phase in phase_order
    ]
    return sample_ids, phase_order, labels, expert, predicted


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _fmt(value: object, digits: int = 8) -> object:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.{digits}g}"
    return value


def _curve_grid(samples: Sequence[_Sample]) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    records: list[tuple[_Sample, np.ndarray, np.ndarray]] = []
    for sample in samples:
        if sample.anomaly is None or np.asarray(sample.anomaly).size == 0:
            continue
        x, curve = _ordered_curve(
            sample.x,
            np.maximum(np.asarray(sample.anomaly, dtype=float).reshape(-1), 0.0),
            context=f"sample {sample.sample_id!r} anomaly curve",
        )
        records.append((sample, x, curve))
    if not records:
        return None, {}
    prepared: dict[str, np.ndarray] = {}
    lower = -math.inf
    upper = math.inf
    first_step: float | None = None
    for _sample, x, curve in records:
        if x.size > 1:
            # The source similarity figure defines the common grid from the
            # first record's native spacing. Keep that convention
            # even when later records have a slightly different sampling step.
            if first_step is None:
                first_step = float(np.median(np.diff(x)))
        lower = max(lower, float(x[0]))
        upper = min(upper, float(x[-1]))
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("anomaly curves do not have a finite common axis")
    if upper <= lower:
        raise ValueError(
            "anomaly curves have no overlapping axis range; use the same two-theta window"
        )
    if first_step is None or first_step <= 0.0:
        raise ValueError("anomaly curves need at least two distinct axis points")
    else:
        step = first_step
        # ``np.arange`` with the half-step endpoint is the source helper's
        # exact endpoint rule and avoids silently changing the grid when
        # the overlap is not an integer number of samples.
        grid = np.arange(lower, upper + step / 2.0, step)
    for sample, x, curve in records:
        prepared[sample.sample_id] = np.interp(grid, x, curve, left=0.0, right=0.0)
    return grid, prepared


def _anomaly_rank_map(data: _ResultData) -> dict[str, int]:
    """Return stable descending anomaly ranks for the summary table."""

    scored = [
        (
            sample,
            _fmt(sample.anomaly_pct if sample.anomaly_pct is not None else 0.0),
        )
        for sample in data.samples.values()
    ]
    # Rank the formatted value, matching the historical anomaly score CSV.
    # Python's stable sort retains source sample order for equal displayed
    # scores.
    scored.sort(key=lambda item: -(_float(item[1], 0.0) or 0.0))
    return {sample.sample_id: rank for rank, (sample, _score) in enumerate(scored, 1)}


def _write_sample_summary(
    path: Path,
    data: _ResultData,
    *,
    anomaly_ranks: Mapping[str, int] | None = None,
) -> None:
    """Write the compact per-sample table for downstream analysis."""

    fields = (
        "sample_id",
        "selected_phase_ids",
        "selected_phase_labels",
        "abundance_pct",
        "anomaly_pct",
        "anomaly_rank",
        "objective",
        "solver_status",
        "warnings",
        "element_present",
        "element_missing",
        "element_violated",
    )
    rows: list[dict[str, object]] = []
    for sample in data.samples.values():
        selected = list(sample.selected_phase_ids)
        if not selected:
            selected = [
                phase_id
                for phase_id, amount in sample.abundances.items()
                if float(amount) > 0.0 and not _is_anomaly_phase(phase_id)
            ]
        # The source workflow orders recovered phases by abundance while
        # retaining input order for equal values.
        selected.sort(key=lambda phase_id: -float(sample.abundances.get(phase_id, 0.0)))
        labels = [data.labels.get(phase_id, phase_id) for phase_id in selected]
        abundance_values = [_fmt(sample.abundances.get(phase_id, 0.0)) for phase_id in selected]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "selected_phase_ids": ";".join(selected),
                "selected_phase_labels": ";".join(str(label) for label in labels),
                "abundance_pct": ";".join(str(value) for value in abundance_values),
                "anomaly_pct": _fmt(sample.anomaly_pct if sample.anomaly_pct is not None else 0.0),
                "anomaly_rank": (anomaly_ranks or {}).get(sample.sample_id, ""),
                "objective": _fmt(sample.objective),
                "solver_status": sample.solver_status,
                "warnings": sample.warnings,
                "element_present": _first(sample.summary_fields, ("element_present",)),
                "element_missing": _first(sample.summary_fields, ("element_missing",)),
                "element_violated": _first(sample.summary_fields, ("element_violated",)),
            }
        )
    _write_csv(path, fields, rows)


def _write_anomaly_outputs(
    output_root: Path,
    data: _ResultData,
    *,
    reconstruction_sigma: float | None,
) -> tuple[Path, Path, dict[str, np.ndarray]]:
    sample_ids = list(data.samples)
    grid, curves = _curve_grid(list(data.samples.values()))
    ordered_anomalies: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sample_id in sample_ids:
        sample = data.samples[sample_id]
        if sample.anomaly is None:
            continue
        ordered_anomalies[sample_id] = _ordered_curve(
            sample.x,
            np.asarray(sample.anomaly, dtype=float).reshape(-1),
            context=f"sample {sample_id!r} anomaly profile",
        )
    lengths = np.asarray(
        [ordered_anomalies[sid][1].size if sid in ordered_anomalies else 0 for sid in sample_ids],
        dtype=int,
    )
    max_len = int(max(lengths, default=0))

    def padded(attribute: str) -> np.ndarray:
        matrix = np.full((len(sample_ids), max_len), np.nan, dtype=float)
        for idx, sid in enumerate(sample_ids):
            value = getattr(data.samples[sid], attribute)
            if value is None:
                continue
            row = np.asarray(value, dtype=float).reshape(-1)
            sample_x = data.samples[sid].x
            if sample_x is not None and np.asarray(sample_x).size == row.size:
                _ordered_x, row = _ordered_curve(
                    sample_x,
                    row,
                    context=f"sample {sid!r} {attribute} profile",
                )
            if attribute == "anomaly":
                row = np.maximum(row, 0.0)
            matrix[idx, : min(row.size, max_len)] = row[:max_len]
        return matrix

    profile_path = output_root / "anomaly_profiles.npz"
    x_values = np.full((len(sample_ids), max_len), np.nan, dtype=float)
    for idx, sid in enumerate(sample_ids):
        sample = data.samples[sid]
        ordered = ordered_anomalies.get(sid)
        if ordered is None:
            continue
        x, _curve = ordered
        curve_len = min(lengths[idx], max_len)
        x_values[idx, :curve_len] = x[:curve_len]
    np.savez_compressed(
        profile_path,
        sample_ids=np.asarray(sample_ids),
        x=x_values,
        lengths=lengths,
        reconstruction_sigma=np.asarray(
            np.nan if reconstruction_sigma is None else reconstruction_sigma
        ),
        similarity_grid=np.asarray([] if grid is None else grid),
        mixture=padded("mixture"),
        known_reconstruction=padded("known"),
        anomaly_reconstruction=padded("anomaly"),
        total_reconstruction=padded("total"),
    )

    similarity_path = output_root / "anomaly_similarity.csv"
    if grid is None or not curves:
        _write_csv(similarity_path, ("sample_id_i", "sample_id_j", "cosine_similarity"), [])
        return (
            profile_path,
            similarity_path,
            {
                "grid": np.asarray([]),
                "similarity": np.zeros((0, 0)),
            },
        )
    vectors = np.vstack([curves.get(sid, np.zeros_like(grid)) for sid in sample_ids])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    similarity = np.clip(unit @ unit.T, 0.0, 1.0)
    rows = [
        {
            "sample_id_i": sample_ids[i],
            "sample_id_j": sample_ids[j],
            # ``str(float(...))`` is round-trip safe for the binary64 cosine
            # matrix and lets the CSV replace the former NPZ matrix without
            # losing precision.  The old eight-digit display remains
            # reproducible by formatting the loaded value at presentation.
            "cosine_similarity": float(similarity[i, j]),
        }
        for i in range(len(sample_ids))
        for j in range(len(sample_ids))
    ]
    _write_csv(similarity_path, ("sample_id_i", "sample_id_j", "cosine_similarity"), rows)
    return profile_path, similarity_path, {"grid": grid, "similarity": similarity}


def _write_anomaly_ranking_metrics(
    output_root: Path,
    data: _ResultData,
    truth: Mapping[str, Mapping[str, float]],
    truth_labels: Mapping[str, str],
    summary_path: Path,
    *,
    holdouts: Sequence[str],
    threshold_pct: float,
) -> dict[str, Path]:
    """Write anomaly-discovery ranking tables when a holdout is available."""

    if not holdouts or not truth:
        return {}
    summary_rows = _read_rows(summary_path)
    if not summary_rows:
        return {}
    # ``sample_summary.csv`` retains source sample order.  Sorting by the
    # exported rank reproduces the former score table order and remains stable
    # for equal displayed anomaly scores.
    ranked_rows = list(enumerate(summary_rows))
    ranked_rows.sort(
        key=lambda item: (
            _float(_first(item[1], ("anomaly_rank",)), float("inf"))
            if _float(_first(item[1], ("anomaly_rank",))) is not None
            else float("inf"),
            item[0],
        )
    )
    score_rows = [row for _source_index, row in ranked_rows]
    output: dict[str, Path] = {}
    metrics_rows: list[dict[str, object]] = []
    labels_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for holdout in holdouts:
        ranked: list[tuple[str, float, bool, int, float]] = []
        for rank, row in enumerate(score_rows, start=1):
            sample_id = _first(row, ("sample_id", "mixture_name"))
            score = _float(_first(row, ("anomaly_pct", "anomalous_area_pct")), 0.0) or 0.0
            truth_amount = truth_phase_amount(_truth_for_sample(truth, sample_id), holdout, data)
            positive = truth_amount > threshold_pct
            ranked.append((sample_id, float(score), positive, rank, float(truth_amount)))
        positive_count = sum(1 for _sample, _score, positive, _rank, _truth in ranked if positive)
        top_k = positive_count
        top_k_hits = sum(
            1 for _sample, _score, positive, _rank, _truth in ranked[:top_k] if positive
        )
        cumulative = 0
        precision_sum = 0.0
        for index, (sample_id, score, positive, rank, truth_amount) in enumerate(ranked, start=1):
            if positive:
                cumulative += 1
                precision_sum += cumulative / float(index)
            labels_rows.append(
                {
                    "holdout_phase_id": holdout,
                    "mixture_name": sample_id,
                    "sample_id": sample_id,
                    "anomaly_rank": rank,
                    "anomalous_area_pct": _fmt(score),
                    "anomaly_pct": _fmt(score),
                    "gt_phase_pct": _fmt(truth_amount),
                    "gt_positive": int(positive),
                    "match_type": (
                        "matched_positive"
                        if positive and rank <= top_k
                        else "missed_positive"
                        if positive
                        else "false_positive"
                        if rank <= top_k
                        else "true_negative"
                    ),
                }
            )
        for k, (_sample, _score, positive, _rank, _truth) in enumerate(ranked[:top_k], start=1):
            hits = sum(1 for item in ranked[:k] if item[2])
            curve_rows.append(
                {
                    "holdout_phase_id": holdout,
                    "omitted_phase": holdout,
                    "k": k,
                    "mixture_name_at_k": _sample,
                    "gt_positive_at_k": int(positive),
                    "cumulative_hits": hits,
                    "precision_at_k": _fmt(hits / float(k)),
                }
            )
        ap = precision_sum / positive_count if positive_count else None
        # Ranking and Pearson use the same GT presence mask.
        positive_items = [item for item in ranked if item[2]]
        positive_truth = np.asarray([item[4] for item in positive_items], dtype=float)
        positive_scores = np.asarray([item[1] for item in positive_items], dtype=float)
        pearson = None
        if (
            positive_truth.size >= 2
            and float(np.std(positive_truth)) > 0.0
            and float(np.std(positive_scores)) > 0.0
        ):
            pearson = float(np.corrcoef(positive_truth, positive_scores)[0, 1])
        metrics_rows.append(
            {
                "holdout_phase_id": holdout,
                "omitted_phase": holdout,
                "phase_name": truth_labels.get(holdout, data.labels.get(holdout, holdout)),
                "samples": len(ranked),
                "gt_positive_count": positive_count,
                "gt_negative_count": len(ranked) - positive_count,
                "top_k": top_k,
                "top_k_match_count": top_k_hits,
                "recall_at_k": _fmt(top_k_hits / float(positive_count)) if positive_count else "",
                "top_k_precision": _fmt(top_k_hits / float(positive_count))
                if positive_count
                else "",
                "average_precision": _fmt(ap),
                "exact_top_k_set_match": str(top_k_hits == positive_count).lower(),
                "gt_anomaly_pearson_r": _fmt(pearson),
            }
        )

    labels_path = output_root / "anomaly_ranking_labels.csv"
    _write_csv(
        labels_path,
        (
            "holdout_phase_id",
            "mixture_name",
            "sample_id",
            "anomaly_rank",
            "anomalous_area_pct",
            "anomaly_pct",
            "gt_phase_pct",
            "gt_positive",
            "match_type",
        ),
        labels_rows,
    )
    metrics_path = output_root / "anomaly_ranking_metrics.csv"
    _write_csv(
        metrics_path,
        (
            "holdout_phase_id",
            "omitted_phase",
            "phase_name",
            "samples",
            "gt_positive_count",
            "gt_negative_count",
            "top_k",
            "top_k_match_count",
            "recall_at_k",
            "top_k_precision",
            "average_precision",
            "exact_top_k_set_match",
            "gt_anomaly_pearson_r",
        ),
        metrics_rows,
    )
    curve_path = output_root / "anomaly_precision_at_k.csv"
    _write_csv(
        curve_path,
        (
            "holdout_phase_id",
            "omitted_phase",
            "k",
            "mixture_name_at_k",
            "gt_positive_at_k",
            "cumulative_hits",
            "precision_at_k",
        ),
        curve_rows,
    )
    output.update(
        {
            "anomaly_ranking_labels": labels_path,
            "anomaly_ranking_metrics": metrics_path,
            "anomaly_precision_at_k": curve_path,
        }
    )
    return output


def _write_reference_similarity(
    output_root: Path,
    data: _ResultData,
    grid: np.ndarray,
    references: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> Path | None:
    if not references:
        return None
    rows: list[dict[str, object]] = []
    records = [sample for sample in data.samples.values() if sample.anomaly is not None]
    for sample in records:
        x, curve = _ordered_curve(
            sample.x,
            np.maximum(np.asarray(sample.anomaly, dtype=float).reshape(-1), 0.0),
            context=f"sample {sample.sample_id!r} anomaly reference curve",
        )
        vector = np.interp(grid, x, curve, left=0.0, right=0.0)
        norm = np.linalg.norm(vector)
        vector = vector / norm if norm > 0 else np.zeros_like(vector)
        for reference_id, (ref_x, ref_y) in references.items():
            ref_x, ref_y = _ordered_curve(
                ref_x,
                np.maximum(np.asarray(ref_y, dtype=float).reshape(-1), 0.0),
                context=f"reference {reference_id!r}",
            )
            ref = np.interp(
                grid,
                ref_x,
                ref_y,
                left=0.0,
                right=0.0,
            )
            ref_norm = np.linalg.norm(ref)
            ref = ref / ref_norm if ref_norm > 0 else np.zeros_like(ref)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "reference_id": reference_id,
                    "cosine_similarity": _fmt(float(np.clip(vector @ ref, 0.0, 1.0))),
                }
            )
    path = output_root / "anomaly_reference_similarity.csv"
    _write_csv(path, ("sample_id", "reference_id", "cosine_similarity"), rows)
    return path

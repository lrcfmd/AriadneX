"""Static plots for unmixing and anomaly reports."""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ..artifacts import ensure_unique_sanitized_ids
from .common import _float, _ordered_curve, _read_rows, _safe_report_id, _truth_for_sample
from .models import (
    _ANOMALY_COLOR,
    _BASELINE_FILL_COLOR,
    _BASELINE_LINE_COLOR,
    _KNOWN_COLOR,
    _OBSERVED_COLOR,
    _PHASE_COLORS,
    _RASTER_DPI,
    _REFERENCE_COLOR,
    _ResultData,
    _Sample,
)
from .phases import truth_phase_amount


def _save_phase_matrix(
    output_png: Path,
    sample_ids: Sequence[str],
    labels: Sequence[str],
    expert: np.ndarray,
    predicted: np.ndarray,
    *,
    truth_available: bool,
) -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 7,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )
    phase_colors = (
        "#B7791F",
        "#009E73",
        "#CC79A7",
        "#7B6A9E",
        "#0072B2",
        "#D55E00",
        "#56B4E9",
        "#E69F00",
        "#999999",
        "#2B2926",
    )
    phase_count = len(labels)
    sample_count = len(sample_ids)
    rows_per_phase = 2 if truth_available else 1
    display_rows = phase_count * rows_per_phase + max(phase_count - 1, 0)
    if display_rows == 0:
        display_rows = 1
    display = np.full((display_rows, max(sample_count, 1)), np.nan, dtype=float)
    phase_row_starts: list[int] = []
    row = 0
    for phase_index in range(phase_count):
        phase_row_starts.append(row)
        if truth_available:
            display[row] = np.asarray(expert[:, phase_index], dtype=float) / 100.0
            display[row + 1] = np.asarray(predicted[:, phase_index], dtype=float) / 100.0
            row += 2
        else:
            display[row] = np.asarray(predicted[:, phase_index], dtype=float) / 100.0
            row += 1
        if phase_index + 1 < phase_count:
            row += 1

    fig_width = max(6.0, min(12.5, 0.27 * max(sample_count, 1) + 3.6))
    fig_height = max(2.8, min(12.0, 0.29 * display_rows + 1.55))
    fig = Figure(figsize=(fig_width, fig_height))
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(2.9, max(float(sample_count), 8.0)),
        left=0.025,
        right=0.985,
        top=0.90,
        bottom=max(0.12, min(0.28, 1.0 / fig_height)),
        wspace=0.025,
    )
    label_ax = fig.add_subplot(grid[0, 0])
    heat_ax = fig.add_subplot(grid[0, 1])

    if phase_count:
        for phase_index, start in enumerate(phase_row_starts):
            phase_display = np.full_like(display, np.nan)
            phase_display[start : start + rows_per_phase] = display[start : start + rows_per_phase]
            color = phase_colors[phase_index % len(phase_colors)]
            cmap = LinearSegmentedColormap.from_list(
                f"phase_{phase_index}",
                ["#FAFAFA", color],
            )
            cmap.set_bad((1.0, 1.0, 1.0, 0.0))
            heat_ax.imshow(
                np.ma.masked_invalid(phase_display),
                aspect="auto",
                interpolation="none",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
            )
    else:
        heat_ax.imshow(
            np.zeros((1, max(sample_count, 1))), cmap="Greys", vmin=0.0, vmax=1.0, aspect="auto"
        )

    heat_ax.set_xlim(-0.5, max(sample_count, 1) - 0.5)
    heat_ax.set_ylim(display_rows - 0.5, -0.5)
    heat_ax.set_xticks(np.arange(sample_count))
    heat_ax.set_xticklabels([str(item) for item in sample_ids], rotation=90, fontsize=5.8)
    heat_ax.set_yticks([])
    heat_ax.set_xticks(np.arange(-0.5, max(sample_count, 1), 1), minor=True)
    heat_ax.set_yticks(np.arange(-0.5, display_rows, 1), minor=True)
    heat_ax.grid(which="minor", color="#FFFFFF", linewidth=0.42)
    heat_ax.tick_params(axis="both", which="both", length=0, pad=1.5)
    for spine in heat_ax.spines.values():
        spine.set_visible(False)

    label_ax.set_xlim(0.0, 1.0)
    label_ax.set_ylim(heat_ax.get_ylim())
    label_ax.axis("off")
    if phase_count:
        for phase_index, (label, start) in enumerate(zip(labels, phase_row_starts)):
            color = phase_colors[phase_index % len(phase_colors)]
            center = start + (rows_per_phase - 1) / 2.0
            label_ax.plot(
                [0.02, 0.10], [center, center], color=color, linewidth=2.2, solid_capstyle="butt"
            )
            label_ax.text(
                0.13,
                center,
                str(label),
                ha="left",
                va="center",
                fontsize=6.4,
                fontweight="semibold",
                color="#2B2926",
            )
            if truth_available:
                label_ax.text(
                    0.98, start, "expert", ha="right", va="center", fontsize=6.0, color="#746F69"
                )
                label_ax.text(
                    0.98, start + 1, "model", ha="right", va="center", fontsize=6.0, color="#746F69"
                )
            else:
                label_ax.text(
                    0.98, start, "model", ha="right", va="center", fontsize=6.0, color="#746F69"
                )
    else:
        label_ax.text(
            0.98, 0.0, "no recovered phases", ha="right", va="center", fontsize=6.4, color="#746F69"
        )

    title = "Known-phase recovery" if truth_available else "Recovered phase abundance"
    fig.text(
        0.025,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="semibold",
        color="#2B2926",
    )
    fig.text(
        0.985,
        0.965,
        "Colour intensity encodes abundance (%)",
        ha="right",
        va="top",
        fontsize=6.2,
        color="#746F69",
    )
    heat_ax.set_xlabel("Experimental sample", fontsize=7.0, labelpad=4.0)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=_RASTER_DPI, bbox_inches="tight")
    fig.clear()


def _max_normalise(curve: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(curve, dtype=float).reshape(-1), 0.0)
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0.0 else values


def _sample_phase_entries(
    sample: _Sample,
    data: _ResultData,
) -> list[tuple[str, str, float, np.ndarray]]:
    """Return selected phase curves with their stable global phase IDs."""

    rows: list[tuple[str, str, float, np.ndarray]] = []
    selected_phase_ids = set(sample.selected_phase_ids)
    for phase_id in data.phase_order:
        if phase_id not in selected_phase_ids:
            continue
        curve = sample.phase_fingerprints.get(phase_id)
        if curve is None:
            continue
        amount = max(float(sample.abundances.get(phase_id, 0.0)), 0.0)
        rows.append((phase_id, data.labels.get(phase_id, phase_id), amount, curve))
    return rows


def _format_phase_formula(label: object) -> str:
    """Render chemical subscripts while leaving source IDs readable."""

    text = str(label or "").strip()
    if not text or "_" in text or text.lower().startswith("icsd"):
        return text
    pieces: list[str] = []
    position = 0
    while position < len(text):
        character = text[position]
        if character in "([":
            pieces.append(character)
            position += 1
            continue
        if character in ")]":
            pieces.append(character)
            position += 1
            match = re.match(r"\d+(?:\.\d+)?", text[position:])
            if match is not None:
                pieces.append(f"$_{{{match.group(0)}}}$")
                position += match.end()
            continue
        match = re.match(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?", text[position:])
        if match is None:
            return text
        pieces.append(match.group(1))
        if match.group(2):
            pieces.append(f"$_{{{match.group(2)}}}$")
        position += match.end()
    return "".join(pieces)


def _phase_color_map(data: _ResultData) -> dict[str, str]:
    """Assign colors once from the complete phase order, never per sample."""

    formula_colors = {
        "caco3": 0,
        "zno": 1,
        "tio2": 2,
        "sn1.8o3.4": 3,
        "sno2": 3,
    }
    color_map: dict[str, str] = {}
    for index, phase_id in enumerate(data.phase_order):
        key = str(phase_id).strip().lower()
        if len(key) == 1 and key in "abcd":
            color_index = ord(key) - ord("a")
        else:
            formula = re.sub(r"[^a-z0-9.]", "", str(data.labels.get(phase_id, phase_id)).lower())
            color_index = formula_colors.get(formula, index)
        color_map[phase_id] = _PHASE_COLORS[color_index % len(_PHASE_COLORS)]
    return color_map


def _reference_curve(
    reference: tuple[np.ndarray, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    ref_x, ref_y = _ordered_curve(
        reference[0],
        np.maximum(np.asarray(reference[1], dtype=float).reshape(-1), 0.0),
        context="reference curve",
    )
    if ref_x.size != ref_y.size or ref_x.size < 2:
        return np.zeros_like(x)
    return np.interp(x, ref_x, ref_y, left=0.0, right=0.0)


def _paper_mixture_display(
    sample: _Sample,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mixture = np.zeros_like(x)
    if sample.mixture is not None:
        _mixture_x, mixture = _ordered_curve(
            sample.x,
            np.maximum(np.asarray(sample.mixture, dtype=float).reshape(-1), 0.0),
            context="mixture curve",
        )
        if _mixture_x.size != x.size or not np.array_equal(_mixture_x, x):
            mixture = np.interp(x, _mixture_x, mixture, left=0.0, right=0.0)
    raw = mixture
    baseline = np.zeros_like(x)
    if sample.raw_x is not None and sample.raw_y is not None:
        raw_x, raw_y = _ordered_curve(
            sample.raw_x,
            np.maximum(np.asarray(sample.raw_y, dtype=float).reshape(-1), 0.0),
            context="raw mixture curve",
        )
        if raw_x.size == raw_y.size and raw_x.size > 1:
            raw = np.interp(x, raw_x, raw_y)
    if sample.baseline is not None:
        baseline_x, candidate = _ordered_curve(
            sample.x,
            np.maximum(np.asarray(sample.baseline, dtype=float).reshape(-1), 0.0),
            context="baseline curve",
        )
        if baseline_x.size == x.size and np.array_equal(baseline_x, x):
            baseline = np.minimum(candidate, raw)
        elif baseline_x.size > 1:
            baseline = np.minimum(np.interp(x, baseline_x, candidate), raw)
    clean = np.maximum(raw - baseline, 0.0)
    known = np.zeros_like(x)
    if sample.known is not None:
        known_x, known = _ordered_curve(
            sample.x,
            np.maximum(np.asarray(sample.known, dtype=float).reshape(-1), 0.0),
            context="known reconstruction curve",
        )
        if known_x.size != x.size or not np.array_equal(known_x, x):
            known = np.interp(x, known_x, known, left=0.0, right=0.0)
    anomaly = np.zeros_like(x)
    if sample.anomaly is not None:
        anomaly_x, anomaly = _ordered_curve(
            sample.x,
            np.maximum(np.asarray(sample.anomaly, dtype=float).reshape(-1), 0.0),
            context="anomaly reconstruction curve",
        )
        if anomaly_x.size != x.size or not np.array_equal(anomaly_x, x):
            anomaly = np.interp(x, anomaly_x, anomaly, left=0.0, right=0.0)
    total = known + anomaly
    known_fraction = np.divide(known, total, out=np.zeros_like(total), where=total > 0.0)
    anomaly_fraction = np.divide(anomaly, total, out=np.zeros_like(total), where=total > 0.0)
    scale = max(float(raw.max()), 1e-30)
    return (
        np.sqrt(raw / scale),
        np.sqrt(baseline / scale),
        np.sqrt((baseline + clean * known_fraction) / scale),
        np.sqrt((baseline + clean * (known_fraction + anomaly_fraction)) / scale),
    )


def _style_unmix_axis(ax: Any, *, show_bottom: bool) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(show_bottom)
    ax.spines["bottom"].set_linewidth(0.45)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, labelsize=6.5, labelbottom=show_bottom, pad=1.5)


def _draw_fingerprint_lane(
    ax: Any,
    x: np.ndarray,
    curve: np.ndarray,
    label: str,
    color: str,
    *,
    fill_alpha: float = 0.10,
) -> None:
    display = _max_normalise(curve)
    ax.fill_between(x, 0.0, display, color=color, alpha=fill_alpha, linewidth=0.0)
    ax.plot(x, display, color=color, linewidth=0.58)
    ax.set_ylim(-0.02, 1.12)
    ax.text(
        0.988,
        0.78,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=color,
        bbox={
            "boxstyle": "square,pad=0.08",
            "facecolor": matplotlib.colors.to_rgba(color, 0.07),
            "edgecolor": "none",
        },
    )


def _save_sample_reconstructions(
    output_dir: Path,
    data: _ResultData,
    *,
    references: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    truth: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Path]:
    """Write compact Figure 2c-style unmix plots without changing curves."""

    ensure_unique_sanitized_ids(
        data.samples,
        _safe_report_id,
        context="sample reconstruction output",
    )
    if output_dir.is_symlink():
        output_dir.unlink()
    elif output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_colors = _phase_color_map(data)
    files: dict[str, Path] = {}
    for sample in data.samples.values():
        safe_id = _safe_report_id(sample.sample_id)
        path = output_dir / f"{safe_id}.png"
        available = next(
            (
                np.asarray(curve, dtype=float).reshape(-1)
                for curve in (sample.mixture, sample.total, sample.known, sample.anomaly)
                if curve is not None
            ),
            np.asarray([], dtype=float),
        )
        x, _available = _ordered_curve(
            sample.x,
            available,
            context=f"sample {sample.sample_id!r} report curve",
        )
        phase_entries = _sample_phase_entries(sample, data)
        reference_entries: list[tuple[str, str, np.ndarray]] = []
        sample_truth = _truth_for_sample(truth, sample.sample_id) if truth else {}
        for reference_id, reference in (references or {}).items():
            curve = _reference_curve(reference, x)
            if not np.any(curve > 0.0):
                continue
            label = _format_phase_formula(data.labels.get(reference_id, reference_id))
            amount = truth_phase_amount(sample_truth, reference_id, data)
            suffix = f"  {float(amount):.1f}%" if amount > 0.0 else ""
            reference_entries.append((reference_id, f"{label} reference{suffix}", curve))

        lower_count = len(phase_entries) + 1 + len(reference_entries)
        figure_height = max(3.35, min(7.2, 2.0 + 0.40 * lower_count))
        fig = Figure(figsize=(3.65, figure_height))
        FigureCanvasAgg(fig)
        grid = fig.add_gridspec(
            lower_count + 1,
            1,
            height_ratios=(1.35, *([0.52] * lower_count)),
            left=0.095,
            right=0.965,
            top=0.91,
            bottom=0.105,
            hspace=0.055,
        )
        axes = [fig.add_subplot(grid[index, 0]) for index in range(lower_count + 1)]
        fit_ax = axes[0]
        raw_display, baseline_display, known_top, anomaly_top = _paper_mixture_display(sample, x)
        fit_ax.fill_between(
            x,
            0.0,
            baseline_display,
            color=_BASELINE_FILL_COLOR,
            alpha=0.18,
            linewidth=0.0,
        )
        fit_ax.fill_between(
            x,
            baseline_display,
            known_top,
            color=_KNOWN_COLOR,
            alpha=0.56,
            linewidth=0.0,
        )
        fit_ax.fill_between(
            x,
            known_top,
            anomaly_top,
            color=_ANOMALY_COLOR,
            alpha=0.74,
            linewidth=0.0,
        )
        observed_line = fit_ax.plot(
            x,
            raw_display,
            color=_OBSERVED_COLOR,
            linewidth=0.46,
            label="Observed",
            zorder=5,
        )[0]
        fit_ax.plot(
            x,
            baseline_display,
            color=_BASELINE_LINE_COLOR,
            linewidth=0.30,
            alpha=0.50,
        )
        fit_ax.set_ylim(0.0, max(1.05, float(raw_display.max()) * 1.06))
        fit_ax.legend(
            handles=(
                observed_line,
                Patch(facecolor=_KNOWN_COLOR, edgecolor="none", label="Known basis"),
                Patch(facecolor=_ANOMALY_COLOR, edgecolor="none", label="Anomaly"),
            ),
            loc="upper right",
            ncol=3,
            frameon=False,
            fontsize=5.8,
            handlelength=0.9,
            handletextpad=0.25,
            columnspacing=0.55,
            borderaxespad=0.0,
        )
        _style_unmix_axis(fit_ax, show_bottom=False)

        axis_index = 1
        for phase_id, label, amount, curve in phase_entries:
            ax = axes[axis_index]
            _phase_x, curve = _ordered_curve(
                sample.x,
                curve,
                context=f"sample {sample.sample_id!r} phase {phase_id!r} curve",
            )
            if _phase_x.size != x.size or not np.array_equal(_phase_x, x):
                curve = np.interp(x, _phase_x, curve, left=0.0, right=0.0)
            _draw_fingerprint_lane(
                ax,
                x,
                curve,
                f"{_format_phase_formula(label)}  {amount:.1f}%",
                phase_colors[phase_id],
            )
            _style_unmix_axis(ax, show_bottom=False)
            axis_index += 1

        anomaly_ax = axes[axis_index]
        anomaly_curve = (
            sample.anomaly_fingerprint
            if sample.anomaly_fingerprint is not None
            else np.zeros_like(x)
        )
        _anomaly_x, anomaly_curve = _ordered_curve(
            sample.x,
            anomaly_curve,
            context=f"sample {sample.sample_id!r} anomaly fingerprint",
        )
        if _anomaly_x.size != x.size or not np.array_equal(_anomaly_x, x):
            anomaly_curve = np.interp(x, _anomaly_x, anomaly_curve, left=0.0, right=0.0)
        anomaly_amount = 0.0 if sample.anomaly_pct is None else float(sample.anomaly_pct)
        _draw_fingerprint_lane(
            anomaly_ax,
            x,
            anomaly_curve,
            f"Anomaly pattern  {anomaly_amount:.1f}%",
            _ANOMALY_COLOR,
        )
        _style_unmix_axis(anomaly_ax, show_bottom=not reference_entries)
        axis_index += 1

        for index, (_reference_id, label, curve) in enumerate(reference_entries):
            reference_ax = axes[axis_index + index]
            _draw_fingerprint_lane(
                reference_ax,
                x,
                curve,
                label,
                _REFERENCE_COLOR,
                fill_alpha=0.08,
            )
            _style_unmix_axis(reference_ax, show_bottom=index == len(reference_entries) - 1)
        bottom_ax = axes[-1]
        bottom_ax.set_xlabel("2θ (deg.)", fontsize=6.5, labelpad=1.5)
        if x.size:
            tick_start = math.ceil(float(x.min()) / 5.0) * 5.0
            tick_stop = math.floor(float(x.max()) / 5.0) * 5.0
            ticks = np.arange(tick_start, tick_stop + 0.1, 5.0)
            for ax in axes:
                ax.set_xlim(float(x.min()), float(x.max()))
                ax.set_xticks(ticks)

        fig.text(
            0.095,
            0.965,
            f"{sample.sample_id} unmixing",
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="semibold",
            color="#2B2926",
        )
        fig.savefig(path, dpi=_RASTER_DPI, bbox_inches="tight")
        fig.clear()
        files[f"sample_reconstruction/{safe_id}"] = path
    return files


def _save_anomaly_summary(path_png: Path, summary_path: Path) -> None:
    rows = _read_rows(summary_path)
    ranked_rows = list(enumerate(rows))
    ranked_rows.sort(
        key=lambda item: (
            _float(item[1].get("anomalyrank"), float("inf"))
            if _float(item[1].get("anomalyrank")) is not None
            else float("inf"),
            item[0],
        )
    )
    ranking_rows = [row for _source_index, row in ranked_rows[:15]]
    labels = [row.get("sampleid", "") for row in ranking_rows]
    scores = [float(row.get("anomalypct", "0") or 0.0) for row in ranking_rows]
    fig = Figure(figsize=(6.2, max(3.0, 1.1 + 0.24 * len(ranking_rows))))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    positions = np.arange(len(ranking_rows))
    if ranking_rows:
        ax.barh(positions, scores, color=_ANOMALY_COLOR, alpha=0.84, height=0.68)
        ax.set_yticks(
            positions,
            [f"{index + 1}   {label}" for index, label in enumerate(labels)],
            fontsize=7,
        )
        ax.invert_yaxis()
        for position, score in zip(positions, scores):
            ax.text(
                score + 1.0, position, f"{score:.1f}%", va="center", fontsize=7, color="#595550"
            )
    ax.set_xlim(0.0, max(100.0, max(scores, default=0.0) + 8.0))
    ax.set_xlabel("Anomalous area (%)", fontsize=8)
    ax.set_title("Anomaly ranking", loc="left", fontsize=10, fontweight="semibold")
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=_RASTER_DPI, bbox_inches="tight")
    fig.clear()

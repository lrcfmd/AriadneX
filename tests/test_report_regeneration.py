"""Regression coverage for regenerating reports in an existing output directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import pytest

from ariadnex.config import ProjectConfig
from ariadnex.contracts import UnmixArtifact
from ariadnex.reports import render_reports
from ariadnex.reports.metrics import _phase_matrix
from ariadnex.reports.models import _ResultData, _Sample
from ariadnex.reports.plotting import _save_sample_reconstructions


def _completed_result(run_root: Path) -> UnmixArtifact:
    result_root = run_root / "03_unmix"
    result_root.mkdir(parents=True)

    summary_path = result_root / "sample_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id",))
        writer.writerow(("sample-1",))

    np.savez(
        result_root / "results.npz",
        sample_ids=np.asarray(["sample-1"]),
        anomalies=np.asarray([[0.0, 0.1, 0.8, 0.1, 0.0]]),
    )

    return UnmixArtifact(
        root=result_root,
        results_pickle=result_root / "results.pkl",
        results_npz=result_root / "results.npz",
        assignments_csv=result_root / "assignments.csv",
        abundance_csv=result_root / "abundance.csv",
        sample_summary_csv=summary_path,
        metadata_path=result_root / "metadata.json",
    )


@pytest.mark.parametrize("phase_recovery_enabled", [True, False], ids=["enabled", "disabled"])
def test_report_regeneration_removes_retired_phase_metrics_but_preserves_user_files(
    tmp_path: Path, phase_recovery_enabled: bool
) -> None:
    run_root = tmp_path / "run"
    result = _completed_result(run_root)
    output_root = tmp_path / "reports"
    output_root.mkdir()
    retired_metrics = output_root / "phase_recovery_metrics.csv"
    retired_metrics.write_text("stale report output\n", encoding="utf-8")
    user_file = output_root / "notes.txt"
    user_file.write_text("keep this file\n", encoding="utf-8")

    config = ProjectConfig.from_mapping(
        {"reports": {"anomaly": False, "phase_recovery": phase_recovery_enabled}},
        base_dir=tmp_path,
    )
    report = render_reports(config, result, output_root)

    assert not retired_metrics.exists()
    assert user_file.read_text(encoding="utf-8") == "keep this file\n"
    assert (output_root / "sample_summary.csv").is_file()
    assert (output_root / "index.html").is_file()
    assert report.metadata_path == output_root / "report_manifest.json"
    manifest = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    assert manifest["phase_recovery"] is phase_recovery_enabled
    assert manifest["files"]["index"] == "index.html"
    assert (output_root / "phase_recovery_matrix.png").is_file() is phase_recovery_enabled


def test_sample_reconstruction_displays_only_milp_selected_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = _Sample(
        sample_id="sample-1",
        abundances={"selected": 79.4, "solver-residue": 3.287441e-5},
        selected_phase_ids=["selected"],
        x=np.asarray([10.0, 11.0, 12.0]),
        mixture=np.asarray([1.0, 2.0, 1.0]),
        known=np.asarray([0.8, 1.6, 0.8]),
        anomaly=np.asarray([0.2, 0.4, 0.2]),
        total=np.asarray([1.0, 2.0, 1.0]),
        phase_fingerprints={
            "selected": np.asarray([0.2, 1.0, 0.2]),
            "solver-residue": np.asarray([1.0, 0.2, 1.0]),
        },
        anomaly_fingerprint=np.asarray([0.0, 0.1, 0.0]),
    )
    data = _ResultData(
        samples={sample.sample_id: sample},
        phase_order=["selected", "solver-residue"],
        labels={"selected": "Selected_ID", "solver-residue": "Residual_ID"},
    )

    _sample_ids, phase_ids, _labels, _expert, predicted = _phase_matrix(data, {}, {})
    assert predicted[0, phase_ids.index("selected")] == 79.4
    assert predicted[0, phase_ids.index("solver-residue")] == 0.0

    captured_phase_labels: list[str] = []

    def capture_figure(figure: Figure, _path: Path, **_kwargs: object) -> None:
        captured_phase_labels.extend(
            text.get_text() for axis in figure.axes for text in axis.texts
        )

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    _save_sample_reconstructions(tmp_path / "sample_reconstructions", data)

    assert "Selected_ID  79.4%" in captured_phase_labels
    assert not any(label.startswith("Residual_ID") for label in captured_phase_labels)

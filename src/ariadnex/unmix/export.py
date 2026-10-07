"""Persistence of unmix results.

The writer keeps the existing path contract and file schemas. It is isolated
from input loading and solving so output formatting can evolve independently.
"""

from __future__ import annotations

import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _write_outputs(
    root: Path,
    library: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    solved: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any],
    metadata_path: Path,
):
    phase_ids = [str(item["phase_id"]) for item in library]
    sample_ids = [str(item["sample_id"]) for item in samples]
    basis_positions = np.asarray(
        [np.asarray(item["signature"][0], dtype=float) for item in library],
        dtype=object,
    )
    basis_weights = np.asarray(
        [np.asarray(item["signature"][1], dtype=float) for item in library],
        dtype=object,
    )
    basis_x = np.asarray(
        [np.asarray(item["clean_x"], dtype=float) for item in library],
        dtype=object,
    )
    basis_y = np.asarray(
        [np.asarray(item["clean_y"], dtype=float) for item in library],
        dtype=object,
    )
    abundance = np.asarray(
        [
            [
                float(np.sum(result["components"][phase_index]))
                for phase_index in range(len(library))
            ]
            for result in solved
        ],
        dtype=float,
    )
    anomaly_profiles = np.asarray(
        [np.asarray(result["anomaly"], dtype=float) for result in solved], dtype=object
    )
    components = np.asarray(
        [
            [np.asarray(component, dtype=float) for component in result["components"]]
            for result in solved
        ],
        dtype=object,
    )
    costs = np.asarray([float(result["objective"]) for result in solved], dtype=float)
    results_payload = {
        "schema_version": 1,
        "sample_ids": sample_ids,
        "phase_ids": phase_ids,
        "basis_positions": basis_positions,
        "basis_weights": basis_weights,
        "basis_x": basis_x,
        "basis_y": basis_y,
        "components": components,
        "anomalies": anomaly_profiles,
        "abundance": abundance,
        "objectives": costs,
        "records": list(solved),
        "options": dict(options),
    }
    results_pickle = root / "results.pkl"
    results_npz = root / "results.npz"
    with results_pickle.open("wb") as handle:
        pickle.dump(results_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(
        results_npz,
        sample_ids=np.asarray(sample_ids, dtype=str),
        phase_ids=np.asarray(phase_ids, dtype=str),
        basis_positions=basis_positions,
        basis_weights=basis_weights,
        basis_x=basis_x,
        basis_y=basis_y,
        components=components,
        anomalies=anomaly_profiles,
        abundance=abundance,
        objectives=costs,
    )
    assignments_csv = root / "assignments.csv"
    with assignments_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "phase_id", "abundance", "selected", "anomaly"]
        )
        writer.writeheader()
        for sample_id, result in zip(sample_ids, solved):
            selected_set = set(result.get("selected_global_indices", []))
            total_anomaly = float(np.sum(result["anomaly"]))
            for index, phase_id in enumerate(phase_ids):
                value = float(np.sum(result["components"][index]))
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "phase_id": phase_id,
                        "abundance": value,
                        "selected": str(index in selected_set).lower(),
                        "anomaly": "false",
                    }
                )
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "phase_id": "__anomaly__",
                    "abundance": total_anomaly,
                    "selected": str(total_anomaly > float(options["selection_threshold"])).lower(),
                    "anomaly": "true",
                }
            )
    abundance_csv = root / "abundance.csv"
    with abundance_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *phase_ids, "__anomaly__"])
        for sample_id, result, row in zip(sample_ids, solved, abundance):
            writer.writerow(
                [sample_id, *[float(value) for value in row], float(np.sum(result["anomaly"]))]
            )
    summary_csv = root / "sample_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "sample_id",
            "selected_phase_ids",
            "selected_phase_labels",
            "abundance_pct",
            "anomaly_pct",
            "objective",
            "solver_status",
            "warnings",
            "element_present",
            "element_missing",
            "element_violated",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id, result in zip(sample_ids, solved):
            indices = list(result.get("selected_global_indices", []))
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "selected_phase_ids": ";".join(phase_ids[index] for index in indices),
                    "selected_phase_labels": ";".join(
                        str(library[index].get("label", phase_ids[index])) for index in indices
                    ),
                    "abundance_pct": ";".join(
                        f"{float(np.sum(result['components'][index])) * 100:.8g}"
                        for index in indices
                    ),
                    "anomaly_pct": f"{float(np.sum(result['anomaly'])) * 100:.8g}",
                    "objective": float(result["objective"]),
                    "solver_status": result.get("solver", "unknown"),
                    "warnings": " | ".join(result.get("warnings", [])),
                    "element_present": ";".join(result.get("element_prior", {}).get("present", [])),
                    "element_missing": ";".join(result.get("element_missing", [])),
                    "element_violated": ";".join(result.get("element_violated", [])),
                }
            )
    metadata = {
        "schema_version": 1,
        "stage": "unmix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "phase_count": len(library),
        "sample_ids": sample_ids,
        "phase_ids": phase_ids,
        "options": dict(options),
        "solver_statuses": [result.get("solver", "unknown") for result in solved],
        "files": {
            "results_pickle": results_pickle.name,
            "results_npz": results_npz.name,
            "assignments": assignments_csv.name,
            "abundance": abundance_csv.name,
            "sample_summary": summary_csv.name,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return results_pickle, results_npz, assignments_csv, abundance_csv, summary_csv, metadata

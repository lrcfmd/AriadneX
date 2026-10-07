"""Public mixture preparation seam and artifact persistence."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ariadnex.config import ProjectConfig
from ariadnex.contracts import MixtureArtifact

from .clean import preprocess_pattern
from .load import _coerce_path, load_mixture_records_from_config


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _output_root(config: ProjectConfig, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    configured = config.get("project.output_root") or config.get("output_root")
    if configured is None:
        configured = "runs/02_mixtures"
    path = _coerce_path(configured, base_dir=config.base_dir)
    if path is None:
        path = Path("runs/02_mixtures").resolve()
    # A direct call receives the project root, while workflow.run_project passes
    # the stage directory explicitly.  Keep the direct call self-contained.
    if path.name != "02_mixtures":
        path = path / "02_mixtures"
    return path


_SIGNATURE_CLEANING_KEYS = (
    "mode",
    "keep_top_k",
    "min_components",
    "unsupported_strength_mult",
    "strong_quantile_low_snr",
    "strong_quantile_high_snr",
)


def _cleaning_options(config: ProjectConfig) -> dict[str, Any]:
    """Return ``preprocess_pattern`` settings for the mixture branch.

    The public schema has one preprocessing contract shared by basis and
    mixture preparation.  This stage consumes only the mixture branch; the
    unmix stage applies the basis branch when it loads the library.
    """

    preprocessing = config.unmix.get("preprocessing", {})
    if not isinstance(preprocessing, Mapping):
        preprocessing = {}
    result: dict[str, Any] = {
        key: preprocessing[key]
        for key in (
            "theta_min",
            "theta_max",
            "gridpoints",
            "signature_points",
            "signature_reduce_mode",
            "center",
            "smooth",
            "window",
        )
        if key in preprocessing
    }

    background = preprocessing.get("background")
    if not isinstance(background, Mapping) or "mixture" not in background:
        raise ValueError("unmix.preprocessing.background.mixture must be configured")
    result["background"] = background["mixture"]

    denoise = preprocessing.get("denoise")
    if not isinstance(denoise, Mapping):
        raise ValueError("unmix.preprocessing.denoise must be configured")
    for key in ("window", "polyorder"):
        if key in denoise:
            result[f"denoise_{key}"] = denoise[key]
    if "mixture" not in denoise:
        raise ValueError("unmix.preprocessing.denoise.mixture must be configured")
    result["denoise"] = denoise["mixture"]

    cleaning = preprocessing.get("signature_cleaning")
    if not isinstance(cleaning, Mapping) or "mixture" not in cleaning:
        raise ValueError("unmix.preprocessing.signature_cleaning.mixture must be configured")
    mixture_cleaning = cleaning["mixture"]
    if not isinstance(mixture_cleaning, Mapping):
        raise ValueError("unmix.preprocessing.signature_cleaning.mixture must be a mapping")
    result["signature_cleaning"] = {
        key: mixture_cleaning[key] for key in _SIGNATURE_CLEANING_KEYS if key in mixture_cleaning
    }
    return result


def _write_manifest(path: Path, records: list[Mapping[str, Any]]) -> None:
    fields = [
        "sample_id",
        "source_path",
        "composition",
        "x_min",
        "x_max",
        "n_points",
        "n_signature",
        "signature_mass",
        "baseline_fraction",
        "offset",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            original_y = record.get("y")
            baseline = record.get("baseline")
            sig_mass = float(record["signature_weights"].sum())
            baseline_fraction = (
                float(baseline.sum() / max(float(original_y.sum()), 1e-30))
                if baseline is not None
                else 0.0
            )
            writer.writerow(
                {
                    "sample_id": str(record.get("sample_id", "")),
                    "source_path": str(record.get("source_path", "")),
                    "composition": ""
                    if record.get("composition") is None
                    else str(record.get("composition")),
                    "x_min": record.get("x_min", ""),
                    "x_max": record.get("x_max", ""),
                    "n_points": len(record.get("clean_x", [])),
                    "n_signature": len(record.get("signature_positions", [])),
                    "signature_mass": sig_mass,
                    "baseline_fraction": baseline_fraction,
                    "offset": "" if record.get("offset") is None else record.get("offset"),
                }
            )


def _short_range_count(records: list[Mapping[str, Any]], theta_min, theta_max) -> int:
    """Count patterns whose measured range falls short of the requested range by more than one step."""

    short = 0
    for record in records:
        x = np.sort(np.asarray(record["raw_x"], dtype=float))
        step = float(np.median(np.diff(x)))
        if (theta_min is not None and float(theta_min) < x[0] - step) or (
            theta_max is not None and float(theta_max) > x[-1] + step
        ):
            short += 1
    return short


def prepare_mixtures(config: ProjectConfig, output_dir: str | Path | None = None):
    """Load, clean, and persist all experimental patterns.

    Parameters
    ----------
    config:
        Resolved ``ProjectConfig``.  ``inputs.mixtures`` may reference a
        directory, one pattern, or a CSV sample manifest.
    output_dir:
        Artifact directory.  ``workflow.run_project`` supplies this path;
        direct callers may omit it and use ``project.output_root``.
    """

    root = _output_root(config, output_dir)
    root.mkdir(parents=True, exist_ok=True)
    options = _cleaning_options(config)
    records = load_mixture_records_from_config(config)
    processed = []
    for index, record in enumerate(records, 1):
        processed.append(preprocess_pattern(record, **options))
        print(
            f"[mixtures] {index}/{len(records)} samples complete: {record['sample_id']}",
            file=sys.stderr,
            flush=True,
        )
    short = _short_range_count(processed, options.get("theta_min"), options.get("theta_max"))
    if short:
        left, right = (
            "-" if options.get(key) is None else f"{float(options[key]):g}"
            for key in ("theta_min", "theta_max")
        )
        print(
            f"[mixtures] {short}/{len(processed)} samples cover less than the requested range "
            f"[{left}, {right}]; using the measured points within it.",
            file=sys.stderr,
            flush=True,
        )

    manifest_path = root / "manifest.csv"
    patterns_path = root / "patterns.npz"
    signatures_path = root / "signatures.npz"
    metadata_path = root / "metadata.json"
    _write_manifest(manifest_path, processed)

    # Object arrays preserve variable-length source grids and signatures while
    # remaining readable from any NumPy version with allow_pickle=True.
    np.savez_compressed(
        patterns_path,
        sample_ids=np.asarray([str(r["sample_id"]) for r in processed], dtype=str),
        source_paths=np.asarray([str(r.get("source_path", "")) for r in processed], dtype=str),
        compositions=np.asarray(
            ["" if r.get("composition") is None else str(r.get("composition")) for r in processed],
            dtype=str,
        ),
        x=np.asarray([r["clean_x"] for r in processed], dtype=object),
        y=np.asarray([r["clean_y"] for r in processed], dtype=object),
        raw_x=np.asarray([r["raw_x"] for r in processed], dtype=object),
        raw_y=np.asarray([r["raw_y"] for r in processed], dtype=object),
        baseline=np.asarray(
            [
                np.zeros_like(r["clean_y"]) if r.get("baseline") is None else r["baseline"]
                for r in processed
            ],
            dtype=object,
        ),
    )
    np.savez_compressed(
        signatures_path,
        sample_ids=np.asarray([str(r["sample_id"]) for r in processed], dtype=str),
        positions=np.asarray([r["signature_positions"] for r in processed], dtype=object),
        weights=np.asarray([r["signature_weights"] for r in processed], dtype=object),
        widths=np.asarray([r["signature_widths"] for r in processed], dtype=object),
    )
    metadata = {
        "schema_version": 1,
        "stage": "mixtures",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(processed),
        "sample_ids": [str(r["sample_id"]) for r in processed],
        "options": options,
        "files": {
            "manifest": manifest_path.name,
            "patterns": patterns_path.name,
            "signatures": signatures_path.name,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    return MixtureArtifact(
        root=root,
        manifest_path=manifest_path,
        patterns_path=patterns_path,
        signatures_path=signatures_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def load_cleaned_mixtures(artifact: MixtureArtifact) -> list[dict[str, Any]]:
    """Read a ``MixtureArtifact`` into in-memory records for the unmix seam."""

    # Prepared mixtures intentionally store ragged curves as object arrays, so
    # each archive is opened once with the mode required by this artifact
    # format.  Materialising values before the contexts close also releases
    # both archive handles before the solver starts.
    with np.load(Path(artifact.patterns_path), allow_pickle=True) as payload:
        with np.load(Path(artifact.signatures_path), allow_pickle=True) as signatures:
            ids = [str(value) for value in payload["sample_ids"].tolist()]
            source_paths = payload.get("source_paths", np.asarray([""] * len(ids))).tolist()
            compositions = payload.get("compositions", np.asarray([""] * len(ids))).tolist()
            raw_x = payload["raw_x"].tolist()
            raw_y = payload["raw_y"].tolist()
            clean_x = payload["x"].tolist()
            clean_y = payload["y"].tolist()
            positions = signatures["positions"].tolist()
            weights = signatures["weights"].tolist()
            widths = signatures.get("widths", np.asarray([None] * len(ids), dtype=object)).tolist()

    result = []
    for index, sample_id in enumerate(ids):
        result.append(
            {
                "sample_id": sample_id,
                "source_path": str(source_paths[index]),
                "composition": compositions[index] or None,
                "x": raw_x[index],
                "y": raw_y[index],
                "clean_x": clean_x[index],
                "clean_y": clean_y[index],
                "signature_positions": positions[index],
                "signature_weights": weights[index],
                "signature_widths": widths[index],
            }
        )
    return result


__all__ = ["prepare_mixtures", "load_cleaned_mixtures"]

from __future__ import annotations

from pathlib import Path

import pytest

from ariadnex.cache import phase_cif_paths
from ariadnex.config import ProjectConfig
from ariadnex.doctor import input_preflight
from ariadnex.library import build as build_module
from ariadnex.library.manifest import discover_cifs


def _config(tmp_path: Path, *, cifs: Path | None, phase_manifest: Path | None) -> ProjectConfig:
    inputs: dict[str, str] = {"mixtures": str(tmp_path / "patterns")}
    if cifs is not None:
        inputs["cifs"] = str(cifs)
    if phase_manifest is not None:
        inputs["phase_manifest"] = str(phase_manifest)
    return ProjectConfig.from_mapping(
        {"project": {}, "inputs": inputs},
        base_dir=tmp_path,
    )


def _write_pattern_dir(tmp_path: Path) -> Path:
    patterns = tmp_path / "patterns"
    patterns.mkdir()
    (patterns / "sample.xy").write_text("10 1\n11 2\n", encoding="utf-8")
    return patterns


@pytest.mark.parametrize("absolute_reference", [True, False])
def test_manifest_only_preflight_discovery_and_cache_use_same_cifs(
    tmp_path: Path, absolute_reference: bool
) -> None:
    cif = tmp_path / "local cifs" / "phase.cif"
    cif.parent.mkdir()
    cif.write_text("data_phase\n", encoding="utf-8")
    manifest = tmp_path / "prepared" / "raw_phases.csv"
    manifest.parent.mkdir()
    cif_reference = str(cif.resolve()) if absolute_reference else "../local cifs/phase.cif"
    manifest.write_text(
        f"phase_id,cif_path,label\nphase,{cif_reference},Phase\n", encoding="utf-8"
    )
    _write_pattern_dir(tmp_path)
    config = _config(tmp_path, cifs=None, phase_manifest=manifest)

    report = input_preflight(config)
    assert report.ok, [(check.name, check.detail) for check in report.checks if not check.ok]
    assert all(check.name != "inputs.cifs" for check in report.checks)

    entries = discover_cifs(manifest.parent, phase_manifest=manifest)
    cached_paths = phase_cif_paths(config)
    assert [entry.path for entry in entries] == [cif.resolve()]
    assert cached_paths == [manifest.resolve(), cif.resolve()]


def test_manifest_only_preflight_reports_missing_manifest_or_cif(tmp_path: Path) -> None:
    _write_pattern_dir(tmp_path)
    missing_manifest = tmp_path / "missing.csv"
    report = input_preflight(_config(tmp_path, cifs=None, phase_manifest=missing_manifest))
    checks = {check.name: check for check in report.checks}
    assert not checks["inputs.phase_manifest"].ok
    assert all(check.name != "inputs.cifs" for check in report.checks)

    manifest = tmp_path / "raw_phases.csv"
    manifest.write_text(
        "phase_id,cif_path,label\nphase,/missing/phase.cif,Phase\n", encoding="utf-8"
    )
    report = input_preflight(_config(tmp_path, cifs=None, phase_manifest=manifest))
    checks = {check.name: check for check in report.checks}
    assert not checks["inputs.phase_manifest.references"].ok
    assert "/missing/phase.cif" in checks["inputs.phase_manifest.references"].detail


def test_directory_cif_input_remains_valid_without_phase_manifest(tmp_path: Path) -> None:
    cifs = tmp_path / "cifs"
    cifs.mkdir()
    (cifs / "phase.cif").write_text("data_phase\n", encoding="utf-8")
    _write_pattern_dir(tmp_path)

    report = input_preflight(_config(tmp_path, cifs=cifs, phase_manifest=None))
    assert report.ok, [(check.name, check.detail) for check in report.checks if not check.ok]


def test_library_build_uses_manifest_parent_when_cifs_root_is_omitted(tmp_path: Path) -> None:
    manifest = tmp_path / "prepared" / "raw_phases.csv"
    manifest.parent.mkdir()
    manifest.write_text(
        "phase_id,cif_path\nphase,../missing/phase.cif\n", encoding="utf-8"
    )
    config = ProjectConfig.from_mapping(
        {"project": {}, "inputs": {"phase_manifest": str(manifest)}},
        base_dir=tmp_path,
    )

    with pytest.raises(FileNotFoundError, match=r"Phase manifest row 1 references missing CIF"):
        build_module.build_library(config, tmp_path / "artifacts")

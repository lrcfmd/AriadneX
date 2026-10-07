from __future__ import annotations

import csv
import shlex
import shutil
from pathlib import Path

import pytest

import ariadnex.example_inputs as example_inputs
from ariadnex.cli import main
from ariadnex.config import load_config
from ariadnex.example_inputs import (
    ExampleEntry,
    ExampleInputError,
    _match_catalog_to_cifs,
    prepare_example,
)

ROOT = Path(__file__).resolve().parents[1]


def _read_catalog(example: str) -> list[dict[str, str]]:
    with (ROOT / "examples" / example / "entries.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_icsd(path: Path, database_id: str, *, next_line: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = f"_database_code_ICSD\n'{database_id}'" if next_line else f"_database_code_ICSD '{database_id}'"
    path.write_text(
        f"# _database_code_ICSD 999999\n{tag} # source metadata\n",
        encoding="utf-8",
    )


def _write_pcd(path: Path, database_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"data_{database_id}\n#_database_code_PCD {database_id}\n",
        encoding="utf-8",
    )


@pytest.fixture
def isolated_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repository = tmp_path / "isolated repository with spaces"
    shutil.copytree(ROOT / "examples", repository / "examples")
    shutil.copytree(ROOT / "configs", repository / "configs")
    monkeypatch.setattr(example_inputs, "_REPOSITORY_ROOT", repository)
    return repository


@pytest.mark.parametrize(
    "example,database_code",
    [
        ("controlled_leave_one_out", "PCD"),
        ("open_world_phase_discovery", "ICSD"),
    ],
)
def test_prepare_example_preserves_catalog_order_and_uses_canonical_config(
    tmp_path: Path,
    isolated_repository: Path,
    example: str,
    database_code: str,
) -> None:
    source_dir = tmp_path / "exported CIFs"
    catalog = _read_catalog(example)
    source_by_id: dict[str, Path] = {}
    for index, row in enumerate(catalog):
        # Filenames intentionally carry no usable database ID.
        path = source_dir / f"nested-{index % 7}" / f"structure-{index:04d}.CIF"
        if database_code == "ICSD":
            _write_icsd(path, row["database_id"], next_line=index % 2 == 0)
        else:
            _write_pcd(path, row["database_id"])
        source_by_id[row["database_id"]] = path.resolve()

    # Extra files from another source do not invalidate the intended database export.
    unrelated = source_dir / "unrelated.cif"
    unrelated.write_text("# a file without matching database metadata\n", encoding="utf-8")

    config_path = isolated_repository / "configs" / f"{example}.yaml"
    config_bytes = config_path.read_bytes()
    prepared = prepare_example(example, source_dir)
    with prepared.manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert [(row["phase_id"], row["label"]) for row in manifest] == [
        (row["phase_id"], row["label"]) for row in catalog
    ]
    assert [Path(row["cif_path"]) for row in manifest] == [
        source_by_id[row["database_id"]] for row in catalog
    ]
    assert all(Path(row["cif_path"]).is_absolute() for row in manifest)
    assert prepared.entry_count == len(catalog)

    assert prepared.config_path == config_path
    assert prepared.manifest_path == (
        isolated_repository / "data" / "local_examples" / example / "raw_phases.csv"
    )
    assert prepared.output_dir == prepared.manifest_path.parent
    canonical_config = load_config(config_path)
    assert canonical_config.get("inputs.cifs") is None
    assert canonical_config.get("inputs.phase_manifest") == str(prepared.manifest_path.resolve())
    assert config_path.read_bytes() == config_bytes
    assert not (prepared.output_dir / "run.yaml").exists()


def test_metadata_matching_uses_random_filenames_and_ignores_comment_only_icsd_tags(
    tmp_path: Path,
) -> None:
    cif_dir = tmp_path / "icdd-export"
    high = cif_dir / "random-name.CIF"
    low = cif_dir / "nested" / "another-random-name.cif"
    _write_icsd(high, "42", next_line=True)
    _write_icsd(low, "7")
    rows = _match_catalog_to_cifs(
        cif_dir,
        "ICSD",
        [ExampleEntry("later", "later row", "42"), ExampleEntry("first", "first row", "7")],
    )
    assert [entry.phase_id for entry, _ in rows] == ["later", "first"]
    assert [path for _, path in rows] == [high.resolve(), low.resolve()]


def test_pcd_comment_metadata_must_match_data_record(tmp_path: Path) -> None:
    cif_dir = tmp_path / "pcd-export"
    source = cif_dir / "unrelated-name.cif"
    _write_pcd(source, "1234")
    rows = _match_catalog_to_cifs(cif_dir, "PCD", [ExampleEntry("A", "calcite", "1234")])
    assert rows == [(ExampleEntry("A", "calcite", "1234"), source.resolve())]

    source.write_text("data_9999\n#_database_code_PCD 1234\n", encoding="utf-8")
    with pytest.raises(ExampleInputError, match="do not match a data_<id> record"):
        _match_catalog_to_cifs(cif_dir, "PCD", [ExampleEntry("A", "calcite", "1234")])


@pytest.mark.parametrize(
    "database_code,body",
    [
        (
            "ICSD",
            "data_unrelated\n_description\n;\n" "_database_code_ICSD 42\n;\n",
        ),
        (
            "PCD",
            "data_unrelated\n_description\n;\n" "data_42\n#_database_code_PCD 42\n;\n",
        ),
    ],
)
def test_database_tags_inside_semicolon_text_fields_are_ignored(
    tmp_path: Path, database_code: str, body: str
) -> None:
    source = tmp_path / "text-field.cif"
    source.write_text(body, encoding="utf-8")
    with pytest.raises(ExampleInputError, match=r"catalog .* ID\(s\) were not found"):
        _match_catalog_to_cifs(
            tmp_path,
            database_code,
            [ExampleEntry("wanted", "wanted", "42")],
        )


def test_prepare_example_is_idempotent_and_backs_up_an_edited_manifest(
    tmp_path: Path, isolated_repository: Path
) -> None:
    source_dir = tmp_path / "pcd"
    for index, row in enumerate(_read_catalog("controlled_leave_one_out")):
        _write_pcd(source_dir / f"structure-{index}.cif", row["database_id"])

    first = prepare_example("controlled_leave_one_out", source_dir)
    manifest_bytes = first.manifest_path.read_bytes()
    config_bytes = first.config_path.read_bytes()
    identical = prepare_example("controlled_leave_one_out", source_dir)
    assert identical.backup_dir is None
    assert identical.manifest_path.read_bytes() == manifest_bytes
    assert identical.config_path.read_bytes() == config_bytes

    edited_manifest = b"# user change to retain\n"
    first.manifest_path.write_bytes(edited_manifest)
    replaced = prepare_example("controlled_leave_one_out", source_dir)
    assert replaced.backup_dir is not None
    assert replaced.backup_dir.name.startswith("previous-")
    assert (replaced.backup_dir / "raw_phases.csv").read_bytes() == edited_manifest
    assert replaced.manifest_path.read_bytes() == manifest_bytes
    assert replaced.config_path.read_bytes() == config_bytes
    assert len(list(replaced.output_dir.glob("previous-*"))) == 1
    assert not (replaced.output_dir / "run.yaml").exists()


def test_missing_and_duplicate_requested_database_ids_fail_clearly(tmp_path: Path) -> None:
    cif_dir = tmp_path / "ids"
    _write_icsd(cif_dir / "one.cif", "11")
    missing = [ExampleEntry("A", "missing", "12")]
    with pytest.raises(ExampleInputError, match=r"catalog ICSD ID\(s\) were not found"):
        _match_catalog_to_cifs(cif_dir, "ICSD", missing)

    _write_icsd(cif_dir / "nested" / "duplicate.cif", "11")
    with pytest.raises(ExampleInputError, match="duplicate ICSD database IDs"):
        _match_catalog_to_cifs(
            cif_dir, "ICSD", [ExampleEntry("A", "first", "11")]
        )


def test_preparation_never_writes_through_manifest_symlinks(
    tmp_path: Path, isolated_repository: Path
) -> None:
    source_dir = tmp_path / "pcd"
    for index, row in enumerate(_read_catalog("controlled_leave_one_out")):
        _write_pcd(source_dir / f"original-{index}.cif", row["database_id"])
    manifest_path = (
        isolated_repository
        / "data"
        / "local_examples"
        / "controlled_leave_one_out"
        / "raw_phases.csv"
    )
    manifest_path.parent.mkdir(parents=True)
    original = source_dir / "original-0.cif"
    original_bytes = original.read_bytes()
    manifest_path.symlink_to(original)
    with pytest.raises(ExampleInputError, match="must be a regular file"):
        prepare_example("controlled_leave_one_out", source_dir)
    assert original.read_bytes() == original_bytes
    assert manifest_path.is_symlink()


def test_cli_prints_canonical_config_run_command_with_quoted_path(
    tmp_path: Path,
    isolated_repository: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_dir = tmp_path / "local CIF export with spaces"
    for index, row in enumerate(_read_catalog("controlled_leave_one_out")):
        _write_pcd(source_dir / f"random-{index}.cif", row["database_id"])
    config_path = isolated_repository / "configs" / "controlled_leave_one_out.yaml"
    result = main(
        [
            "prepare-example",
            "controlled_leave_one_out",
            "--cif-dir",
            str(source_dir),
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "Run command: cd " in captured.out
    repository_root = Path(__file__).resolve().parents[1]
    expected_command = (
        f"cd {shlex.quote(str(repository_root))} && ./workflow run "
        f"{shlex.quote(str(config_path))}"
    )
    assert f"Run command: {expected_command}" in captured.out
    assert f"Config: {config_path}" in captured.out
    assert "Previous outputs backed up:" not in captured.out
    manifest_dir = (
        isolated_repository / "data" / "local_examples" / "controlled_leave_one_out"
    )
    assert not (manifest_dir / "run.yaml").exists()

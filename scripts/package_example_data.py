#!/usr/bin/env python3
"""Package only the measured inputs needed by the two paper examples."""

from __future__ import annotations

import argparse
import csv
import io
import tarfile
from pathlib import Path

DATA_RELEASE = "example-data-v5"
DATASETS = ("controlled_leave_one_out", "open_world_phase_discovery")
TABLES = ("samples.csv", "ground_truth.csv", "basis_order.csv")
PATTERN_SUFFIXES = {".asc", ".xy", ".xye", ".txt", ".csv", ".npy", ".npz"}


def input_files(data_root: Path) -> list[Path]:
    """Use an explicit allowlist so local CIFs and results cannot enter a release."""
    files: set[Path] = set()
    for dataset in DATASETS:
        root = data_root / dataset
        files.update(root / name for name in (*TABLES, "experimental.instprm"))
        with (root / "samples.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No measured samples in {root / 'samples.csv'}")
        for row in rows:
            relative = Path(row["pattern_path"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.parts[:1] != ("mixed_patterns",)
                or relative.suffix.lower() not in PATTERN_SUFFIXES
            ):
                raise ValueError(f"Unexpected measured-pattern path: {relative}")
            files.add(root / relative)
    files.add(data_root / "open_world_phase_discovery" / "candidate_library.instprm")
    for path in files:
        relative = path.relative_to(data_root)
        if any((data_root / part).is_symlink() for part in (relative, *relative.parents)):
            raise ValueError(f"Example inputs must be regular files inside the data folder: {path}")
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty example input: {path}")
    return sorted(files)


def package(data_root: Path, output: Path) -> None:
    files = input_files(data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an existing release archive.
    with output.open("xb") as destination, tarfile.open(fileobj=destination, mode="w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(Path("data") / path.relative_to(data_root)), recursive=False)
        version = (DATA_RELEASE + "\n").encode()
        info = tarfile.TarInfo("data/.example-data-version")
        info.size = len(version)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(version))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new example-data.tar.gz path")
    parser.add_argument(
        "--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    args = parser.parse_args()
    package(args.data_root, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

"""Command-line interface for running and inspecting AriadneX projects."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Sequence

from .config import ConfigError
from .example_inputs import ExampleInputError, prepare_example
from .workflow import DoctorReport, WorkflowError, doctor, report_run, run_project


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ariadnex", description="PXRD library and unmix workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="check configuration, inputs, and runtime prerequisites"
    )
    doctor_parser.add_argument("config", nargs="?", type=Path)
    doctor_parser.add_argument("--config", dest="config_option", type=Path)

    run_parser = subparsers.add_parser("run", help="run all workflow stages")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--force", action="store_true", help="ignore completed stage caches")

    report_parser = subparsers.add_parser("report", help="render reports for an existing run")
    report_parser.add_argument("run_root", type=Path)
    report_parser.add_argument("--config", dest="config_option", type=Path)
    report_parser.add_argument("--force", action="store_true", help="replace cached report files")

    example_parser = subparsers.add_parser(
        "prepare-example",
        help="match local CIFs to a paper example and write raw_phases.csv",
        description="Match the example's entry list against CIF database IDs and write raw_phases.csv.",
    )
    example_parser.add_argument(
        "example",
        choices=("controlled_leave_one_out", "open_world_phase_discovery"),
    )
    example_parser.add_argument(
        "--cif-dir",
        required=True,
        type=Path,
        metavar="FOLDER",
        help=(
            "search this folder and its subfolders for the required CIF entries; "
            "any folder name is fine"
        ),
    )
    return parser


def _print_doctor(report: DoctorReport) -> None:
    for check in report.checks:
        marker = "OK" if check.ok else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")
    print("doctor: " + ("ok" if report.ok else "failed"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "doctor":
        config = args.config_option or args.config
        try:
            report = doctor(config)
        except (ConfigError, OSError, ValueError) as exc:
            print(f"doctor failed: {exc}", file=sys.stderr)
            return 1
        _print_doctor(report)
        return 0 if report.ok else 1
    if args.command == "run":
        try:
            artifact = run_project(args.config, force=args.force)
        except (ConfigError, OSError, ValueError, WorkflowError) as exc:
            print(f"run failed: {exc}", file=sys.stderr)
            return 1
        print(artifact.root)
        return 0
    if args.command == "report":
        try:
            artifact = report_run(args.run_root, config_path=args.config_option, force=args.force)
        except (ConfigError, OSError, ValueError, WorkflowError) as exc:
            print(f"report failed: {exc}", file=sys.stderr)
            return 1
        print(artifact.root)
        return 0
    if args.command == "prepare-example":
        try:
            prepared = prepare_example(args.example, args.cif_dir)
        except (ExampleInputError, OSError, ValueError) as exc:
            print(f"prepare-example failed: {exc}", file=sys.stderr)
            return 1
        repository_root = Path(__file__).resolve().parents[2]
        print(f"Prepared {prepared.entry_count} phase entries for {prepared.example}.")
        print(f"Manifest: {prepared.manifest_path}")
        print(f"Config: {prepared.config_path}")
        if prepared.backup_dir is not None:
            print(f"Previous outputs backed up: {prepared.backup_dir}")
        command = (
            f"cd {shlex.quote(str(repository_root))} && ./workflow run "
            f"{shlex.quote(str(prepared.config_path))}"
        )
        print(f"Run command: {command}")
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

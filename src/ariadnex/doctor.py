"""Runtime and input preflight checks for the workflow CLI."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cache import manifest_reference_paths
from .config import ConfigError, ProjectConfig, load_config
from .ground_truth import load_ground_truth
from .mixtures.load import (
    PATTERN_PATH_ALIASES,
    SUPPORTED_SUFFIXES,
    csv_input_kind,
    load_pattern_file,
)
from .runtime import CondaRunner, EncoderRunner, Executor, RuntimeSpec

_CIF_PATH_ALIASES = ("cif_path", "cif", "path", "source_path", "file", "filename")
_PHASE_ID_ALIASES = ("phase_id", "phase", "id")


class DoctorCheck:
    def __init__(self, name: str, ok: bool, detail: str):
        self.name = name
        self.ok = ok
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


class DoctorReport:
    def __init__(self, checks: Iterable[DoctorCheck]):
        self.checks = tuple(checks)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [check.to_dict() for check in self.checks]}


def _config_value(config: ProjectConfig, *keys: str) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _normalise_fields(fieldnames: Iterable[Any]) -> dict[str, Any]:
    return {
        str(field).strip().lower(): field
        for field in fieldnames
        if field is not None and str(field).strip()
    }


def _csv_contract(
    checks: list[DoctorCheck],
    *,
    name: str,
    path: Path,
    required_groups: Iterable[tuple[str, ...]],
    reference_aliases: Iterable[str] = (),
    reference_base: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]] | None:
    """Check a CSV header, rows, and optionally all referenced files."""

    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError) as exc:
        checks.append(DoctorCheck(f"{name}.csv", False, f"cannot read {path}: {exc}"))
        return None

    fields = _normalise_fields(fieldnames)
    if not fields:
        checks.append(DoctorCheck(f"{name}.csv", False, f"CSV has no header: {path}"))
        return None
    missing = ["/".join(group) for group in required_groups if not set(group).intersection(fields)]
    if missing:
        checks.append(
            DoctorCheck(
                f"{name}.csv",
                False,
                f"{path} is missing a required column (one of: {', '.join(missing)})",
            )
        )
        return fields, rows
    if not rows:
        checks.append(DoctorCheck(f"{name}.csv", False, f"{path} has a header but no data rows"))
        return fields, rows

    checks.append(
        DoctorCheck(f"{name}.csv", True, f"{len(rows)} data rows; required columns present")
    )
    aliases = tuple(reference_aliases)
    if aliases:
        reference_key = next((fields[alias] for alias in aliases if alias in fields), None)
        if reference_key is not None:
            blank_rows = [
                index
                for index, row in enumerate(rows, start=2)
                if not str(row.get(reference_key, "") or "").strip()
            ]
            if blank_rows:
                checks.append(
                    DoctorCheck(
                        f"{name}.references",
                        False,
                        f"{path} has blank file references on CSV rows {', '.join(map(str, blank_rows[:8]))}"
                        + (" ..." if len(blank_rows) > 8 else ""),
                    )
                )
                return fields, rows
            references = manifest_reference_paths(path, aliases, base_path=reference_base)
            missing_paths = [item for item in references if not item.is_file()]
            if missing_paths:
                shown = ", ".join(str(item) for item in missing_paths[:8])
                suffix = " ..." if len(missing_paths) > 8 else ""
                checks.append(
                    DoctorCheck(
                        f"{name}.references", False, f"missing referenced files: {shown}{suffix}"
                    )
                )
            else:
                checks.append(
                    DoctorCheck(
                        f"{name}.references", True, f"{len(references)} referenced files found"
                    )
                )
    return fields, rows


def _pattern_contract(checks: list[DoctorCheck], *, name: str, path: Path) -> None:
    """Validate a CSV that the mixture loader treats as one numeric pattern."""

    try:
        record = load_pattern_file(path)
    except (OSError, UnicodeError, ValueError) as exc:
        checks.append(DoctorCheck(f"{name}.csv", False, f"cannot parse pattern {path}: {exc}"))
        return
    checks.append(
        DoctorCheck(
            f"{name}.csv",
            True,
            f"valid numeric diffraction pattern ({len(record['x'])} points)",
        )
    )


def _pattern_manifest_contract(
    checks: list[DoctorCheck],
    *,
    name: str,
    manifest: Path,
    fields: Mapping[str, Any],
    rows: list[dict[str, str]],
) -> None:
    """Parse every pattern named by a sample manifest during preflight."""

    reference_key = next((fields[alias] for alias in PATTERN_PATH_ALIASES if alias in fields), None)
    if reference_key is None:
        return
    failures: list[str] = []
    checked = 0
    for row_number, row in enumerate(rows, start=2):
        raw_path = str(row.get(reference_key, "") or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = manifest.parent / candidate
        try:
            load_pattern_file(candidate)
        except (OSError, UnicodeError, ValueError) as exc:
            failures.append(f"row {row_number}: {exc}")
        else:
            checked += 1
    if failures:
        shown = "; ".join(failures[:4])
        suffix = " ..." if len(failures) > 4 else ""
        checks.append(
            DoctorCheck(
                f"{name}.patterns",
                False,
                f"invalid referenced diffraction pattern(s): {shown}{suffix}",
            )
        )
    else:
        checks.append(
            DoctorCheck(f"{name}.patterns", True, f"{checked} referenced patterns parsed")
        )


def _append_path_check(
    checks: list[DoctorCheck],
    name: str,
    value: Any,
    *,
    expected: str = "path",
) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if expected == "file":
        ok = path.is_file()
    elif expected == "directory":
        ok = path.is_dir()
    else:
        ok = path.exists()
    checks.append(
        DoctorCheck(
            name,
            ok,
            str(path) if ok else f"missing or wrong type ({expected}): {path}",
        )
    )
    if ok and path.is_dir():
        broken = _broken_symlinks(path)
        checks.append(
            DoctorCheck(
                f"{name}.symlinks",
                not broken,
                "no broken symlinks" if not broken else ", ".join(str(item) for item in broken),
            )
        )
    return path


def _append_ground_truth_check(checks: list[DoctorCheck], config: ProjectConfig) -> None:
    ground_truth_value = _config_value(config, "inputs.ground_truth", "ground_truth")
    if ground_truth_value is not None:
        try:
            truth, _ = load_ground_truth(ground_truth_value)
        except (OSError, ValueError, csv.Error) as exc:
            checks.append(DoctorCheck("inputs.ground_truth", False, str(exc)))
        else:
            checks.append(
                DoctorCheck("inputs.ground_truth", True, f"{len(truth)} samples with ground truth")
            )


def _append_input_checks(checks: list[DoctorCheck], config: ProjectConfig) -> None:
    cifs_value = _config_value(config, "inputs.cifs", "cifs", "input_path", "library.input_path")
    phase_manifest_value = _config_value(
        config,
        "inputs.phase_manifest",
        "phase_manifest",
        "library.phase_manifest",
    )
    if cifs_value is None and phase_manifest_value is None:
        detail = (
            "missing required CIF input path or phase manifest "
            "(configure inputs.cifs or inputs.phase_manifest)"
        )
        checks.append(
            DoctorCheck(
                "inputs.cifs",
                False,
                detail,
            )
        )
    cifs = _append_path_check(checks, "inputs.cifs", cifs_value)
    phase_manifest = _append_path_check(
        checks, "inputs.phase_manifest", phase_manifest_value, expected="file"
    )

    if phase_manifest is not None and phase_manifest.is_file():
        _csv_contract(
            checks,
            name="inputs.phase_manifest",
            path=phase_manifest,
            required_groups=(_CIF_PATH_ALIASES,),
            reference_aliases=_CIF_PATH_ALIASES,
            reference_base=cifs,
        )
    elif cifs is not None and cifs.is_file() and cifs.suffix.lower() == ".csv":
        _csv_contract(
            checks,
            name="inputs.cifs",
            path=cifs,
            required_groups=(_CIF_PATH_ALIASES,),
            reference_aliases=_CIF_PATH_ALIASES,
            reference_base=cifs,
        )
    elif (
        cifs is not None
        and cifs.is_dir()
        and not any(item.is_file() and item.suffix.lower() == ".cif" for item in cifs.rglob("*"))
    ):
        checks.append(DoctorCheck("inputs.cifs.data", False, f"no .cif files found under {cifs}"))

    mixtures_value = _config_value(config, "inputs.mixtures", "mixtures")
    if mixtures_value is None:
        checks.append(
            DoctorCheck(
                "inputs.mixtures",
                False,
                "missing required diffraction input path (configure inputs.mixtures)",
            )
        )
    mixtures = _append_path_check(checks, "inputs.mixtures", mixtures_value)
    if mixtures is not None and mixtures.is_file() and mixtures.suffix.lower() == ".csv":
        try:
            mixture_kind = csv_input_kind(mixtures)
        except (OSError, UnicodeError, csv.Error) as exc:
            checks.append(
                DoctorCheck("inputs.mixtures.csv", False, f"cannot read {mixtures}: {exc}")
            )
        else:
            if mixture_kind == "manifest":
                contract = _csv_contract(
                    checks,
                    name="inputs.mixtures",
                    path=mixtures,
                    required_groups=(PATTERN_PATH_ALIASES,),
                    reference_aliases=PATTERN_PATH_ALIASES,
                )
                if contract is not None:
                    fields, rows = contract
                    _pattern_manifest_contract(
                        checks,
                        name="inputs.mixtures",
                        manifest=mixtures,
                        fields=fields,
                        rows=rows,
                    )
            else:
                _pattern_contract(checks, name="inputs.mixtures", path=mixtures)
    elif mixtures is not None and mixtures.is_dir():
        pattern_files = [
            item
            for item in mixtures.rglob("*")
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if pattern_files:
            checks.append(
                DoctorCheck(
                    "inputs.mixtures.data",
                    True,
                    f"{len(pattern_files)} supported pattern files found",
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    "inputs.mixtures.data",
                    False,
                    f"no supported diffraction files found under {mixtures}",
                )
            )

    _append_ground_truth_check(checks, config)

    instrument_values = (
        ("inputs.instrument", _config_value(config, "inputs.instrument")),
        (
            "library.simulation.instrument",
            _config_value(
                config, "library.simulation.instrument", "library.simulation.instrument_path"
            ),
        ),
        (
            "library.basis_simulation.instrument",
            _config_value(
                config,
                "library.basis_simulation.instrument",
                "library.basis_simulation.instrument_path",
            ),
        ),
    )
    seen_instruments: set[Path] = set()
    for name, value in instrument_values:
        if value is None:
            continue
        path = Path(str(value)).expanduser()
        if path in seen_instruments:
            continue
        seen_instruments.add(path)
        _append_path_check(checks, name, path, expected="file")

    phase_order = _config_value(
        config,
        "unmix.solver.phase_order_manifest",
        "unmix.phase_order_manifest",
    )
    phase_order_path = _append_path_check(
        checks, "unmix.solver.phase_order_manifest", phase_order, expected="file"
    )
    if (
        phase_order_path is not None
        and phase_order_path.is_file()
        and phase_order_path.suffix.lower() == ".csv"
    ):
        _csv_contract(
            checks,
            name="unmix.solver.phase_order_manifest",
            path=phase_order_path,
            required_groups=(("phase_id",),),
        )

    for name, value in (
        ("inputs.anomaly_reference", _config_value(config, "inputs.anomaly_reference")),
        ("reports.anomaly_reference", _config_value(config, "reports.anomaly_reference")),
    ):
        if value is not None:
            _append_path_check(checks, name, value)


def input_preflight(config: ProjectConfig) -> DoctorReport:
    """Validate configured data contracts before starting scientific stages."""

    checks: list[DoctorCheck] = []
    _append_input_checks(checks, config)
    return DoctorReport(checks)


def run_preflight(config: ProjectConfig) -> DoctorReport:
    """Validate run inputs and checkpoint availability without loading the model."""

    checks: list[DoctorCheck] = []
    _append_checkpoint_check(checks, config)
    _append_input_checks(checks, config)
    return DoctorReport(checks)


def report_preflight(config: ProjectConfig) -> DoctorReport:
    """Check only external files consumed while rendering existing results."""

    checks: list[DoctorCheck] = []
    _append_ground_truth_check(checks, config)
    reference = _config_value(config, "inputs.anomaly_reference", "reports.anomaly_reference")
    if isinstance(reference, (str, Path)):
        _append_path_check(checks, "reports.anomaly_reference", reference)
    elif isinstance(reference, Mapping):
        for alias in ("path", "csv_path", "reference_path", "file", "filename", "directory", "dir"):
            value = reference.get(alias)
            if isinstance(value, (str, Path)) and str(value).strip():
                _append_path_check(checks, f"reports.anomaly_reference.{alias}", value)
    return DoctorReport(checks)


def _append_checkpoint_check(checks: list[DoctorCheck], config: ProjectConfig) -> bool | None:
    value = _config_value(config, "model.checkpoint", "model.checkpoint_path")
    if value is None:
        return None
    checkpoint = Path(str(value)).expanduser()
    available = checkpoint.is_file()
    checks.append(
        DoctorCheck(
            "model.checkpoint",
            available,
            str(checkpoint) if available else f"checkpoint file not found: {checkpoint}",
        )
    )
    return available


def _broken_symlinks(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        (item for item in root.rglob("*") if item.is_symlink() and not item.exists()), key=str
    )


def _probe_runtime(
    runner: CondaRunner,
    code: str,
    name: str,
    *,
    env: Mapping[str, str] | None = None,
) -> DoctorCheck:
    try:
        result = runner.run(["python", "-c", code], check=True, env=env)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = str(exc)
        stderr = getattr(exc, "stderr", None)
        if stderr:
            detail = str(stderr).strip() or detail
        return DoctorCheck(name, False, detail)
    if getattr(result, "returncode", 0) != 0:
        detail = str(
            getattr(result, "stderr", "") or getattr(result, "stdout", "") or "probe failed"
        ).strip()
        return DoctorCheck(name, False, detail)
    return DoctorCheck(name, True, "probe succeeded")


def _encoder_probe(config: ProjectConfig | None) -> tuple[str, str]:
    """Build an import or strict checkpoint-load probe."""

    imports = "import numpy as np; import torch; import omegaconf; "
    checkpoint_value = (
        None
        if config is None
        else _config_value(config, "model.checkpoint", "model.checkpoint_path")
    )
    if checkpoint_value is None:
        return (
            "encoder.import",
            imports
            + "from ariadnex.library.pxrd_encoder import PXRDPatchTransformer; "
            + "probe = torch.from_numpy(np.zeros(1, dtype=np.float32)); "
            + "assert probe.detach().cpu().numpy().shape == (1,)",
        )

    checkpoint = Path(str(checkpoint_value))
    code = (
        imports
        + "from ariadnex.library.pxrd_encoder import load_pxrd_patch_transformer_checkpoint; "
        + f"model, cfg = load_pxrd_patch_transformer_checkpoint({str(checkpoint)!r}); "
        + "probe = torch.linspace(0.0, 1.0, int(cfg['signal_length']), dtype=torch.float32).reshape(1, -1); "
        + "output = np.asarray(model(probe)['cls_token'].detach().cpu().numpy(), dtype=np.float32); "
        + "assert output.shape == (1, int(cfg['encoder_embed_dim'])); "
    )
    return "encoder.checkpoint", code


def doctor(
    config_path: str | Path | ProjectConfig | None = None,
    *,
    runtime: RuntimeSpec | None = None,
    executor: Executor | None = None,
    environ: Mapping[str, str] | None = None,
) -> DoctorReport:
    """Check configuration, staged inputs, and runtime prerequisites."""

    checks = [
        DoctorCheck("python", True, "python runtime available"),
        DoctorCheck("yaml", True, "PyYAML import succeeded"),
    ]
    config: ProjectConfig | None = None
    checkpoint_available: bool | None = None
    if isinstance(config_path, ProjectConfig):
        config = config_path
    elif config_path is not None:
        try:
            config = load_config(config_path)
        except ConfigError as exc:
            checks.append(DoctorCheck("config", False, str(exc)))
    if config is not None:
        checks.append(DoctorCheck("config", True, str(config.source_path)))
        checkpoint_available = _append_checkpoint_check(checks, config)
        _append_input_checks(checks, config)

    host_environment = os.environ if environ is None else environ
    selected_runtime = runtime or RuntimeSpec.from_environment(host_environment)
    conda_available = (
        executor is not None
        or selected_runtime.is_current(host_environment)
        or shutil.which(selected_runtime.conda_exe, path=host_environment.get("PATH")) is not None
    )
    checks.append(
        DoctorCheck(
            "conda",
            conda_available,
            f"{selected_runtime.conda_exe} available"
            if conda_available
            else f"executable not found: {selected_runtime.conda_exe}",
        )
    )
    runner = CondaRunner(selected_runtime, executor=executor, environ=host_environment)
    encoder_check_name, encoder_code = _encoder_probe(config)
    if conda_available:
        process_env = EncoderRunner.worker_environment(
            repo_src=Path(__file__).resolve().parents[1],
            environ=host_environment,
        )
        checks.append(
            _probe_runtime(
                runner,
                "import sys; print(sys.executable)",
                "env.ariadnex",
                env=process_env,
            )
        )
        checks.append(
            _probe_runtime(
                runner,
                "import GSASIIscriptable",
                "gsasii.import",
                env=process_env,
            )
        )
        checks.append(
            _probe_runtime(
                runner,
                "import gurobipy as gp; m = gp.Model(); m.Params.OutputFlag = 0",
                "gurobi.license",
                env=process_env,
            )
        )
        if checkpoint_available is False:
            checks.append(
                DoctorCheck(
                    encoder_check_name,
                    False,
                    "strict checkpoint load skipped because the checkpoint file is missing",
                )
            )
        else:
            checks.append(_probe_runtime(runner, encoder_code, encoder_check_name, env=process_env))
    else:
        checks.extend(
            DoctorCheck(name, False, "conda executable unavailable")
            for name in ("env.ariadnex", "gsasii.import", "gurobi.license", encoder_check_name)
        )
    return DoctorReport(checks)


__all__ = [
    "DoctorCheck",
    "DoctorReport",
    "doctor",
    "input_preflight",
    "run_preflight",
    "report_preflight",
]

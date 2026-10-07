"""GSAS-II simulation adapter for the single AriadneX environment.

Only this module knows about the GSAS-II Python API. The production path uses
the installed ``GSASIIscriptable`` module; ``SimulationAdapter`` keeps the
library-construction seam explicit.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import GSASIIscriptable as G2sc
import numpy as np

from .artifact import CifMetadata, PxrdPattern


class SimulationAdapter(Protocol):
    """Adapter seam for converting CIFs into patterns."""

    def simulate(
        self,
        records: Iterable[CifMetadata],
        *,
        instrument_path: Path | None,
        theta_min: float,
        theta_max: float,
        n_points: int,
        output_dir: Path,
    ) -> Mapping[str, PxrdPattern | Any]: ...


def _target_grid(theta_min: float, theta_max: float, n_points: int) -> np.ndarray:
    if not np.isfinite(theta_min) or not np.isfinite(theta_max) or theta_max <= theta_min:
        raise ValueError("theta_max must be greater than theta_min")
    if int(n_points) < 2:
        raise ValueError("n_points must be at least two")
    return np.linspace(float(theta_min), float(theta_max), int(n_points), dtype=np.float64)


def _normalise_pattern(
    source_id: str,
    value: Any,
    target_theta: np.ndarray,
    *,
    decimal_places: int | None = None,
) -> PxrdPattern:
    if isinstance(value, PxrdPattern):
        x, y = value.x, value.y
        source_path = value.source_path
    elif isinstance(value, Mapping):
        x = value.get("x", value.get("theta", value.get("pxrd_x")))
        y = value.get("y", value.get("intensity", value.get("pxrd_y", value.get("ycalc"))))
        source_path = value.get("source_path")
        if x is None or y is None:
            raise ValueError(f"Simulation output for {source_id!r} must contain x/y arrays")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        x, y = value
        source_path = None
    else:
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError(f"Unsupported simulation output for {source_id!r}: {type(value)!r}")
        x, y = array[:, 0], array[:, 1]
        source_path = None

    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.size != y_arr.size or x_arr.size < 2:
        raise ValueError(f"Simulation output for {source_id!r} has invalid x/y lengths")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError(f"Simulation output for {source_id!r} contains non-finite values")
    order = np.argsort(x_arr, kind="stable")
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    unique_x, unique_index = np.unique(x_arr, return_index=True)
    x_arr, y_arr = unique_x, y_arr[unique_index]
    if x_arr.size < 2:
        raise ValueError(
            f"Simulation output for {source_id!r} has fewer than two unique x positions"
        )
    target_start = float(target_theta[0])
    target_end = float(target_theta[-1])
    tolerance = max(1e-5, abs(float(target_theta[1] - target_theta[0])) * 1e-3)
    if float(x_arr[0]) > target_start + tolerance or float(x_arr[-1]) < target_end - tolerance:
        raise ValueError(
            f"Simulation output for {source_id!r} does not cover target theta window "
            f"[{target_start}, {target_end}]"
        )

    if x_arr.size != target_theta.size or not np.allclose(
        x_arr, target_theta, rtol=0, atol=tolerance
    ):
        y_arr = np.interp(target_theta.astype(np.float64), x_arr, y_arr)
    output_x = np.asarray(target_theta, dtype=np.float64)
    output_y = np.asarray(y_arr, dtype=np.float64)
    if decimal_places is not None:
        # GSAS-II's historical ASC exporter writes both columns with five
        # decimals.  Reproducing that materialisation boundary is necessary:
        # the downstream dynamic-programming signature reduction is sensitive
        # to float32/last-bit changes even when the dense curves correlate at
        # >0.999999999999.
        output_x = np.round(output_x, int(decimal_places))
        output_y = np.round(output_y, int(decimal_places))
    return PxrdPattern(source_id=source_id, x=output_x, y=output_y, source_path=source_path)


class GsasIISimulator:
    """Direct GSAS-II powder-pattern simulator.

    The installed ``GSASIIscriptable`` module is used unless a compatible
    in-process backend is supplied.
    """

    def __init__(
        self,
        *,
        backend: Any | None = None,
        scale: float = 1.0,
        quiet: bool = True,
        output_decimal_places: int | None = 5,
    ) -> None:
        self.backend = backend
        self.scale = float(scale)
        self.quiet = bool(quiet)
        self.output_decimal_places = (
            None if output_decimal_places is None else int(output_decimal_places)
        )

    def simulate(
        self,
        records: Iterable[CifMetadata],
        *,
        instrument_path: Path | None,
        theta_min: float,
        theta_max: float,
        n_points: int,
        output_dir: Path,
    ) -> Mapping[str, PxrdPattern]:
        ordered_records = tuple(sorted(records, key=lambda item: item.source_id))
        if not ordered_records:
            return {}
        theta = _target_grid(theta_min, theta_max, n_points)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.backend is not None:
            raw = self._call_backend(
                ordered_records,
                instrument_path=instrument_path,
                theta_min=theta_min,
                theta_max=theta_max,
                n_points=n_points,
                output_dir=output_dir,
            )
            return {
                record.source_id: _normalise_pattern(
                    record.source_id,
                    self._lookup_output(raw, record, index=index),
                    theta,
                    decimal_places=self.output_decimal_places,
                )
                for index, record in enumerate(ordered_records)
            }
        return self._simulate_gsasii(
            ordered_records,
            instrument_path=instrument_path,
            theta_min=theta_min,
            theta_max=theta_max,
            n_points=n_points,
            theta=theta,
        )

    def _call_backend(self, records: tuple[CifMetadata, ...], **kwargs: Any) -> Any:
        if hasattr(self.backend, "simulate"):
            return self.backend.simulate(records, **kwargs)
        if callable(self.backend):
            return self.backend(records, **kwargs)
        raise TypeError("simulation backend must provide simulate(...) or be callable")

    @staticmethod
    def _lookup_output(outputs: Any, record: CifMetadata, *, index: int = 0) -> Any:
        if isinstance(outputs, Mapping):
            for key in (
                record.source_id,
                record.source_path,
                str(record.source_path),
                record.source_path.stem,
            ):
                if key in outputs:
                    return outputs[key]
            raise KeyError(f"Simulation backend did not return pattern for {record.source_id!r}")
        if isinstance(outputs, (list, tuple)):
            for value in outputs:
                if isinstance(value, PxrdPattern) and value.source_id == record.source_id:
                    return value
            # Backends that return patterns in input order are accepted.
            try:
                return outputs[int(record.metadata.get("simulation_index", index))]
            except Exception as exc:
                raise KeyError(
                    f"Simulation backend did not return pattern for {record.source_id!r}"
                ) from exc
        if isinstance(outputs, np.ndarray):
            if outputs.ndim == 2 and index == 0:
                return outputs
            if outputs.ndim >= 3 and index < outputs.shape[0]:
                return outputs[index]
        raise TypeError("Simulation backend must return a mapping or sequence")

    def _simulate_gsasii(
        self,
        records: tuple[CifMetadata, ...],
        *,
        instrument_path: Path | None,
        theta_min: float,
        theta_max: float,
        n_points: int,
        theta: np.ndarray,
    ) -> Mapping[str, PxrdPattern]:
        if instrument_path is None:
            raise ValueError("instrument_path is required for the GSAS-II simulator")
        instrument_path = Path(instrument_path).expanduser().resolve()
        if instrument_path.is_dir():
            candidates = sorted(instrument_path.glob("*.instprm"))
            if not candidates:
                raise FileNotFoundError(f"No .instprm files found in {instrument_path}")
            instrument_path = candidates[0]
        if not instrument_path.is_file():
            raise FileNotFoundError(f"Instrument parameter file not found: {instrument_path}")

        outputs: dict[str, PxrdPattern] = {}
        for index, record in enumerate(records):
            # GSAS-II writes transient project files in cwd.  Keep that state
            # outside the caller's project and remove it before returning.
            with tempfile.TemporaryDirectory(prefix="ariadnex_gsas_") as scratch:
                project_name = f"library_{index}"
                try:
                    stdout_context = (
                        contextlib.redirect_stdout(io.StringIO())
                        if self.quiet
                        else contextlib.nullcontext()
                    )
                    stderr_context = (
                        contextlib.redirect_stderr(io.StringIO())
                        if self.quiet
                        else contextlib.nullcontext()
                    )
                    with stdout_context, stderr_context:
                        project = G2sc.G2Project(newgpx=str(Path(scratch) / project_name))
                        phase = project.add_phase(
                            str(record.source_path), phasename=record.source_id, fmthint="CIF"
                        )
                        hist = project.add_simulated_powder_histogram(
                            f"{record.source_id}_simulation",
                            str(instrument_path),
                            float(theta_min),
                            float(theta_max),
                            Npoints=int(n_points),
                            phases=[phase],
                            scale=self.scale,
                        )
                        project.do_refinements([hist])
                        x_values = np.ma.getdata(hist.getdata("x"))
                        y_values = np.ma.getdata(hist.getdata("ycalc"))
                    outputs[record.source_id] = _normalise_pattern(
                        record.source_id,
                        (np.asarray(x_values), np.abs(np.asarray(y_values))),
                        theta,
                        decimal_places=self.output_decimal_places,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"GSAS-II simulation failed for {record.source_id!r}"
                    ) from exc
            if (index + 1) % 25 == 0 or index + 1 == len(records):
                print(
                    f"[library] simulated patterns: {index + 1}/{len(records)}",
                    file=sys.stderr,
                    flush=True,
                )
        return outputs


__all__ = ["SimulationAdapter", "GsasIISimulator"]

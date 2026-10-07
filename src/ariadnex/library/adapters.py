"""Small adapter seams for the library construction workflow.

The production adapters have explicit protocols, while tests and downstream
callers may still inject a callable.  This module keeps that compatibility at
one seam and avoids treating a ``TypeError`` raised inside an adapter as a
signature mismatch.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact import CifMetadata, PxrdPattern
from .embedding import EmbeddingOutput
from .validation import coerce_cif_metadata


def _invoke_compatible(function: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    """Call a compatible adapter signature without catching implementation errors."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)

    accepts_arbitrary_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    filtered = (
        dict(kwargs)
        if accepts_arbitrary_keywords
        else {key: value for key, value in kwargs.items() if key in signature.parameters}
    )
    candidates = ((args, filtered), ((), kwargs), ((), filtered))
    for call_args, call_kwargs in candidates:
        try:
            signature.bind(*call_args, **call_kwargs)
        except TypeError:
            continue
        return function(*call_args, **call_kwargs)

    # Raise the useful signature error from inspect rather than invoking the
    # adapter with a shape that could hide the actual contract violation.
    signature.bind(*args, **filtered)
    raise AssertionError("unreachable")


def _adapter_method(adapter: Any, name: str) -> Any:
    method = getattr(adapter, name, None)
    if callable(method):
        return method
    if callable(adapter):
        return adapter
    raise TypeError(f"{name} adapter must provide {name}(...) or be callable")


def call_validator(validator: Any, path: Path, source_id: str) -> CifMetadata:
    method = _adapter_method(validator, "validate")
    result = _invoke_compatible(method, (path,), {"source_id": source_id})
    return coerce_cif_metadata(result, path=path, source_id=source_id)


def call_rewriter(rewriter: Any, path: Path, output_path: Path) -> Path:
    """Invoke a CIF rewriter and verify that it materialised its output."""

    method = _adapter_method(rewriter, "rewrite")
    result = _invoke_compatible(
        method, (path, output_path), {"path": path, "output_path": output_path}
    )
    resolved = output_path if result is None else Path(result).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"CIF rewriter did not produce {resolved}")
    return resolved


def call_embedder(
    embedder: Any,
    patterns: Sequence[PxrdPattern],
    *,
    theta: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    method = getattr(embedder, "embed", None)
    if callable(method):
        value = _invoke_compatible(
            method,
            (patterns,),
            {"theta": theta, "batch_size": batch_size, "device": device},
        )
    elif callable(embedder):
        # Preserve the callable injection contract: function adapters receive
        # the target theta, while object adapters may consume batching/device.
        value = _invoke_compatible(embedder, (patterns,), {"theta": theta})
    else:
        raise TypeError("embedder must provide embed(...) or be callable")

    if isinstance(value, EmbeddingOutput):
        return np.asarray(value.embeddings, dtype=np.float32), np.asarray(
            value.theta, dtype=np.float32
        )
    if isinstance(value, Mapping):
        if "embeddings" in value:
            array = np.asarray(value["embeddings"], dtype=np.float32)
            output_theta = np.asarray(value.get("theta", theta), dtype=np.float32)
            return array, output_theta
        vectors = []
        for pattern in patterns:
            vector = value.get(pattern.source_id)
            if vector is None:
                raise KeyError(f"Embedding output missing {pattern.source_id!r}")
            vectors.append(np.asarray(vector, dtype=np.float32).reshape(-1))
        return np.stack(vectors, axis=0), np.asarray(theta, dtype=np.float32)
    if isinstance(value, tuple) and len(value) == 2:
        return np.asarray(value[0], dtype=np.float32), np.asarray(
            value[1], dtype=np.float32
        ).reshape(-1)
    return np.asarray(value, dtype=np.float32), np.asarray(theta, dtype=np.float32).reshape(-1)


def call_simulator(
    simulator: Any,
    records: Sequence[CifMetadata],
    *,
    instrument_path: Path | None,
    theta_min: float,
    theta_max: float,
    n_points: int,
    output_dir: Path,
) -> dict[str, PxrdPattern]:
    method = _adapter_method(simulator, "simulate")
    value = method(
        records,
        instrument_path=instrument_path,
        theta_min=theta_min,
        theta_max=theta_max,
        n_points=n_points,
        output_dir=output_dir,
    )
    theta = np.linspace(float(theta_min), float(theta_max), int(n_points), dtype=np.float64)
    outputs: dict[str, PxrdPattern] = {}
    if isinstance(value, Mapping):
        for record in records:
            found = None
            for key in (
                record.source_id,
                record.source_path,
                str(record.source_path),
                record.source_path.stem,
            ):
                if key in value:
                    found = value[key]
                    break
            if found is None:
                raise KeyError(f"Simulation output missing {record.source_id!r}")
            outputs[record.source_id] = coerce_pattern(record.source_id, found, theta)
    else:
        array_value = np.asarray(value) if isinstance(value, np.ndarray) else None
        if array_value is not None and array_value.ndim == 2 and len(records) == 1:
            values = [array_value]
        else:
            values = list(value)
        if len(values) != len(records):
            raise ValueError(
                f"Simulation returned {len(values)} patterns for {len(records)} records"
            )
        for record, found in zip(records, values):
            outputs[record.source_id] = coerce_pattern(record.source_id, found, theta)
    return outputs


def coerce_pattern(source_id: str, value: Any, theta: np.ndarray) -> PxrdPattern:
    if isinstance(value, PxrdPattern):
        x, y, source_path = value.x, value.y, value.source_path
    elif isinstance(value, Mapping):
        x = value.get("x", value.get("theta"))
        y = value.get("y", value.get("intensity", value.get("ycalc")))
        source_path = value.get("source_path")
        if x is None or y is None:
            raise ValueError(f"Pattern output for {source_id!r} must include x and y")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        x, y = value
        source_path = None
    else:
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError(f"Unsupported pattern output for {source_id!r}")
        x, y, source_path = array[:, 0], array[:, 1], None
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.size != y_arr.size or x_arr.size < 2:
        raise ValueError(f"Pattern output for {source_id!r} has invalid x/y arrays")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError(f"Pattern output for {source_id!r} contains non-finite values")
    order = np.argsort(x_arr, kind="stable")
    x_arr, y_arr = x_arr[order], y_arr[order]
    unique_x, unique_index = np.unique(x_arr, return_index=True)
    y_arr = y_arr[unique_index]
    if unique_x.size < 2:
        raise ValueError(f"Pattern output for {source_id!r} has fewer than two unique x positions")
    tolerance = max(1e-5, float(theta[1] - theta[0]) * 1e-3)
    if (
        float(unique_x[0]) > float(theta[0]) + tolerance
        or float(unique_x[-1]) < float(theta[-1]) - tolerance
    ):
        raise ValueError(
            f"Pattern output for {source_id!r} does not cover target theta window "
            f"[{float(theta[0])}, {float(theta[-1])}]"
        )
    aligned = unique_x.size == theta.size and np.allclose(unique_x, theta, rtol=0, atol=tolerance)
    if not aligned:
        y_arr = np.interp(theta.astype(np.float64), unique_x, y_arr)
    output_x = unique_x if aligned else theta
    return PxrdPattern(source_id=source_id, x=output_x, y=y_arr, source_path=source_path)


__all__ = [
    "call_embedder",
    "call_rewriter",
    "call_simulator",
    "call_validator",
    "coerce_pattern",
]

"""Input adapters for experimental diffraction patterns.

The workflow intentionally keeps file-format handling at the mixture seam.  A
loader returns plain dictionaries containing numeric ``x``/``y`` arrays and
stable sample metadata; the cleaning and unmix stages do not need to know
whether a pattern came from an ASC, XY, CSV, NumPy, or sample-manifest file.

"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ariadnex.config import ProjectConfig

SUPPORTED_SUFFIXES = {".asc", ".txt", ".xy", ".xye", ".csv", ".npy", ".npz"}

# Manifest aliases are intentionally shared by the parser and the CSV
# detector below.  Keeping one vocabulary prevents a valid manifest such as
# ``sample,filename`` from being mistaken for a numeric two-column pattern.
SAMPLE_ID_ALIASES = ("sample_id", "sample", "id")
PATTERN_PATH_ALIASES = ("pattern_path", "pattern", "path", "file", "filename")
COMPOSITION_ALIASES = ("composition", "nominal_composition", "formula")
MANIFEST_FIELD_ALIASES = frozenset(
    (*SAMPLE_ID_ALIASES, *PATTERN_PATH_ALIASES, *COMPOSITION_ALIASES)
)


def _normalise_fields(fieldnames: Iterable[Any]) -> dict[str, Any]:
    return {
        str(field).strip().lower(): field
        for field in fieldnames
        if field is not None and str(field).strip()
    }


def csv_input_kind(path: str | Path) -> str:
    """Return ``manifest`` for a sample manifest, otherwise ``pattern``.

    A path column is the defining feature of a sample manifest.  ``pattern``
    is also a valid legacy path alias, but it is a common intensity-column
    name, so use the other column names and the first value to disambiguate it.
    Keeping this check next to the loader lets preflight use the same decision.
    """

    source = Path(path).expanduser()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        fieldnames = next(reader, [])
        first_row = next(reader, [])
    fields = _normalise_fields(fieldnames)
    strong_path_aliases = set(PATTERN_PATH_ALIASES) - {"pattern"}
    if set(fields).intersection(strong_path_aliases):
        return "manifest"
    if "pattern" not in fields:
        return "pattern"
    if set(fields).intersection(SAMPLE_ID_ALIASES):
        return "manifest"
    if set(fields).intersection({"x", "theta", "two_theta", "twotheta"}):
        return "pattern"
    pattern_index = next(
        index for index, field in enumerate(fieldnames) if str(field).strip().lower() == "pattern"
    )
    if pattern_index < len(first_row):
        try:
            float(first_row[pattern_index])
        except (TypeError, ValueError):
            return "manifest"
        return "pattern"
    return "manifest"


def _coerce_path(value: Any, *, base_dir: Path | None = None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _numeric_rows(path: Path):
    """Read two numeric columns while tolerating comments and headers."""

    with path.open(encoding="utf-8-sig") as handle:
        first_line = next(
            (line for line in handle if line.strip() and not line.lstrip().startswith("#")),
            "",
        )
    delimiter = "," if "," in first_line else None
    try:
        data = np.loadtxt(path, comments="#", delimiter=delimiter, ndmin=2, encoding="utf-8-sig")
        if data.size and data.shape[-1] >= 2:
            return np.asarray(data, dtype=float)
    except ValueError:
        pass

    # ``genfromtxt`` handles a single text header more gracefully than
    # ``loadtxt``.  A header row is removed explicitly; malformed data rows
    # are rejected by ``_validate_text_rows``.
    try:
        data = np.genfromtxt(
            path, comments="#", delimiter=delimiter, names=None, ndmin=2, encoding="utf-8-sig"
        )
    except ValueError as exc:
        raise ValueError(f"Unable to parse numeric diffraction data from {path}") from exc
    if not data.size or data.shape[-1] < 2:
        raise ValueError(f"Pattern {path} has fewer than two numeric columns")
    return _validate_text_rows(np.asarray(data, dtype=float), path)


def _validate_text_rows(data: np.ndarray, path: Path) -> np.ndarray:
    """Remove textual header rows and reject malformed data rows.

    ``genfromtxt`` represents a header as a row of NaNs.  Dropping every
    non-finite row would also turn a damaged intensity into a shorter pattern,
    changing the experiment without telling the user.  Only contiguous rows
    with no numeric x/y value at the beginning are treated as headers; any
    later non-finite x/y value is an input error.
    """

    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Pattern {path} does not have two numeric columns")
    pair_finite = np.isfinite(values[:, :2]).all(axis=1)
    first_valid = 0
    while first_valid < pair_finite.size and not pair_finite[first_valid]:
        first_valid += 1
    if first_valid:
        try:
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Unable to inspect header rows in {path}") from exc
        malformed_leading = (
            0
            if len(lines) < first_valid
            else next(
                (
                    index
                    for index, line in enumerate(lines[:first_valid])
                    if not _is_text_header(line)
                ),
                None,
            )
        )
        if malformed_leading is not None:
            raise ValueError(
                f"Pattern {path} contains non-finite x/y values on row(s) {malformed_leading + 1}"
            )
        values = values[first_valid:]
        pair_finite = pair_finite[first_valid:]
    if not pair_finite.all():
        bad_rows = np.flatnonzero(~pair_finite) + first_valid + 1
        shown = ", ".join(str(int(row)) for row in bad_rows[:8])
        suffix = " ..." if bad_rows.size > 8 else ""
        raise ValueError(f"Pattern {path} contains non-finite x/y values on row(s) {shown}{suffix}")
    if values.size == 0:
        raise ValueError(f"Pattern {path} contains no numeric data rows")
    return values


def _is_text_header(line: str) -> bool:
    """Return whether a non-comment line is a textual header."""

    fields = next(csv.reader([line]), []) if "," in line else line.split()
    if len(fields) < 2:
        try:
            float(fields[0])
        except (IndexError, TypeError, ValueError):
            return True
        return False
    parsed: list[float | None] = []
    for field in fields[:2]:
        try:
            parsed.append(float(str(field).strip()))
        except (TypeError, ValueError):
            parsed.append(None)
    return parsed[0] is None and parsed[1] is None


def _normalise_xy(x, y, *, source: Path):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.dtype.kind not in "biuf" or y.dtype.kind not in "biuf":
        raise ValueError(f"NumPy pattern {source} must contain numeric x/y arrays")
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"Pattern {source} has x/y lengths {x.size}/{y.size}")
    if x.size < 2:
        raise ValueError(f"Pattern {source} must contain at least two points")
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.all():
        bad = np.flatnonzero(~finite) + 1
        shown = ", ".join(str(int(row)) for row in bad[:8])
        suffix = " ..." if bad.size > 8 else ""
        raise ValueError(
            f"Pattern {source} contains non-finite x/y values on row(s) {shown}{suffix}"
        )
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    unique, inverse = np.unique(x, return_inverse=True)
    if unique.size != x.size:
        sums = np.zeros(unique.size, dtype=float)
        counts = np.zeros(unique.size, dtype=float)
        np.add.at(sums, inverse, y)
        np.add.at(counts, inverse, 1.0)
        y = sums / np.maximum(counts, 1.0)
        x = unique
    return x, y


def load_pattern_file(
    path: str | Path, *, sample_id: str | None = None, metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Load one diffraction pattern from a supported file.

    The returned mapping is intentionally serialisable apart from NumPy arrays:
    ``sample_id``, ``source_path``, ``composition``, ``x`` and ``y`` are the
    stable fields consumed by the rest of the workflow.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Diffraction pattern not found: {source}")
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported diffraction suffix {source.suffix!r}: {source}")

    if suffix == ".npy":
        # User-provided NumPy files are deliberately restricted to numeric
        # arrays.  Loading object arrays would deserialize arbitrary pickle
        # payloads, so never opt into pickle support at this input boundary.
        payload = np.load(source, allow_pickle=False)
        x, y = _extract_array_payload(payload, source)
    elif suffix == ".npz":
        # NpzFile keeps the archive open until it is closed.  Materialising
        # the selected arrays inside the context also ensures no descriptor
        # leaks when payload extraction raises.
        with np.load(source, allow_pickle=False) as payload:
            x, y = _extract_npz_payload(payload, source)
    else:
        rows = _numeric_rows(source)
        if rows.ndim != 2:
            raise ValueError(f"Pattern {source} is not a two-dimensional table")
        # Text patterns are column tables; XYE's third column is uncertainty.
        x, y = rows[:, 0], rows[:, 1]

    x, y = _normalise_xy(x, y, source=source)
    result: dict[str, Any] = {
        "composition": None,
        **dict(metadata or {}),
        "sample_id": str(sample_id or source.stem),
        "source_path": str(source),
        "x": x,
        "y": y,
    }
    return result


def _extract_array_payload(payload, source: Path):
    array = np.asarray(payload)
    return _extract_numeric_payload(array, source)


def _extract_numeric_payload(array, source: Path):
    array = np.asarray(array)
    if array.dtype.kind not in "biuf":
        raise ValueError(
            f"NumPy pattern {source} must contain numeric arrays (object arrays are not supported)"
        )
    if array.ndim != 2:
        raise ValueError(f"NumPy pattern {source} must have shape (N,2) or (2,N)")
    if array.shape[0] == 2 and array.shape[1] > 2:
        return array[0], array[1]
    if array.shape[1] >= 2:
        return array[:, 0], array[:, 1]
    if array.shape[0] >= 2:
        return array[0], array[1]
    raise ValueError(f"NumPy pattern {source} does not have x and y arrays")


def _extract_mapping_payload(payload: Mapping[str, Any], source: Path):
    keys = {str(k).lower(): k for k in payload}
    x_key = next((keys[k] for k in ("x", "theta", "two_theta", "twotheta") if k in keys), None)
    y_key = next((keys[k] for k in ("y", "intensity", "signal", "pattern") if k in keys), None)
    if x_key is None or y_key is None:
        raise ValueError(f"NumPy mapping {source} needs x/theta and y/intensity keys")
    return payload[x_key], payload[y_key]


def _extract_npz_payload(payload, source: Path):
    keys = list(payload.files)
    lowered = {str(k).lower(): k for k in keys}
    x_key = next(
        (lowered[k] for k in ("x", "theta", "two_theta", "twotheta") if k in lowered), None
    )
    y_key = next(
        (lowered[k] for k in ("y", "intensity", "signal", "pattern") if k in lowered), None
    )
    if x_key is not None and y_key is not None:
        return payload[x_key], payload[y_key]
    if len(keys) == 1:
        return _extract_array_payload(payload[keys[0]], source)
    if "arr_0" in keys:
        return _extract_array_payload(payload["arr_0"], source)
    raise ValueError(f"NumPy archive {source} needs x/y keys or one 2D array")


def _manifest_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Samples manifest is empty: {path}")
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        normalised = {
            str(k).strip().lower(): (v.strip() if isinstance(v, str) else v)
            for k, v in row.items()
            if k is not None
        }
        sample = next(
            (normalised.get(alias) for alias in SAMPLE_ID_ALIASES if normalised.get(alias)), None
        )
        pattern = next(
            (normalised.get(alias) for alias in PATTERN_PATH_ALIASES if normalised.get(alias)), None
        )
        if not sample:
            sample = Path(str(pattern or f"sample_{index}")).stem
        if not pattern:
            raise ValueError(
                f"Samples manifest row {index} has no pattern path; expected one of "
                f"{', '.join(PATTERN_PATH_ALIASES)}"
            )
        pattern_path = Path(str(pattern)).expanduser()
        if not pattern_path.is_absolute():
            pattern_path = path.parent / pattern_path
        records.append(
            {
                "sample_id": str(sample),
                "pattern_path": pattern_path.resolve(),
                "composition": next(
                    (
                        normalised.get(alias)
                        for alias in COMPOSITION_ALIASES
                        if normalised.get(alias)
                    ),
                    None,
                ),
                "metadata": {
                    k: v for k, v in normalised.items() if k not in MANIFEST_FIELD_ALIASES
                },
            }
        )
    return records


def discover_pattern_files(path: str | Path) -> list[Path]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Pattern directory not found: {root}")
    files = [
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda item: item.relative_to(root).as_posix().lower())


def load_mixture_records(
    source: str | Path, *, base_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    """Load all records from a directory, one file, or ``samples.csv``."""

    root = Path(source).expanduser()
    if not root.is_absolute() and base_dir is not None:
        root = Path(base_dir).expanduser() / root
    root = root.resolve()
    if root.is_dir():
        records = []
        for pattern in discover_pattern_files(root):
            records.append(load_pattern_file(pattern, sample_id=pattern.stem))
    elif root.suffix.lower() == ".csv":
        # Share the numeric-pattern/manifest decision with preflight and caching.
        if csv_input_kind(root) == "manifest":
            records = []
            manifest_rows = _manifest_records(root)
            for row in manifest_rows:
                records.append(
                    load_pattern_file(
                        row["pattern_path"],
                        sample_id=row["sample_id"],
                        metadata={"composition": row.get("composition"), **row.get("metadata", {})},
                    )
                )
        else:
            records = [load_pattern_file(root)]
    elif root.is_file():
        records = [load_pattern_file(root)]
    else:
        raise FileNotFoundError(f"Mixture input not found: {root}")

    if not records:
        raise ValueError(f"No supported diffraction files found in {root}")
    sample_ids = [str(record["sample_id"]) for record in records]
    duplicates = sorted({sample for sample in sample_ids if sample_ids.count(sample) > 1})
    if duplicates:
        raise ValueError(f"Duplicate sample_id values: {', '.join(duplicates)}")
    return records


def load_mixture_records_from_config(config: ProjectConfig) -> list[dict[str, Any]]:
    source = config.inputs.get("mixtures")
    if source is None:
        source = config.get("mixtures")
    if source is None:
        raise ValueError("Configuration needs inputs.mixtures (directory, file, or samples.csv)")
    return load_mixture_records(source, base_dir=config.base_dir)


__all__ = [
    "SUPPORTED_SUFFIXES",
    "SAMPLE_ID_ALIASES",
    "PATTERN_PATH_ALIASES",
    "COMPOSITION_ALIASES",
    "MANIFEST_FIELD_ALIASES",
    "csv_input_kind",
    "discover_pattern_files",
    "load_pattern_file",
    "load_mixture_records",
    "load_mixture_records_from_config",
]

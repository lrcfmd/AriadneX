"""Chemistry-aware candidate filtering without dataset-specific element names."""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import openpyxl

ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(?:\s*(\d+(?:\.\d+)?))?")
FORMULA_IN_NAME_RE = re.compile(r"(?:ICSD_\d+_)?['\"]?([A-Z][A-Za-z0-9().]*)['\"]?")


def normalize_sample_name(sample_name: Any) -> str | None:
    if sample_name is None:
        return None
    value = Path(str(sample_name).strip()).stem
    if not value:
        return None
    match = re.search(r"([A-Za-z]+\d+(?:[._-]\d+)?)", value)
    return match.group(1).upper() if match else value


def parse_formula_elements(formula: Any) -> set[str]:
    """Return element symbols in a nominal chemical formula.

    Parentheses and hydrate separators are intentionally tolerated; only
    element presence is needed by the prior, not stoichiometric balancing.
    """

    text = str(formula or "")
    return {symbol for symbol, _ in ELEMENT_RE.findall(text)}


def parse_nominal_composition(
    composition: Any, *, considered_elements: Sequence[str] | None = None
) -> dict[str, float]:
    values: dict[str, float] = {}
    for symbol, number in ELEMENT_RE.findall(str(composition or "")):
        values[symbol] = values.get(symbol, 0.0) + (float(number) if number else 1.0)
    if considered_elements is None:
        # Oxygen and halides are normally anionic carriers in PXRD sample
        # labels.  If no profile is provided, treat all non-O elements as
        # candidate cations while retaining the full parsed composition.
        considered = [element for element in values if element != "O"]
    else:
        considered = [str(element) for element in considered_elements]
    return {element: float(values.get(element, 0.0)) for element in considered}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    try:
        sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
        values = list(sheet.values)
        if not values:
            return []
        first = values[0]
        # The source composition workbook has a real header row (``Lable`` and
        # ``Nominal composition``), while older exports were headerless and
        # used the first two Excel columns as A/B.  Detect only known semantic
        # headers; otherwise retain every row and map columns to A/B/C... .
        known_headers = {
            "label",
            "lable",
            "sample",
            "sample id",
            "sample_id",
            "composition",
            "nominal",
            "nominal composition",
            "nominal_composition",
            "formula",
        }
        first_text = {str(value or "").strip().lower() for value in first}
        header_mode = bool(first_text & known_headers)
        if header_mode:
            headers = [str(value or "").strip() for value in first]
            rows = values[1:]
        else:
            headers = [
                chr(ord("A") + index) if index < 26 else f"col_{index + 1}"
                for index in range(len(first))
            ]
            rows = values
        return [
            {
                headers[i]: "" if value is None else str(value).strip()
                for i, value in enumerate(row)
                if i < len(headers)
            }
            for row in rows
        ]
    except Exception:
        # A small XML fallback handles workbook variants that openpyxl cannot
        # parse; the first two columns match the source composition file.
        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with zipfile.ZipFile(path) as archive:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                shared = [
                    "".join(node.text or "" for node in item.iterfind(".//a:t", ns))
                    for item in root.findall("a:si", ns)
                ]
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            relmap = {item.attrib["Id"]: item.attrib["Target"] for item in rels}
            sheet = workbook.find("a:sheets/a:sheet", ns)
            if sheet is None:
                return []
            rid = sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            sheet_path = "xl/" + relmap[rid]
            data = ET.fromstring(archive.read(sheet_path))
            rows: list[dict[str, str]] = []
            for row in data.findall(".//a:sheetData/a:row", ns):
                values: dict[str, str] = {}
                for cell in row.findall("a:c", ns):
                    ref = cell.attrib.get("r", "")
                    column = "".join(ch for ch in ref if ch.isalpha())
                    value = cell.find("a:v", ns)
                    text = "" if value is None else value.text or ""
                    if cell.attrib.get("t") == "s":
                        try:
                            text = shared[int(text)]
                        except (IndexError, ValueError):
                            pass
                    values[column] = text.strip()
                if values:
                    rows.append(values)
            return rows


def build_element_prior(
    source: str | Path | Iterable[Mapping[str, Any]],
    *,
    considered_elements: Sequence[str] | None = None,
) -> dict[str, dict[str, set[str]]]:
    """Build ``sample_id -> {present, absent, composition}`` from CSV/XLSX rows."""

    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            rows: Iterable[Mapping[str, Any]] = _read_xlsx_rows(path)
        else:
            rows = _read_csv_rows(path)
    else:
        rows = source
    priors: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        # Normalize labels so workbook headers such as ``Lable`` and
        # ``Nominal composition`` are accepted alongside machine-readable CSV
        # names and source A/B columns.
        lowered = {
            re.sub(r"[^a-z0-9]+", "", str(key).strip().lower()): value for key, value in row.items()
        }
        sample = next(
            (
                lowered.get(key)
                for key in ("sampleid", "sample", "id", "label", "lable", "samplename", "a")
                if lowered.get(key) not in (None, "")
            ),
            None,
        )
        composition = next(
            (
                lowered.get(key)
                for key in ("composition", "nominalcomposition", "formula", "nominal", "b")
                if lowered.get(key) not in (None, "")
            ),
            None,
        )
        if not sample or not composition:
            continue
        parsed = parse_nominal_composition(composition, considered_elements=considered_elements)
        considered = set(str(element) for element in (considered_elements or parsed.keys()))
        present = {element for element in considered if parsed.get(element, 0.0) > 0}
        absent = considered - present
        prior = {"present": present, "absent": absent, "composition": str(composition)}
        priors[str(sample)] = prior
        normalized = normalize_sample_name(sample)
        if normalized:
            priors[normalized] = prior
    return priors


def extract_formula_from_basis_name(name: Any) -> str | None:
    if not name:
        return None
    value = Path(str(name)).stem
    quoted = re.search(r"['\"]([^'\"]+)['\"]", value)
    if quoted:
        return quoted.group(1)
    # Prefer a contiguous segment containing at least one digit; this avoids
    # interpreting generic labels such as ``phase_001`` as formulas.
    candidates = re.findall(r"[A-Z][A-Za-z0-9().]*", value)
    return max(candidates, key=len) if candidates else None


def parse_basis_element_set(name: Any, explicit: Iterable[str] | None = None) -> set[str]:
    if explicit is not None:
        return {str(element) for element in explicit}
    return parse_formula_elements(extract_formula_from_basis_name(name))


def parse_basis_elements(
    names: Sequence[Any], explicit: Sequence[Iterable[str] | None] | None = None
) -> list[set[str]]:
    return [
        parse_basis_element_set(
            name, explicit[index] if explicit is not None and index < len(explicit) else None
        )
        for index, name in enumerate(names)
    ]


def filter_basis_indices(
    names: Sequence[Any], element_sets: Sequence[set[str]], prior: Mapping[str, Any] | None
):
    if len(names) != len(element_sets):
        raise ValueError("names and element_sets must contain the same number of basis records")
    if not prior:
        return list(range(len(names))), []
    absent = set(prior.get("absent", set()))
    keep: list[int] = []
    removed: list[tuple[Any, set[str]]] = []
    for index, (name, elements) in enumerate(zip(names, element_sets)):
        overlap = set(elements) & absent
        if overlap:
            removed.append((name, overlap))
        else:
            keep.append(index)
    return keep, removed


def resolve_sample_max_base(
    setting: Any,
    prior: Mapping[str, Any] | None,
    *,
    default_elements: Sequence[str] | None = None,
    max_base_extra: int = 0,
    n_candidates: int = 0,
) -> int:
    if setting is None or (isinstance(setting, str) and setting.lower() == "auto"):
        if prior:
            count = len(set(prior.get("present", set())))
        elif default_elements:
            count = len(default_elements)
        else:
            # Without composition information every candidate remains eligible.
            count = int(n_candidates)
        return count + int(max_base_extra)
    return max(0, int(setting))


def check_element_coverage(
    element_sets: Sequence[set[str]],
    selected_indices: Sequence[int],
    prior: Mapping[str, Any] | None,
):
    if not prior:
        return set(), set(), set()
    selected = (
        set().union(*(set(element_sets[index]) for index in selected_indices))
        if selected_indices
        else set()
    )
    present = set(prior.get("present", set()))
    absent = set(prior.get("absent", set()))
    return present & selected, present - selected, absent & selected


def format_removed_phases(removed: Sequence[tuple[Any, set[str]]]) -> list[str]:
    return [
        f"{name or '[unknown basis]'} (contains: {', '.join(sorted(elements))})"
        for name, elements in removed
    ]


__all__ = [
    "normalize_sample_name",
    "parse_formula_elements",
    "parse_nominal_composition",
    "build_element_prior",
    "extract_formula_from_basis_name",
    "parse_basis_elements",
    "filter_basis_indices",
    "resolve_sample_max_base",
    "check_element_coverage",
    "format_removed_phases",
]

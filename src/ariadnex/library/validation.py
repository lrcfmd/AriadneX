"""CIF validation and metadata extraction using the project environment.

The single AriadneX environment provides pymatgen and SMACT. Adapter
protocols keep validation and CIF rewriting replaceable where needed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from math import cos, pi, sqrt
from pathlib import Path
from typing import Any, Protocol

from pymatgen.core import Composition
from pymatgen.io.cif import CifFile, CifParser, CifWriter

from .artifact import CifMetadata


class CifValidator(Protocol):
    """Adapter seam for structural CIF validation."""

    def validate(self, path: Path, *, source_id: str | None = None) -> CifMetadata: ...


class CifRewriter(Protocol):
    """Adapter seam for materialising GSAS-II-compatible CIF files."""

    def rewrite(self, path: Path, output_path: Path) -> Path: ...


_FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def source_id_from_path(path: Path, *, existing: set[str] | None = None) -> str:
    """Return a stable ID, disambiguating duplicate stems deterministically."""

    base = path.stem
    if existing is None or base not in existing:
        return base
    # Avoid hashing machine-specific absolute paths.  Parent-relative names are
    # not available here, so use the filename suffix deterministically.
    digest = 0
    for byte in path.name.encode("utf-8"):
        digest = (digest * 131 + byte) % (2**32)
    candidate = f"{base}__{digest:08x}"
    counter = 1
    while candidate in existing:
        candidate = f"{base}__{digest:08x}_{counter}"
        counter += 1
    return candidate


def _tag_values(path: Path) -> dict[str, str]:
    """Read declared values before structure parsing can rewrite their notation."""

    blocks = CifFile.from_file(path).data
    if len(blocks) != 1:
        raise ValueError(f"Expected one CIF data block, found {len(blocks)} in {path}")
    name, block = next(iter(blocks.items()))
    values = {
        str(key).lower(): str(value).strip()
        for key, value in block.data.items()
        if not isinstance(value, list)
    }
    values["__block_name__"] = str(name)
    return values


def _first_tag(values: dict[str, str], *names: str) -> str:
    for name in names:
        value = values.get(name.lower())
        if value and value not in {"?", "."}:
            return value
    return ""


def _parse_float(value: str) -> float | None:
    match = _FLOAT_PATTERN.search(str(value)) if value else None
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _source_metadata(path: Path, tags: dict[str, str]) -> dict[str, Any]:
    """Keep source labels separate from the structure used for simulation."""

    accession = ""
    for database, names in (
        ("ICSD", ("_database_code_icsd",)),
        ("COD", ("_cod_database_code", "_database_code_cod")),
        ("CCDC", ("_database_code_depnum_ccdc_archive", "_database_code_ccdc")),
    ):
        code = _first_tag(tags, *names)
        if code:
            accession = f"{database}_{code}"
            break
    number = _parse_float(
        _first_tag(
            tags, "_space_group_it_number", "_symmetry_int_tables_number", "_space_group.it_number"
        )
    )
    source_number = (
        int(number) if number is not None and number.is_integer() and 1 <= number <= 230 else None
    )
    return {
        "source_accession": accession,
        "source_digest": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_block": tags.get("__block_name__", ""),
        "source_formula": _first_tag(
            tags, "_chemical_formula_structural", "_chemical_formula_sum", "_chemical_formula.sum"
        ),
        "source_formula_sum": _first_tag(tags, "_chemical_formula_sum", "_chemical_formula.sum"),
        "source_space_group": _first_tag(
            tags,
            "_space_group_name_h-m_alt",
            "_symmetry_space_group_name_h-m",
            "_space_group.name_h-m_alt",
        ),
        "source_space_group_number": source_number,
        **_text_metadata(tags),
    }


def _elements_from_formula(formula: str) -> tuple[str, ...]:
    if not formula:
        return ()
    # Element symbols are sufficient for filter policy.  This deliberately
    # ignores charge/isotope decorations and does not attempt stoichiometry.
    return tuple(sorted(set(re.findall(r"[A-Z][a-z]?", formula))))


def _atom_symbols(path: Path) -> tuple[str, ...]:
    """Best-effort extraction of symbols from a CIF atom loop."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    symbols: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip().lower() != "loop_":
            index += 1
            continue
        index += 1
        headers: list[str] = []
        while index < len(lines) and lines[index].strip().startswith("_"):
            headers.append(lines[index].strip().split()[0].lower())
            index += 1
        if not headers:
            continue
        symbol_index = next(
            (
                headers.index(header)
                for header in (
                    "_atom_site_type_symbol",
                    "_atom_site_symbol",
                    "_atom_site_label",
                )
                if header in headers
            ),
            None,
        )
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#"):
                index += 1
                if not stripped:
                    break
                continue
            if (
                stripped.lower() == "loop_"
                or stripped.startswith("_")
                or stripped.lower().startswith("data_")
            ):
                break
            fields = stripped.split()
            if symbol_index is not None and len(fields) > symbol_index:
                token = fields[symbol_index].strip("'\"")
                match = re.match(r"([A-Z][a-z]?)", token)
                if match:
                    symbols.append(match.group(1))
            index += 1
        if symbols:
            return tuple(sorted(set(symbols)))
    return ()


def _volume_from_tags(tags: dict[str, str]) -> float | None:
    direct = _parse_float(_first_tag(tags, "_cell_volume", "_cell.volume"))
    if direct is not None:
        return direct
    lengths = [_parse_float(_first_tag(tags, f"_cell_length_{axis}")) for axis in "abc"]
    angles = [
        _parse_float(_first_tag(tags, f"_cell_angle_{axis}")) for axis in "alpha beta gamma".split()
    ]
    if any(value is None for value in (*lengths, *angles)):
        return None
    a, b, c = (float(value) for value in lengths)  # type: ignore[arg-type]
    alpha, beta, gamma = (float(value) * pi / 180.0 for value in angles)  # type: ignore[arg-type]
    value = (
        a
        * b
        * c
        * sqrt(
            1.0
            - cos(alpha) ** 2
            - cos(beta) ** 2
            - cos(gamma) ** 2
            + 2.0 * cos(alpha) * cos(beta) * cos(gamma)
        )
    )
    return float(value) if value > 0 else None


def _estimate_sites_from_formula(formula: str) -> int | None:
    """Estimate atom count from the declared formula before expanding sites."""

    if not formula:
        return None
    try:
        return int(round(float(Composition(formula).num_atoms)))
    except Exception:
        return None


class PymatgenCifValidator:
    """Validate CIFs with pymatgen and a text-level parse fallback."""

    def __init__(
        self,
        *,
        occupancy_tolerance: float = 100.0,
        parse_primitive: bool = True,
        max_sites: int | None = None,
    ) -> None:
        self.occupancy_tolerance = float(occupancy_tolerance)
        self.parse_primitive = bool(parse_primitive)
        self.max_sites = None if max_sites is None else int(max_sites)

    def validate(self, path: Path, *, source_id: str | None = None) -> CifMetadata:
        path = Path(path).expanduser().resolve()
        resolved_id = source_id or path.stem
        if not path.is_file():
            raise FileNotFoundError(f"CIF file not found: {path}")
        tags = _tag_values(path)
        source_metadata = _source_metadata(path, tags)
        source_formula = source_metadata["source_formula"]
        source_space_group = source_metadata["source_space_group"]

        # Reject oversized declared formulas before expanding their atom sites.
        estimated_sites = _estimate_sites_from_formula(source_formula)
        if (
            self.max_sites is not None
            and estimated_sites is not None
            and estimated_sites > self.max_sites
        ):
            return CifMetadata(
                source_id=resolved_id,
                source_path=path,
                parse_status="pass",
                formula=_first_tag(tags, "_chemical_formula_sum", "_chemical_formula_structural")
                or source_formula,
                elements=_elements_from_formula(source_formula),
                space_group=_first_tag(
                    tags, "_symmetry_space_group_name_h-m", "_space_group_name_h-m_alt"
                )
                or source_space_group,
                space_group_number=source_metadata["source_space_group_number"],
                cell_volume=_volume_from_tags(tags),
                n_sites=estimated_sites,
                metadata={
                    **source_metadata,
                    "oversized_estimate": True,
                },
            )

        try:
            parser = CifParser(
                str(path),
                occupancy_tolerance=self.occupancy_tolerance,
                check_cif=True,
            )
            structures = parser.parse_structures(
                primitive=self.parse_primitive,
                on_error="raise",
            )
            if not structures:
                raise ValueError("no structure parsed from CIF")
            structure = structures[0]
            composition = structure.composition
            formula = str(getattr(composition, "reduced_formula", "") or composition.formula)
            elements = tuple(sorted(str(element.symbol) for element in composition.elements))
            sg = ""
            sg_number = None
            space_group_analyzed = True
            try:
                sg, sg_number = structure.get_space_group_info()
            except Exception:
                space_group_analyzed = False
                sg = source_space_group
                sg_number = source_metadata["source_space_group_number"]
            volume = float(structure.volume)
            n_sites = int(len(structure))
            return CifMetadata(
                source_id=resolved_id,
                source_path=path,
                parse_status="pass",
                formula=formula or source_formula,
                elements=elements or _elements_from_formula(source_formula),
                space_group=str(sg or source_space_group),
                space_group_number=sg_number,
                cell_volume=volume,
                n_sites=n_sites,
                metadata={
                    **source_metadata,
                    "space_group_analyzed": space_group_analyzed,
                    "composition": {
                        str(getattr(element, "symbol", element)): float(amount)
                        for element, amount in composition.get_el_amt_dict().items()
                    },
                },
            )
        except Exception as parser_error:
            # Pymatgen is intentionally strict about incomplete atom loops.
            # Try the text-level validator before surfacing that parser
            # error; malformed files still fail the fallback's data/formula
            # checks and retain the useful original exception.
            try:
                return self._validate_text(path, resolved_id, tags, source_metadata)
            except Exception:
                raise parser_error

    def _validate_text(
        self,
        path: Path,
        source_id: str,
        tags: dict[str, str],
        source_metadata: dict[str, Any],
    ) -> CifMetadata:
        formula = (
            _first_tag(tags, "_chemical_formula_sum", "_chemical_formula_structural")
            or source_metadata["source_formula"]
        )
        atom_elements = _atom_symbols(path)
        elements = atom_elements or _elements_from_formula(formula)
        volume = _volume_from_tags(tags)
        if not formula and not elements:
            raise ValueError("CIF contains no chemical formula or atom-site symbols")
        # A data block and either a cell or atom table are the minimum sanity
        # checks available without a full CIF parser.
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"^\s*data_", text, flags=re.IGNORECASE | re.MULTILINE):
            raise ValueError("CIF is missing a data_ block")
        if not atom_elements and volume is None:
            raise ValueError("CIF fallback lacks atom-site symbols and cell volume")
        return CifMetadata(
            source_id=source_id,
            source_path=path,
            parse_status="pass",
            formula=formula,
            elements=elements,
            space_group=_first_tag(
                tags, "_symmetry_space_group_name_h-m", "_space_group_name_h-m_alt"
            )
            or source_metadata["source_space_group"],
            space_group_number=source_metadata["source_space_group_number"],
            cell_volume=volume,
            n_sites=None,
            metadata={**source_metadata, "space_group_analyzed": False},
        )


class OversizedCifError(ValueError):
    """Raised when a CIF exceeds the configured site-count limit."""


class PymatgenCifRewriter:
    """Rewrite a parsed CIF with pymatgen's GSAS-II-friendly ``CifWriter``.

    The historical upstream pipeline parsed raw ICSD files into primitive
    structures and then wrote a clean CIF before simulation.  Keeping that
    operation as a separate adapter lets production builds retain the same
    boundary while callers can inject another compatible rewriter.
    """

    def __init__(
        self,
        *,
        primitive: bool = True,
        occupancy_tolerance: float = 100.0,
        symprec: float | None = None,
        angle_tolerance: float = 5.0,
        max_sites: int | None = None,
    ) -> None:
        self.primitive = bool(primitive)
        self.occupancy_tolerance = float(occupancy_tolerance)
        self.symprec = symprec
        self.angle_tolerance = float(angle_tolerance)
        self.max_sites = None if max_sites is None else int(max_sites)

    def rewrite(self, path: Path, output_path: Path) -> Path:
        path = Path(path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CIF file not found: {path}")

        formula = _first_tag(
            _tag_values(path), "_chemical_formula_structural", "_chemical_formula_sum"
        )
        estimated_sites = _estimate_sites_from_formula(formula)
        if (
            self.max_sites is not None
            and estimated_sites is not None
            and estimated_sites > self.max_sites
        ):
            raise OversizedCifError(f"{estimated_sites} sites > {self.max_sites}")

        parser = CifParser(
            str(path),
            occupancy_tolerance=self.occupancy_tolerance,
        )
        structures = parser.parse_structures(
            primitive=self.primitive,
            on_error="raise",
        )
        if not structures:
            raise ValueError("no structure parsed from CIF")
        structure = structures[0]
        n_sites = int(len(structure))
        if self.max_sites is not None and n_sites > self.max_sites:
            raise OversizedCifError(f"{n_sites} sites > {self.max_sites}")

        writer = CifWriter(
            structure,
            symprec=self.symprec,
            write_magmoms=False,
            significant_figures=8,
            angle_tolerance=self.angle_tolerance,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(writer), encoding="utf-8")
        return output_path


def _text_metadata(tags: dict[str, str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    temperature_raw = _first_tag(
        tags,
        "_diffrn_ambient_temperature",
        "_diffrn_ambient_temp",
        "_diffrn_ambient_temperature_gt",
        "_diffrn_ambient_temperature_lt",
        "_cell_measurement_temperature",
    )
    pressure_raw = _first_tag(
        tags,
        "_diffrn_ambient_pressure",
        "_diffrn_ambient_pressure_gt",
        "_diffrn_ambient_pressure_lt",
        "_cell_measurement_pressure",
    )
    metadata["ambient_temperature_raw"] = temperature_raw
    metadata["ambient_temperature_kelvin"] = _parse_float(temperature_raw)
    metadata["ambient_pressure_raw"] = pressure_raw
    pressure_value = _parse_float(pressure_raw)
    # CIF values in this family are conventionally kPa.  Explicit units are
    # handled for portable user datasets (GPa, MPa, kPa and Pa).
    pressure_lower = pressure_raw.lower()
    if pressure_value is None:
        pressure_gpa = None
    elif "gpa" in pressure_lower:
        pressure_gpa = pressure_value
    elif "mpa" in pressure_lower:
        pressure_gpa = pressure_value / 1e3
    elif "kpa" in pressure_lower:
        pressure_gpa = pressure_value / 1e6
    elif "pa" in pressure_lower:
        pressure_gpa = pressure_value / 1e9
    else:
        pressure_gpa = pressure_value / 1e6
    metadata["ambient_pressure_gpa"] = pressure_gpa
    metadata["audit_creation_date"] = _first_tag(tags, "_audit_creation_date")
    metadata["metadata_phase_values"] = {
        key: value
        for key, value in tags.items()
        if key in {"_chemical_name_common", "_chemical_name_systematic"}
    }
    return metadata


DefaultCifValidator = PymatgenCifValidator


def coerce_cif_metadata(value: Any, *, path: Path, source_id: str) -> CifMetadata:
    """Normalise adapter outputs into :class:`CifMetadata`."""

    if isinstance(value, CifMetadata):
        if value.source_id == source_id and value.source_path == path:
            return value
        return replace(value, source_id=source_id, source_path=path)
    if isinstance(value, dict):
        fields = dict(value)
        # The discovery layer owns identity and provenance; an adapter cannot
        # accidentally redirect a row to a different path or ID.
        fields["source_id"] = source_id
        fields["source_path"] = path
        if "elements" in fields and isinstance(fields["elements"], str):
            fields["elements"] = tuple(re.findall(r"[A-Z][a-z]?", fields["elements"]))
        return CifMetadata(**fields)
    raise TypeError(f"CIF validator returned unsupported value: {type(value)!r}")

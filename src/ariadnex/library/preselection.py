"""Optional chemistry-level preselection for candidate phase libraries.

The open-world workflow uses SMACT after CIF parsing and before
formula/space-group/volume deduplication.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Any, Mapping

from pymatgen.core import Composition
from smact.screening import smact_validity

from .artifact import CifMetadata


@dataclass(frozen=True)
class PreselectionProfile:
    """SMACT integerisation and validity settings."""

    enabled: bool = False
    smact_mixed_valence: bool = True
    smact_use_pauling_test: bool = False
    smact_include_alloys: bool = False
    smact_fractional_max_denominator: int = 32
    smact_fractional_tolerance: float = 1e-3
    integer_snap_tolerance: float = 2e-3
    simple_ratio_snap_max_coefficient: int = 8
    simple_ratio_snap_relative_tolerance: float = 5e-3
    smact_skip_unstable_fractional_formula: bool = True


def _value(profile: PreselectionProfile | Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(profile, Mapping):
        return profile.get(key, default)
    return getattr(profile, key, default)


def _lcm(left: int, right: int) -> int:
    return abs(left * right) // gcd(left, right)


def _snap_near_integer_amounts(composition: Any, tolerance: float) -> Any:
    snapped: dict[str, float] = {}
    for element, amount in composition.get_el_amt_dict().items():
        nearest = round(float(amount))
        snapped[element] = (
            float(nearest) if abs(float(amount) - nearest) <= tolerance else float(amount)
        )
    return Composition(snapped)


def _has_non_integer_amounts(composition: Any) -> bool:
    return any(
        abs(float(amount) - round(float(amount))) > 1e-12
        for amount in composition.get_el_amt_dict().values()
    )


def _simple_ratio_snap(
    composition: Any, *, max_coefficient: int, relative_tolerance: float
) -> tuple[Any | None, str]:
    if max_coefficient < 1 or relative_tolerance <= 0:
        return None, "simple_ratio_snap_disabled"
    amounts = {
        str(element): float(amount)
        for element, amount in composition.get_el_amt_dict().items()
        if float(amount) > 0
    }
    if not amounts:
        return None, "simple_ratio_snap_no_positive_amounts"

    best: tuple[tuple[int, float, tuple[tuple[str, int], ...]], dict[str, int], float] | None = None
    for _, anchor_amount in sorted(amounts.items()):
        for anchor_coefficient in range(1, int(max_coefficient) + 1):
            scale = anchor_coefficient / anchor_amount
            coefficients: dict[str, int] = {}
            max_error = 0.0
            valid = True
            for element, amount in sorted(amounts.items()):
                coefficient = round(amount * scale)
                if coefficient < 1 or coefficient > max_coefficient:
                    valid = False
                    break
                reconstructed = coefficient / scale
                error = abs(reconstructed - amount) / abs(amount)
                if error > relative_tolerance:
                    valid = False
                    break
                coefficients[element] = int(coefficient)
                max_error = max(max_error, error)
            if not valid:
                continue
            divisor = reduce(gcd, coefficients.values())
            normalized = {
                element: coefficient // divisor for element, coefficient in coefficients.items()
            }
            if max(normalized.values()) > max_coefficient:
                continue
            ranking = (sum(normalized.values()), max_error, tuple(sorted(normalized.items())))
            if best is None or ranking < best[0]:
                best = (ranking, normalized, max_error)
    if best is None:
        return None, (
            "simple_ratio_snap_not_applicable"
            f"(max_coefficient={max_coefficient}, relative_tolerance={relative_tolerance})"
        )
    _, coefficients, max_error = best
    snapped = Composition(coefficients)
    return snapped, (
        "simple_ratio_snapped"
        f"(formula={snapped.formula.replace(' ', '')}, max_coefficient={max_coefficient}, "
        f"relative_tolerance={relative_tolerance}, max_relative_error={max_error:.6g})"
    )


def _integerize(
    composition: Any, profile: PreselectionProfile | Mapping[str, Any]
) -> tuple[Any | None, str]:
    snapped = _snap_near_integer_amounts(
        composition, float(_value(profile, "integer_snap_tolerance", 2e-3))
    )
    if _has_non_integer_amounts(snapped):
        ratio_snapped, ratio_reason = _simple_ratio_snap(
            snapped,
            max_coefficient=int(_value(profile, "simple_ratio_snap_max_coefficient", 8)),
            relative_tolerance=float(_value(profile, "simple_ratio_snap_relative_tolerance", 5e-3)),
        )
        if ratio_snapped is not None:
            return ratio_snapped, ratio_reason

    amounts = snapped.get_el_amt_dict()
    chosen: dict[str, Fraction] | None = None
    chosen_limit: int | None = None
    fallback_details: list[str] = []
    max_denominator = int(_value(profile, "smact_fractional_max_denominator", 32))
    tolerance = float(_value(profile, "smact_fractional_tolerance", 1e-3))
    for denominator_limit in range(1, max_denominator + 1):
        fractions: dict[str, Fraction] = {}
        details: list[str] = []
        valid = True
        for element, amount in amounts.items():
            rational = Fraction(float(amount)).limit_denominator(denominator_limit)
            error = abs(float(rational) - float(amount))
            details.append(
                f"{element}:{amount:.6f}->{float(rational):.6f}(d<={denominator_limit},err={error:.6g})"
            )
            if error > tolerance:
                valid = False
                break
            fractions[element] = rational
        if valid:
            chosen = fractions
            chosen_limit = denominator_limit
            break
        fallback_details = details
    if chosen is None or chosen_limit is None:
        return None, (
            "fractional_formula_unstable_integerization"
            f"(max_denominator={max_denominator}, tolerance={tolerance}, "
            f"integer_snap_tolerance={float(_value(profile, 'integer_snap_tolerance', 2e-3))}, "
            f"sample={'; '.join(fallback_details[:3])})"
        )
    common_denominator = reduce(_lcm, (item.denominator for item in chosen.values()), 1)
    integer_amounts = {
        element: int(fraction * common_denominator) for element, fraction in chosen.items()
    }
    divisor = reduce(gcd, integer_amounts.values())
    normalized = {element: amount // divisor for element, amount in integer_amounts.items()}
    return Composition(normalized), (
        f"integerized(min_denominator_limit={chosen_limit}, common_denominator={common_denominator}, "
        f"integer_snap_tolerance={float(_value(profile, 'integer_snap_tolerance', 2e-3))})"
    )


def evaluate_smact(
    record: CifMetadata, profile: PreselectionProfile | Mapping[str, Any]
) -> tuple[bool, str]:
    """Return ``(keep, reason)`` for one parsed record.

    Missing composition metadata is treated as an explicit skip, which keeps
    the adapter usable with injected validators. When enabled, the SMACT
    validity result determines library membership.
    """

    if not bool(_value(profile, "enabled", _value(profile, "smact_enabled", False))):
        return True, "disabled_by_config"
    composition_data = (record.metadata or {}).get("composition")
    if not composition_data:
        return True, "skipped_no_composition_metadata"
    composition = Composition(composition_data)
    integerized, integerization_reason = _integerize(composition, profile)
    if integerized is None:
        if bool(_value(profile, "smact_skip_unstable_fractional_formula", True)):
            return True, f"skipped: {integerization_reason}"
        return False, f"invalid_fractional_formula:{integerization_reason}"
    kwargs = {
        "use_pauling_test": bool(_value(profile, "smact_use_pauling_test", False)),
        "include_alloys": bool(_value(profile, "smact_include_alloys", False)),
    }
    if "mixed_valence" in inspect.signature(smact_validity).parameters:
        kwargs["mixed_valence"] = bool(_value(profile, "smact_mixed_valence", True))
    if bool(smact_validity(integerized, **kwargs)):
        return True, f"pass_{integerization_reason}" if integerization_reason != "pass" else "pass"
    return False, "fail"


__all__ = ["PreselectionProfile", "evaluate_smact"]

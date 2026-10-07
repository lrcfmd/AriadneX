"""Match recorded phase identities to formula labels in report references."""

from __future__ import annotations

import math
import re
from typing import Mapping

from .models import _ResultData

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")
_FORMULA_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")


def _formula_component(text: str) -> dict[str, float] | None:
    """Parse one formula component, including nested groups and a coefficient."""

    coefficient = _FORMULA_NUMBER.match(text)
    factor = float(coefficient.group()) if coefficient else 1.0
    start = coefficient.end() if coefficient else 0

    def group(position: int, *, closing: bool = False) -> tuple[dict[str, float], int] | None:
        amounts: dict[str, float] = {}
        while position < len(text):
            if text[position] == ")":
                return (amounts, position + 1) if closing and amounts else None
            if text[position] == "(":
                parsed = group(position + 1, closing=True)
                if parsed is None:
                    return None
                atoms, position = parsed
                multiplier = _FORMULA_NUMBER.match(text, position)
                scale = float(multiplier.group()) if multiplier else 1.0
                if multiplier:
                    position = multiplier.end()
                for element, amount in atoms.items():
                    amounts[element] = amounts.get(element, 0.0) + amount * scale
            else:
                atom = _FORMULA_TOKEN.match(text, position)
                if atom is None:
                    return None
                element, amount = atom.groups()
                amounts[element] = amounts.get(element, 0.0) + float(amount or 1.0)
                position = atom.end()
        return None if closing else (amounts, position)

    parsed = group(start)
    if parsed is None or not parsed[0]:
        return None
    return {element: amount * factor for element, amount in parsed[0].items()}


def formula_key(value: object) -> tuple[tuple[str, float], ...] | None:
    """Return elemental amounts for a formula or a quoted ICSD-style ID.

    Middle dots separate hydrate components: the coefficient in ``5H2O``
    multiplies both H and O. Decimal points remain part of atom counts.
    """

    text = str(value or "").strip()
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
    for candidate in quoted or [text]:
        candidate = candidate.replace(" ", "").replace("[", "(").replace("]", ")")
        amounts: dict[str, float] = {}
        for component in candidate.split("·"):
            parsed = _formula_component(component)
            if parsed is None:
                break
            for element, amount in parsed.items():
                amounts[element] = amounts.get(element, 0.0) + amount
        else:
            if amounts and all(math.isfinite(amount) for amount in amounts.values()):
                return tuple(
                    sorted((element, round(amount, 8)) for element, amount in amounts.items())
                )
    return None


def phase_formula_key(data: _ResultData, phase_id: str) -> tuple[tuple[str, float], ...] | None:
    """Prefer recorded CIF composition over display labels and legacy filenames."""

    recorded = data.phase_formula_keys.get(phase_id)
    if recorded is not None:
        return recorded
    return formula_key(data.labels.get(phase_id)) or formula_key(phase_id)


def truth_phase_amount(
    sample_truth: Mapping[str, float],
    phase_id: str,
    data: _ResultData,
) -> float:
    """Resolve a reference amount consistently for report metrics and plot labels."""

    if phase_id in sample_truth:
        return float(sample_truth[phase_id])
    key = phase_formula_key(data, phase_id)
    if key is None:
        return 0.0
    target = data.phase_aliases.get(key)
    if target in sample_truth:
        return float(sample_truth[target])
    return sum(
        float(amount)
        for truth_phase, amount in sample_truth.items()
        if phase_formula_key(data, truth_phase) == key
    )

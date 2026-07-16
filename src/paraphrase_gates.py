"""Deterministic semantic-preservation gates used before the independent judge."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?")
MATH_SYMBOL_RE = re.compile(r"(?:<=|>=|!=|==|[=<>±×÷+−*/^%])")
WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")

# Deliberately conservative. A false positive causes regeneration, not a result
# change. Numbers and symbolic expressions are checked separately.
UNIT_WORDS = {
    "amp",
    "amps",
    "cent",
    "cents",
    "centimeter",
    "centimeters",
    "cm",
    "day",
    "days",
    "degree",
    "degrees",
    "dollar",
    "dollars",
    "feet",
    "foot",
    "g",
    "gallon",
    "gallons",
    "gram",
    "grams",
    "hour",
    "hours",
    "inch",
    "inches",
    "kg",
    "kilogram",
    "kilograms",
    "kilometer",
    "kilometers",
    "km",
    "lb",
    "lbs",
    "liter",
    "liters",
    "m",
    "meter",
    "meters",
    "mile",
    "miles",
    "milliliter",
    "milliliters",
    "minute",
    "minutes",
    "ml",
    "month",
    "months",
    "ounce",
    "ounces",
    "percent",
    "percentage",
    "pound",
    "pounds",
    "second",
    "seconds",
    "week",
    "weeks",
    "yard",
    "yards",
    "year",
    "years",
}


def normalized_numbers(text: str) -> Counter[str]:
    normalized = []
    for token in NUMBER_RE.findall(text):
        value = token.replace(",", "")
        if value.startswith("+"):
            value = value[1:]
        normalized.append(value)
    return Counter(normalized)


def math_symbols(text: str) -> Counter[str]:
    return Counter(MATH_SYMBOL_RE.findall(text))


def units(text: str) -> Counter[str]:
    return Counter(word.lower() for word in WORD_RE.findall(text) if word.lower() in UNIT_WORDS)


def normalized_edit_distance(left: str, right: str) -> float:
    return (
        1.0
        - SequenceMatcher(
            None, " ".join(left.lower().split()), " ".join(right.lower().split())
        ).ratio()
    )


def deterministic_gates(
    original: str, paraphrase: str, minimum_edit_distance: float
) -> dict[str, Any]:
    clean = paraphrase.strip().strip('"').strip()
    number_match = normalized_numbers(original) == normalized_numbers(clean)
    symbol_match = math_symbols(original) == math_symbols(clean)
    unit_match = units(original) == units(clean)
    edit_distance = normalized_edit_distance(original, clean)
    nonempty = bool(clean)
    passed = (
        nonempty
        and number_match
        and symbol_match
        and unit_match
        and edit_distance >= minimum_edit_distance
    )
    return {
        "passed": passed,
        "nonempty": nonempty,
        "number_multiset_match": number_match,
        "math_symbol_multiset_match": symbol_match,
        "unit_multiset_match": unit_match,
        "normalized_edit_distance": round(edit_distance, 6),
        "minimum_edit_distance": minimum_edit_distance,
    }

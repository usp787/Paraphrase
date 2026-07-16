"""Three-layer math answer extraction and equivalence helpers."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

FINAL_PATTERNS = [
    re.compile(r"\\boxed\s*\{", re.IGNORECASE),
    re.compile(r"final\s+answer\s*(?:is|:|=)\s*", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|:|=)\s*", re.IGNORECASE),
]
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?")


def extract_braced(text: str, opening: int) -> str | None:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index].strip()
    return None


def extract_raw_answer(text: str, dataset: str) -> tuple[str | None, str]:
    boxed_starts = list(re.finditer(r"\\boxed\s*\{", text, flags=re.IGNORECASE))
    if boxed_starts:
        opening = text.find("{", boxed_starts[-1].start())
        boxed = extract_braced(text, opening)
        if boxed is not None:
            return boxed, "boxed"
    matches = list(re.finditer(r"(?:final\s+)?answer\s*(?:is|:|=)\s*([^\n]+)", text, re.IGNORECASE))
    if matches:
        candidate = matches[-1].group(1).strip().rstrip(". ")
        return candidate, "answer_phrase"
    if dataset == "gsm_symbolic":
        numbers = NUMBER_RE.findall(text)
        if numbers:
            return numbers[-1], "last_number_fallback"
    return None, "unparsed"


def normalize_answer(answer: Any) -> str | None:
    if answer is None:
        return None
    value = str(answer).strip()
    value = re.sub(r"^\$|\$$", "", value)
    value = value.replace(r"\,", "").replace(",", "")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(". ")
    if value.endswith("%"):
        try:
            return str(Decimal(value[:-1]) / Decimal(100))
        except InvalidOperation:
            pass
    try:
        if "/" in value and re.fullmatch(r"[-+]?\d+/\d+", value):
            fraction = Fraction(value)
            return f"{fraction.numerator}/{fraction.denominator}"
        decimal = Decimal(value)
        rendered = format(decimal.normalize(), "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return rendered or "0"
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return value


def canonical_equivalent(
    gold: str, completion: str, extracted: str | None
) -> tuple[bool | None, str]:
    try:
        from math_verify import parse, verify
    except ImportError:
        # Dependency-light fallback is useful for local tests, but confirmatory
        # preflight requires the Apptainer lock that includes math-verify.
        left, right = normalize_answer(gold), normalize_answer(extracted)
        return ((left == right) if right is not None else None), "fallback_normalized"
    try:
        gold_parsed = parse(gold)
        # Parsing the full response retains context for boxed LaTeX. If that is
        # undecidable, try only the extracted final answer.
        predicted = parse(completion)
        if verify(gold_parsed, predicted):
            return True, "math_verify_full"
        if extracted is not None:
            return bool(verify(gold_parsed, parse(extracted))), "math_verify_extracted"
        return False, "math_verify_full"
    except Exception as exc:  # noqa: BLE001
        return None, f"math_verify_error:{type(exc).__name__}"


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    raw, extraction = extract_raw_answer(record["raw_output"], record["dataset"])
    normalized_pred = normalize_answer(raw)
    normalized_gold = normalize_answer(record["canonical_answer"])
    raw_match = raw is not None and raw.strip() == str(record["canonical_answer"]).strip()
    normalized_match = normalized_pred is not None and normalized_pred == normalized_gold
    canonical_match, canonical_status = canonical_equivalent(
        str(record["canonical_answer"]), record["raw_output"], raw
    )
    decided = [raw_match, normalized_match]
    if canonical_match is not None:
        decided.append(canonical_match)
    disagreement = len(set(decided)) > 1
    judge_required = canonical_match is None or disagreement
    format_ok = extraction == "boxed"
    return {
        **record,
        "raw_extracted_answer": raw,
        "parsed_answer": normalized_pred,
        "raw_exact_match": raw_match,
        "normalized_match": normalized_match,
        "canonical_match": canonical_match,
        "correctness": canonical_match,
        "extraction_method": extraction,
        "parser_status": canonical_status,
        "answer_format_ok": format_ok,
        "judge_required": judge_required,
        "judge_status": "pending" if judge_required else "not_required",
    }

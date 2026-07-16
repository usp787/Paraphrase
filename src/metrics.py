"""Preregistered robustness, variance, calibration, and paired-test metrics."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def group_rows(
    rows: Iterable[dict[str, Any]], *fields: str
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return groups


def majority_answer(rows: list[dict[str, Any]]) -> tuple[str | None, float, bool | None]:
    answers = [row.get("parsed_answer") for row in rows if row.get("parsed_answer") is not None]
    if not answers:
        return None, 0.0, None
    counts = Counter(answers)
    best_count = max(counts.values())
    tied = [answer for answer, count in counts.items() if count == best_count]
    if len(tied) > 1:
        logprob = {
            answer: sum(
                float(row.get("normalized_log_probability") or 0.0)
                for row in rows
                if row.get("parsed_answer") == answer
            )
            for answer in tied
        }
        best_logprob = max(logprob.values())
        tied = [answer for answer in tied if logprob[answer] == best_logprob]
    answer = sorted(tied)[0]
    matching = [row.get("correctness") for row in rows if row.get("parsed_answer") == answer]
    decided = [bool(value) for value in matching if value is not None]
    correctness = (sum(decided) >= (len(decided) / 2)) if decided else None
    return answer, best_count / len(rows), correctness


def item_form_summary(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in group_rows(rows, "item_id", "form_id").items():
        correctness = [
            float(row["correctness"]) for row in group if row.get("correctness") is not None
        ]
        answer, confidence, majority_correct = majority_answer(group)
        result[key] = {
            "solve_rate": mean(correctness),
            "majority_correct": majority_correct,
            "majority_answer": answer,
            "confidence": confidence,
        }
    return result


def variance_decomposition(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary = item_form_summary(rows)
    by_item: dict[str, list[float]] = defaultdict(list)
    run_variances = []
    for (item_id, form_id), stats in summary.items():
        group = [row for row in rows if row["item_id"] == item_id and row["form_id"] == form_id]
        values = [float(row["correctness"]) for row in group if row.get("correctness") is not None]
        if values:
            run_variances.append(float(np.var(values, ddof=0)))
            by_item[item_id].append(stats["solve_rate"])
    v_run = mean(run_variances)
    v_form = mean(float(np.var(values, ddof=0)) for values in by_item.values() if values)
    item_means = [mean(values) for values in by_item.values() if values]
    v_item = float(np.var(item_means, ddof=0)) if item_means else float("nan")
    denominator = v_run + v_form + v_item
    return {
        "v_run": v_run,
        "v_form": v_form,
        "v_item": v_item,
        "pi_form": v_form / denominator if denominator > 0 else 0.0,
    }


def main_robustness_metrics(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    summary = item_form_summary(rows)
    item_ids = sorted({row["item_id"] for row in rows})
    forms = sorted({row["form_id"] for row in rows})
    item_outputs: dict[str, dict[str, Any]] = {}
    for item_id in item_ids:
        form_stats = [summary[(item_id, form)] for form in forms if (item_id, form) in summary]
        majority = [stats["majority_correct"] for stats in form_stats]
        decided = [value for value in majority if value is not None]
        answers = {
            stats["majority_answer"] for stats in form_stats if stats["majority_answer"] is not None
        }
        item_outputs[item_id] = {
            "robust": bool(decided) and len(decided) == len(forms) and all(decided),
            "worst_form_solve_rate": min(stats["solve_rate"] for stats in form_stats),
            "answer_flip": len(answers) > 1,
            "correctness_flip": len(set(decided)) > 1,
            "mean_correctness": mean(stats["solve_rate"] for stats in form_stats),
        }
    decided_rows = [row for row in rows if row.get("correctness") is not None]
    originals = [row for row in decided_rows if row["form_id"] == "original"]
    paraphrases = [row for row in decided_rows if row["form_id"] != "original"]
    metrics = {
        "mean_accuracy": mean(float(row["correctness"]) for row in decided_rows),
        "original_accuracy": mean(float(row["correctness"]) for row in originals),
        "mean_paraphrase_accuracy": mean(float(row["correctness"]) for row in paraphrases),
        "robust_accuracy": mean(float(value["robust"]) for value in item_outputs.values()),
        "worst_form_accuracy": mean(
            value["worst_form_solve_rate"] for value in item_outputs.values()
        ),
        "answer_flip_rate": mean(float(value["answer_flip"]) for value in item_outputs.values()),
        "correctness_flip_rate": mean(
            float(value["correctness_flip"]) for value in item_outputs.values()
        ),
        **variance_decomposition(rows),
    }
    return metrics, item_outputs


def entropy(answers: list[str | None]) -> float:
    clean = [answer for answer in answers if answer is not None]
    if not clean:
        return float("nan")
    counts = Counter(clean)
    probabilities = [count / len(clean) for count in counts.values()]
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def expected_calibration_error(confidence: list[float], correctness: list[int], bins: int) -> float:
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            members = [
                i for i, value in enumerate(confidence) if edges[index] <= value <= edges[index + 1]
            ]
        else:
            members = [
                i for i, value in enumerate(confidence) if edges[index] <= value < edges[index + 1]
            ]
        if members:
            result += (
                len(members)
                / len(confidence)
                * abs(mean(confidence[i] for i in members) - mean(correctness[i] for i in members))
            )
    return result


def adaptive_ece(confidence: list[float], correctness: list[int], bins: int) -> float:
    order = sorted(range(len(confidence)), key=confidence.__getitem__)
    chunks = np.array_split(order, min(bins, len(order)))
    return sum(
        len(chunk)
        / len(order)
        * abs(mean(confidence[int(i)] for i in chunk) - mean(correctness[int(i)] for i in chunk))
        for chunk in chunks
        if len(chunk)
    )


def fixed_budget_metrics(
    rows: list[dict[str, Any]], bins: int = 10
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for (item_id,), group in group_rows(rows, "item_id").items():
        answer, confidence, correct = majority_answer(group)
        aggregated[item_id] = {
            "answer": answer,
            "confidence": confidence,
            "correctness": correct,
            "entropy": entropy([row.get("parsed_answer") for row in group]),
            "input_tokens": sum(int(row.get("input_tokens", 0)) for row in group),
            "output_tokens": sum(int(row.get("output_tokens", 0)) for row in group),
            "gpu_seconds": sum(float(row.get("gpu_seconds", 0.0)) for row in group),
        }
    decided = [value for value in aggregated.values() if value["correctness"] is not None]
    confidence = [float(value["confidence"]) for value in decided]
    correctness = [int(value["correctness"]) for value in decided]
    metrics = {
        "final_accuracy": mean(correctness),
        "brier_score": mean((c - y) ** 2 for c, y in zip(confidence, correctness, strict=True)),
        "ece_fixed": expected_calibration_error(confidence, correctness, bins),
        "ece_adaptive": adaptive_ece(confidence, correctness, bins),
        "mean_input_tokens_per_item": mean(value["input_tokens"] for value in decided),
        "mean_output_tokens_per_item": mean(value["output_tokens"] for value in decided),
        "mean_gpu_seconds_per_item": mean(value["gpu_seconds"] for value in decided),
    }
    try:
        from sklearn.metrics import roc_auc_score

        incorrect = [1 - value for value in correctness]
        disagreement = [1.0 - value for value in confidence]
        entropies = [float(value["entropy"]) for value in decided]
        metrics["incorrect_detection_auroc_disagreement"] = (
            float(roc_auc_score(incorrect, disagreement))
            if len(set(incorrect)) > 1
            else float("nan")
        )
        metrics["incorrect_detection_auroc_entropy"] = (
            float(roc_auc_score(incorrect, entropies)) if len(set(incorrect)) > 1 else float("nan")
        )
    except ImportError:
        metrics["incorrect_detection_auroc_disagreement"] = float("nan")
        metrics["incorrect_detection_auroc_entropy"] = float("nan")
    return metrics, aggregated


def bootstrap_ci(values: list[float], resamples: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = [mean(values[rng.randrange(len(values))] for _ in values) for _ in range(resamples)]
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def bootstrap_variance_ci(
    rows: list[dict[str, Any]], resamples: int, seed: int
) -> dict[str, tuple[float, float]]:
    by_item_form = group_rows(rows, "item_id", "form_id")
    per_item: dict[str, dict[str, float]] = {}
    for item_id in sorted({row["item_id"] for row in rows}):
        form_values = []
        run_variances = []
        for (group_item, _form), group in by_item_form.items():
            if group_item != item_id:
                continue
            values = [
                float(row["correctness"]) for row in group if row.get("correctness") is not None
            ]
            if values:
                form_values.append(mean(values))
                run_variances.append(float(np.var(values, ddof=0)))
        if form_values:
            per_item[item_id] = {
                "run": mean(run_variances),
                "form": float(np.var(form_values, ddof=0)),
                "item_mean": mean(form_values),
            }
    item_ids = sorted(per_item)
    if not item_ids:
        return {
            key: (float("nan"), float("nan")) for key in ("v_run", "v_form", "v_item", "pi_form")
        }
    rng = random.Random(seed)
    draws = {key: [] for key in ("v_run", "v_form", "v_item", "pi_form")}
    for _ in range(resamples):
        sampled = [per_item[item_ids[rng.randrange(len(item_ids))]] for _ in item_ids]
        v_run = mean(item["run"] for item in sampled)
        v_form = mean(item["form"] for item in sampled)
        v_item = float(np.var([item["item_mean"] for item in sampled], ddof=0))
        denominator = v_run + v_form + v_item
        values = {
            "v_run": v_run,
            "v_form": v_form,
            "v_item": v_item,
            "pi_form": v_form / denominator if denominator > 0 else 0.0,
        }
        for key, value in values.items():
            draws[key].append(value)
    return {
        key: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for key, values in draws.items()
    }


def paired_difference_ci(
    left: Mapping[str, bool | float],
    right: Mapping[str, bool | float],
    resamples: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    common = sorted(set(left) & set(right))
    if not common:
        return float("nan"), (float("nan"), float("nan"))
    differences = [float(left[item]) - float(right[item]) for item in common]
    return mean(differences), bootstrap_ci(differences, resamples, seed)


def mcnemar_paired(left: Mapping[str, bool], right: Mapping[str, bool]) -> dict[str, float | int]:
    common = sorted(set(left) & set(right))
    b = sum(bool(left[item]) and not bool(right[item]) for item in common)
    c = sum(not bool(left[item]) and bool(right[item]) for item in common)
    try:
        from scipy.stats import binomtest

        pvalue = (
            float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
            if b + c
            else 1.0
        )
    except ImportError:
        pvalue = float("nan")
    return {"n": len(common), "left_only": b, "right_only": c, "pvalue": pvalue}


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [float("nan")] * len(pvalues)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted

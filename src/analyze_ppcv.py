"""Analyze random-sample, conditional, and cost-matched PPCV outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.common import atomic_write_json, load_jsonl, resolve_path, utc_now
from src.metrics import fixed_budget_metrics, group_rows, mean


def clean_json(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [clean_json(item) for item in value]
    return value


def accuracy(rows: list[dict[str, Any]], field: str) -> float:
    decided = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(decided)


def disagreement(row: dict[str, Any]) -> bool:
    answers = {
        rollout.get("parsed_answer")
        for rollout in row.get("sc_rollouts", [])
        if rollout.get("parsed_answer") is not None
    }
    return len(answers) > 1


def summarize_fidelity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_errors = [row for row in rows if row.get("initial_correctness") is False]
    disagreement_rows = [row for row in rows if disagreement(row)]
    return {
        "n": len(rows),
        "ppcv_accuracy": accuracy(rows, "correctness"),
        "sc48_accuracy": accuracy(rows, "sc_correctness"),
        "ppcv_minus_sc48": accuracy(rows, "correctness") - accuracy(rows, "sc_correctness"),
        "baseline_error_subset": {
            "n": len(baseline_errors),
            "ppcv_accuracy": accuracy(baseline_errors, "correctness"),
        },
        "sc_disagreement_subset": {
            "n": len(disagreement_rows),
            "ppcv_accuracy": accuracy(disagreement_rows, "correctness"),
            "sc48_accuracy": accuracy(disagreement_rows, "sc_correctness"),
        },
        "mean_ppcv_output_tokens": mean(float(row["ppcv_output_tokens"]) for row in rows),
        "mean_sc48_output_tokens": mean(float(row["sc_output_tokens"]) for row in rows),
        "mean_probe_seconds": mean(float(row["probe_elapsed_seconds"]) for row in rows),
        "mean_ppcv_seconds": mean(float(row["ppcv_elapsed_seconds"]) for row in rows),
        "p95_ppcv_seconds": float(np.quantile([row["ppcv_elapsed_seconds"] for row in rows], 0.95)),
    }


def summarize_lite(rows: list[dict[str, Any]], scop_rows: list[dict[str, Any]]) -> dict[str, Any]:
    item_ids = {row["item_id"] for row in rows}
    baselines = {}
    for (method,), method_rows in group_rows(
        [row for row in scop_rows if row["item_id"] in item_ids], "method"
    ).items():
        metrics, _ = fixed_budget_metrics(method_rows)
        baselines[method] = metrics
    return {
        "n": len(rows),
        "ppcv_lite_accuracy": accuracy(rows, "correctness"),
        "mean_ppcv_output_tokens": mean(float(row["ppcv_output_tokens"]) for row in rows),
        "mean_probe_seconds": mean(float(row["probe_elapsed_seconds"]) for row in rows),
        "mean_ppcv_seconds": mean(float(row["ppcv_elapsed_seconds"]) for row in rows),
        "round3_same_item_baselines": baselines,
        "note": (
            "Round 3 baselines use the same selected item IDs. Compare both accuracy and "
            "mean output-token/GPU cost; this is not a baseline-failure-only subset."
        ),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# PPCV reproduction report",
        "",
        "This is a reduced Qwen3-8B reproduction of a preprint experiment reported at 32B.",
        "",
    ]
    for section in ("fidelity", "lite"):
        lines.extend([f"## {section.title()}", "", "```json"])
        lines.append(json.dumps(result[section], indent=2, sort_keys=True))
        lines.extend(["```", ""])
    lines.extend(
        [
            "## Interpretation constraint",
            "",
            "Random-sample accuracy is primary. Baseline-error and disagreement results are "
            "conditional analyses, and probe cost is included when considering a router.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity", default="outputs/generations/ppcv_fidelity.jsonl")
    parser.add_argument("--lite", default="outputs/generations/ppcv_lite.jsonl")
    parser.add_argument(
        "--scop-scores", default="outputs/scores/scop_confirmatory_non_thinking_practical.jsonl"
    )
    parser.add_argument("--json", default="reports/ppcv_results.json")
    parser.add_argument("--markdown", default="reports/ppcv_report.md")
    args = parser.parse_args()
    fidelity = load_jsonl(args.fidelity)
    lite = load_jsonl(args.lite)
    scop = load_jsonl(args.scop_scores)
    if len(fidelity) != 50 or len(lite) != 100:
        raise ValueError(
            f"PPCV runs are incomplete: fidelity={len(fidelity)}/50, lite={len(lite)}/100"
        )
    result = clean_json(
        {
            "created_at": utc_now(),
            "fidelity": summarize_fidelity(fidelity),
            "lite": summarize_lite(lite, scop),
        }
    )
    atomic_write_json(args.json, result)
    write_markdown(resolve_path(args.markdown), result)
    print(f"[ppcv-analysis] wrote {args.json} and {args.markdown}")


if __name__ == "__main__":
    main()

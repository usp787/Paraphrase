"""Analyze scored runs and write JSON, Markdown, and the accuracy-cost Pareto plot."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.common import atomic_write_json, load_jsonl, read_yaml, resolve_path, utc_now
from src.metrics import (
    bootstrap_ci,
    bootstrap_variance_ci,
    fixed_budget_metrics,
    group_rows,
    holm_adjust,
    main_robustness_metrics,
    mcnemar_paired,
    paired_difference_ci,
)


def clean_json(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    return value


def analyze(rows: list[dict[str, Any]], config: dict) -> dict[str, Any]:
    result: dict[str, Any] = {"created_at": utc_now(), "groups": {}, "paired_tests": []}
    item_outputs_by_group: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for (dataset, mode, method), group in group_rows(rows, "dataset", "mode", "method").items():
        name = f"{dataset}/{mode}/{method}"
        if method in {"main", "external_main"}:
            metrics, item_outputs = main_robustness_metrics(group)
            per_item = [value["mean_correctness"] for value in item_outputs.values()]
            metrics["mean_accuracy_ci95"] = bootstrap_ci(
                per_item,
                int(config["analysis"]["bootstrap_resamples"]),
                int(config["analysis"]["bootstrap_seed"]),
            )
            metrics["variance_ci95"] = bootstrap_variance_ci(
                group,
                int(config["analysis"]["bootstrap_resamples"]),
                int(config["analysis"]["bootstrap_seed"]),
            )
            for label, field in (
                ("raw", "raw_exact_match"),
                ("normalized", "normalized_match"),
                ("symbolic_pre_judge", "canonical_match"),
            ):
                layer_rows = [
                    {**row, "correctness": row[field]}
                    for row in group
                    if row.get(field) is not None
                ]
                if layer_rows:
                    layer_metrics, _ = main_robustness_metrics(layer_rows)
                    metrics[f"{label}_accuracy"] = layer_metrics["mean_accuracy"]
                    metrics[f"{label}_robust_accuracy"] = layer_metrics["robust_accuracy"]
                    metrics[f"{label}_correctness_flip_rate"] = layer_metrics[
                        "correctness_flip_rate"
                    ]
                    metrics[f"{label}_pi_form"] = layer_metrics["pi_form"]
        else:
            metrics, item_outputs = fixed_budget_metrics(group, int(config["analysis"]["ece_bins"]))
            per_item = [
                float(value["correctness"])
                for value in item_outputs.values()
                if value["correctness"] is not None
            ]
            metrics["final_accuracy_ci95"] = bootstrap_ci(
                per_item,
                int(config["analysis"]["bootstrap_resamples"]),
                int(config["analysis"]["bootstrap_seed"]),
            )
        metrics["raw_exact_accuracy"] = sum(bool(row["raw_exact_match"]) for row in group) / len(
            group
        )
        metrics["normalized_parser_accuracy"] = sum(
            bool(row["normalized_match"]) for row in group
        ) / len(group)
        symbolic_decided = [row for row in group if row.get("canonical_match") is not None]
        metrics["symbolic_pre_judge_accuracy"] = (
            sum(bool(row["canonical_match"]) for row in symbolic_decided) / len(symbolic_decided)
            if symbolic_decided
            else None
        )
        metrics["judge_adjudication_rate"] = sum(
            bool(row.get("judge_required")) for row in group
        ) / len(group)
        metrics["answer_format_violation_rate"] = sum(
            not bool(row.get("answer_format_ok")) for row in group
        ) / len(group)
        metrics["total_input_tokens"] = sum(int(row.get("input_tokens", 0)) for row in group)
        metrics["total_output_tokens"] = sum(int(row.get("output_tokens", 0)) for row in group)
        metrics["total_gpu_seconds"] = sum(float(row.get("gpu_seconds", 0.0)) for row in group)
        item_count = len({row["item_id"] for row in group})
        metrics["mean_input_tokens_per_item"] = metrics["total_input_tokens"] / item_count
        metrics["mean_output_tokens_per_item"] = metrics["total_output_tokens"] / item_count
        metrics["mean_gpu_seconds_per_item"] = metrics["total_gpu_seconds"] / item_count
        item_latencies = [
            sum(float(row.get("latency_seconds", 0.0)) for row in item_rows)
            for item_rows in group_rows(group, "item_id").values()
        ]
        metrics["mean_latency_seconds_per_item"] = sum(item_latencies) / item_count
        metrics["p95_latency_seconds_per_item"] = float(np.quantile(item_latencies, 0.95))
        metrics["peak_gpu_memory_gib"] = max(
            (
                float(row["peak_gpu_memory_gib"])
                for row in group
                if row.get("peak_gpu_memory_gib") is not None
            ),
            default=None,
        )
        metrics["truncation_rate"] = sum(bool(row.get("truncated")) for row in group) / max(
            1, len(group)
        )
        metrics["generation_failure_rate"] = sum(
            not str(row.get("raw_output", "")).strip()
            or row.get("finish_reason") not in {"stop", "length"}
            for row in group
        ) / max(1, len(group))
        accuracy = metrics.get("final_accuracy", metrics.get("mean_accuracy"))
        if accuracy is not None and metrics["mean_output_tokens_per_item"] > 0:
            metrics["accuracy_per_million_generated_tokens"] = (
                accuracy * 1_000_000 / metrics["mean_output_tokens_per_item"]
            )
        if accuracy is not None and metrics["mean_gpu_seconds_per_item"] > 0:
            metrics["accuracy_per_gpu_hour"] = (
                accuracy * 3600 / metrics["mean_gpu_seconds_per_item"]
            )
        if metrics.get("robust_accuracy") is not None and metrics["mean_gpu_seconds_per_item"] > 0:
            metrics["robust_accuracy_per_gpu_hour"] = (
                metrics["robust_accuracy"] * 3600 / metrics["mean_gpu_seconds_per_item"]
            )
        result["groups"][name] = metrics
        item_outputs_by_group[(dataset, mode, method)] = item_outputs

    # Equal-token sensitivity analysis for the four eight-sample methods. Each
    # item's observed SC-8 spend is the cap; rows enter in a deterministic order
    # that never consults correctness.
    nonthinking = [row for row in rows if row["mode"] == "non_thinking"]
    grouped_methods = group_rows(nonthinking, "dataset", "method")
    for dataset in sorted({row["dataset"] for row in nonthinking}):
        method_rows = {
            method: group
            for (group_dataset, method), group in grouped_methods.items()
            if group_dataset == dataset and method.startswith(("sc_", "scop_"))
        }
        if "sc_8" not in method_rows:
            continue
        reference_budgets = {
            item_id: sum(int(row["output_tokens"]) for row in item_group)
            for (item_id,), item_group in group_rows(method_rows["sc_8"], "item_id").items()
        }
        for method, method_group in method_rows.items():
            matched = []
            for (item_id,), item_group in group_rows(method_group, "item_id").items():
                ordered = sorted(
                    item_group,
                    key=lambda row: (
                        row["form_id"],
                        row.get("sample_index") or 0,
                        row["seed"],
                    ),
                )
                selected_for_item = []
                spent = 0
                for row in ordered:
                    cost = int(row["output_tokens"])
                    if selected_for_item and spent + cost > reference_budgets[item_id]:
                        continue
                    selected_for_item.append(row)
                    spent += cost
                matched.extend(selected_for_item)
            matched_metrics, _ = fixed_budget_metrics(matched, int(config["analysis"]["ece_bins"]))
            group_name = f"{dataset}/non_thinking/{method}"
            result["groups"][group_name]["token_matched_to_sc8"] = matched_metrics

    pvalues = []
    for dataset in sorted({row["dataset"] for row in rows}):
        left_key = (dataset, "thinking", "main")
        right_key = (dataset, "non_thinking", "main")
        if left_key not in item_outputs_by_group or right_key not in item_outputs_by_group:
            continue
        left = {
            item: bool(value["robust"]) for item, value in item_outputs_by_group[left_key].items()
        }
        right = {
            item: bool(value["robust"]) for item, value in item_outputs_by_group[right_key].items()
        }
        test = {
            "comparison": f"{dataset}: thinking vs non_thinking robust correctness",
            **mcnemar_paired(left, right),
        }
        delta, delta_ci = paired_difference_ci(
            left,
            right,
            int(config["analysis"]["bootstrap_resamples"]),
            int(config["analysis"]["bootstrap_seed"]),
        )
        test["paired_effect_left_minus_right"] = delta
        test["paired_effect_ci95"] = delta_ci
        result["paired_tests"].append(test)
        pvalues.append(float(test["pvalue"]))

    for dataset in sorted({row["dataset"] for row in rows}):
        method_keys = sorted(
            key
            for key in item_outputs_by_group
            if key[0] == dataset
            and key[1] == "non_thinking"
            and key[2].startswith(("sc_", "scop_"))
        )
        for left_key, right_key in itertools.combinations(method_keys, 2):
            left = {
                item: bool(value["correctness"])
                for item, value in item_outputs_by_group[left_key].items()
                if value["correctness"] is not None
            }
            right = {
                item: bool(value["correctness"])
                for item, value in item_outputs_by_group[right_key].items()
                if value["correctness"] is not None
            }
            test = {
                "comparison": f"{dataset}: {left_key[2]} vs {right_key[2]} final correctness",
                **mcnemar_paired(left, right),
            }
            delta, delta_ci = paired_difference_ci(
                left,
                right,
                int(config["analysis"]["bootstrap_resamples"]),
                int(config["analysis"]["bootstrap_seed"]),
            )
            test["paired_effect_left_minus_right"] = delta
            test["paired_effect_ci95"] = delta_ci
            result["paired_tests"].append(test)
            pvalues.append(float(test["pvalue"]))
    if pvalues and all(not math.isnan(value) for value in pvalues):
        for test, adjusted in zip(result["paired_tests"], holm_adjust(pvalues), strict=True):
            test["holm_adjusted_pvalue"] = adjusted
    return clean_json(result)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Experiment results",
        "",
        f"Generated: {result['created_at']}",
        "",
        "## Metric groups",
        "",
    ]
    for name, metrics in result["groups"].items():
        lines.extend([f"### {name}", "", "| Metric | Value |", "|---|---:|"])
        for metric, value in metrics.items():
            if isinstance(value, dict | list):
                rendered = f"`{json.dumps(value, sort_keys=True)}`"
            elif isinstance(value, float):
                rendered = f"{value:.6g}"
            else:
                rendered = str(value)
            lines.append(f"| {metric} | {rendered} |")
        lines.append("")
    if result["paired_tests"]:
        lines.extend(["## Paired tests", ""])
        for test in result["paired_tests"]:
            lines.append(f"- `{test['comparison']}`: `{json.dumps(test, sort_keys=True)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_pareto(path: Path, result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, metrics in result["groups"].items():
        accuracy = metrics.get("final_accuracy", metrics.get("mean_accuracy"))
        tokens = metrics.get("mean_output_tokens_per_item")
        if tokens is None:
            # Main run spends 15 responses per item (5 forms x 3 seeds).
            tokens = metrics["total_output_tokens"]
        if accuracy is not None:
            ax.scatter(tokens, accuracy, s=55)
            ax.annotate(
                name, (tokens, accuracy), fontsize=8, xytext=(4, 4), textcoords="offset points"
            )
    ax.set_xlabel("Generated tokens (per item where available; otherwise total)")
    ax.set_ylabel("Canonical accuracy")
    ax.set_title("Accuracy-cost comparison")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores", action="append", required=True, help="repeat for each scored run"
    )
    parser.add_argument("--json", default="reports/main_results.json")
    parser.add_argument("--markdown", default="reports/main_results.md")
    parser.add_argument("--figure", default="reports/figures/accuracy_cost_pareto.png")
    args = parser.parse_args()
    config = read_yaml("configs/experiment_main.yaml")
    rows = [row for path in args.scores for row in load_jsonl(path)]
    if not rows:
        raise ValueError("No scored rows were provided")
    undecided = sum(row.get("correctness") is None for row in rows)
    if undecided:
        raise ValueError(f"{undecided} rows remain undecided; complete score adjudication first")
    result = analyze(rows, config)
    atomic_write_json(args.json, result)
    write_markdown(resolve_path(args.markdown), result)
    plot_pareto(resolve_path(args.figure), result)
    print(f"[analyze] wrote {args.json}, {args.markdown}, and {args.figure}")


if __name__ == "__main__":
    main()

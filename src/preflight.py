"""Enforce preregistered go/no-go gates before spending the next GPU round."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from src.build_semantic_audit import validate as validate_audit
from src.common import immutable_revision, load_jsonl, read_yaml, record_key, resolve_path


def check_frozen_assets(config: dict, require_all_models: bool = False) -> list[str]:
    errors = []
    model_config = read_yaml(config["model_config"])
    keys = ["model", "external_model"] if require_all_models else ["model"]
    for key in keys:
        spec = model_config[key]
        try:
            immutable_revision(spec.get("revision"), spec["id"], "environment/model_revisions.json")
        except RuntimeError as exc:
            errors.append(str(exc))
    lock = resolve_path(config["manifest_path"]).with_suffix(".lock.json")
    if not lock.exists():
        errors.append(f"Missing frozen manifest lock: {lock}")
    return errors


def result_gate_errors(rows: list[dict], config: dict) -> list[str]:
    gates = config["failure_gates"]
    errors: list[str] = []
    keys = [record_key(row) for row in rows]
    duplicate_count = sum(count - 1 for count in Counter(keys).values() if count > 1)
    if duplicate_count > int(gates["max_duplicate_keys"]):
        errors.append(f"duplicate result keys: {duplicate_count}")
    for run_id in sorted({row.get("run_id") for row in rows if row.get("run_id")}):
        progress_path = resolve_path(f"outputs/checkpoints/{run_id}.json")
        if not progress_path.exists():
            errors.append(f"missing progress manifest for run {run_id}")
            continue
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("completed_records") != progress.get("expected_records"):
            errors.append(
                f"run {run_id} is incomplete: {progress.get('completed_records')}/"
                f"{progress.get('expected_records')} records"
            )
    total = max(1, len(rows))
    truncation = sum(bool(row.get("truncated")) for row in rows) / total
    if truncation > float(gates["max_truncation_rate"]):
        errors.append(
            f"truncation rate {truncation:.2%} exceeds {gates['max_truncation_rate']:.2%}"
        )
    if rows and "correctness" in rows[0]:
        undecided = sum(row.get("correctness") is None for row in rows) / total
        if undecided > float(gates["max_parser_or_judge_undecided_rate"]):
            errors.append(
                f"parser/judge undecided rate {undecided:.2%} exceeds "
                f"{gates['max_parser_or_judge_undecided_rate']:.2%}"
            )
        violations = sum(not bool(row.get("answer_format_ok")) for row in rows) / total
        if violations > float(gates["max_answer_format_violation_rate"]):
            errors.append(
                f"answer-format violation rate {violations:.2%} exceeds "
                f"{gates['max_answer_format_violation_rate']:.2%}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_main.yaml")
    parser.add_argument("--stage", choices=["pilot", "confirmatory", "results"], required=True)
    parser.add_argument("--input", help="generation or scored JSONL for results gates")
    args = parser.parse_args()
    config = read_yaml(args.config)
    errors = check_frozen_assets(config)
    if args.stage == "pilot":
        for split in ("smoke", "pilot"):
            validation_path = resolve_path(
                f"outputs/checkpoints/paraphrase_validation_{split}.json"
            )
            if not validation_path.exists() or not json.loads(
                validation_path.read_text(encoding="utf-8")
            ).get("passed"):
                errors.append(f"{split} paraphrase validation is missing or failed")
    if args.stage == "confirmatory":
        validation_path = resolve_path(
            "outputs/checkpoints/paraphrase_validation_confirmatory.json"
        )
        if not validation_path.exists() or not json.loads(
            validation_path.read_text(encoding="utf-8")
        ).get("passed"):
            errors.append("Confirmatory paraphrase validation is missing or failed")
        try:
            audit_ok = validate_audit(config)
        except (FileNotFoundError, ValueError) as exc:
            audit_ok = False
            errors.append(f"Semantic audit unavailable: {exc}")
        if not audit_ok and not any("Semantic audit" in error for error in errors):
            errors.append("Semantic audit is incomplete, uncertain, or below 95% equivalence")
    if args.stage == "results":
        if not args.input:
            parser.error("--input is required for --stage results")
        errors.extend(result_gate_errors(load_jsonl(args.input), config))
    if errors:
        print("[preflight] STOP: failure gates did not pass")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(2)
    print(f"[preflight] {args.stage} gates passed")


if __name__ == "__main__":
    main()

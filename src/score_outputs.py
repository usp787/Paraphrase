"""Score generation JSONL, emit resumable scored JSONL, Parquet, and judge queue."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.common import append_jsonl, completed_keys, load_jsonl, record_key, resolve_path
from src.scoring import score_record

JUDGMENT_PATH = "outputs/checkpoints/score_judgments.jsonl"


def judgment_map() -> dict[tuple[str, ...], dict[str, Any]]:
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in load_jsonl(JUDGMENT_PATH):
        result[record_key(row)] = row
    return result


def apply_judgment(
    row: dict[str, Any], judgments: dict[tuple[str, ...], dict[str, Any]]
) -> dict[str, Any]:
    judgment = judgments.get(record_key(row))
    if not judgment:
        return row
    parsed = judgment["judgment"]
    if parsed.get("parse_ok") and not parsed.get("uncertain"):
        row["correctness"] = bool(parsed["equivalent"])
        row["judge_status"] = "decided"
    else:
        row["correctness"] = None
        row["judge_status"] = "uncertain"
    row["score_judge_model"] = judgment.get("judge_model")
    row["score_judge_revision"] = judgment.get("judge_revision")
    return row


def write_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "item_id",
        "form_id",
        "mode",
        "seed",
        "method",
        "raw_extracted_answer",
        "canonical_answer",
        "parser_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-queue")
    parser.add_argument("--parquet")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = load_jsonl(args.input)
    output_path = resolve_path(args.output)
    if args.force:
        output_path.unlink(missing_ok=True)
    seen, duplicates = completed_keys(output_path)
    if duplicates:
        raise RuntimeError(f"Scored output already has {len(duplicates)} duplicate keys")
    judgments = judgment_map()
    pending = [row for row in source if record_key(row) not in seen]
    scored = [apply_judgment(score_record(row), judgments) for row in pending]
    append_jsonl(output_path, scored)
    all_scored = load_jsonl(output_path)
    queue = [
        row for row in all_scored if row["judge_required"] and row["judge_status"] == "pending"
    ]
    queue_path = resolve_path(args.judge_queue or output_path.with_suffix(".judge_queue.csv"))
    write_queue(queue_path, queue)
    print(f"[score] scored {len(scored)} new rows; {len(queue)} need blinded adjudication")

    if args.parquet:
        import pandas as pd

        parquet_path = resolve_path(args.parquet)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_scored).to_parquet(parquet_path, index=False)
        print(f"[score] wrote {parquet_path}")


if __name__ == "__main__":
    main()

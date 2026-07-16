"""Create the frozen, disjoint GSM-Symbolic and MATH-500 manifest.

This command resolves each dataset to a Hub commit before loading it.  It writes
both the row manifest and a lock containing the exact source revisions and item
IDs.  Existing manifests are never overwritten unless ``--force`` is explicit.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.common import (
    append_jsonl,
    atomic_write_json,
    object_fingerprint,
    read_yaml,
    resolve_path,
    utc_now,
)


def gsm_final_answer(answer: Any) -> str:
    text = str(answer).strip()
    if "####" in text:
        text = text.rsplit("####", 1)[1].strip()
    return text.replace(",", "")


def source_item_id(dataset: str, row: Mapping[str, Any], fields: list[str], index: int) -> str:
    values = [str(row[field]) for field in fields if field in row and row[field] is not None]
    if values:
        return f"{dataset}:" + ":".join(values)
    digest = object_fingerprint({"dataset": dataset, "index": index, "row": dict(row)})[:16]
    return f"{dataset}:{digest}"


def normalize_row(
    dataset: str,
    row: Mapping[str, Any],
    index: int,
    spec: Mapping[str, Any],
    revision: str,
) -> dict[str, Any]:
    question_field = spec["question_field"]
    answer_field = spec["answer_field"]
    question = str(row.get(question_field, "")).strip()
    if not question:
        raise ValueError(f"{dataset} row {index} has no {question_field!r}")
    raw_answer = str(row.get(answer_field, "")).strip()
    if not raw_answer:
        raise ValueError(f"{dataset} row {index} has no {answer_field!r}")
    canonical_answer = gsm_final_answer(raw_answer) if dataset == "gsm_symbolic" else raw_answer
    item_id = source_item_id(dataset, row, list(spec["source_id_fields"]), index)
    return {
        "dataset": dataset,
        "item_id": item_id,
        "source_index": index,
        "source_revision": revision,
        "original_text": question,
        "gold_answer": raw_answer,
        "canonical_answer": canonical_answer,
        "source_metadata": {
            key: row[key]
            for key in ("id", "instance", "original_id", "subject", "level", "unique_id")
            if key in row
        },
    }


def deterministic_splits(
    rows: list[dict[str, Any]], plan: Mapping[str, int], seed: int, dataset: str
) -> list[dict[str, Any]]:
    counts = {name: int(plan[name]) for name in ("smoke", "pilot", "confirmatory")}
    needed = sum(counts.values())
    if len(rows) < needed:
        raise ValueError(f"{dataset} has {len(rows)} rows, but the split plan needs {needed}")
    indices = list(range(len(rows)))
    # A dataset-specific salt keeps the two selected orders independent.
    random.Random(f"{seed}:{dataset}").shuffle(indices)
    selected = [rows[index] for index in indices[:needed]]
    cursor = 0
    for split_name in ("smoke", "pilot", "confirmatory"):
        for row in selected[cursor : cursor + counts[split_name]]:
            row["split"] = split_name
        cursor += counts[split_name]
    return selected


def resolve_dataset_revision(hub_id: str, configured: str | None) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(hub_id, revision=configured or "main")
    if not info.sha:
        raise RuntimeError(f"Hub did not return a commit for dataset {hub_id}")
    return info.sha


def load_source(spec: Mapping[str, Any], revision: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": spec["hub_id"],
        "split": spec["split"],
        "revision": revision,
    }
    if spec.get("hub_config"):
        kwargs["name"] = spec["hub_config"]
    return [dict(row) for row in load_dataset(**kwargs)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_main.yaml")
    parser.add_argument("--force", action="store_true", help="replace an existing manifest")
    args = parser.parse_args()

    config = read_yaml(args.config)
    manifest_path = resolve_path(config["manifest_path"])
    lock_path = manifest_path.with_suffix(".lock.json")
    if (manifest_path.exists() or lock_path.exists()) and not args.force:
        raise FileExistsError(
            f"{manifest_path} already exists. Refusing to alter frozen item IDs; "
            "use --force explicitly."
        )
    if args.force:
        manifest_path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)

    all_rows: list[dict[str, Any]] = []
    revisions: dict[str, dict[str, str]] = {}
    plan = config["split_plan"]
    for dataset, spec in config["datasets"].items():
        revision = resolve_dataset_revision(spec["hub_id"], spec.get("revision"))
        revisions[spec["hub_id"]] = {"revision": revision, "kind": "dataset"}
        source = load_source(spec, revision)
        normalized = [
            normalize_row(dataset, row, index, spec, revision) for index, row in enumerate(source)
        ]
        ids = [row["item_id"] for row in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Source ID fields do not uniquely identify {dataset} rows")
        all_rows.extend(deterministic_splits(normalized, plan, int(plan["split_seed"]), dataset))

    append_jsonl(manifest_path, all_rows)
    lock = {
        "created_at": utc_now(),
        "config": str(Path(args.config)),
        "config_fingerprint": object_fingerprint(config),
        "manifest_fingerprint": object_fingerprint(all_rows),
        "dataset_revisions": revisions,
        "split_item_ids": {
            dataset: {
                split: [
                    row["item_id"]
                    for row in all_rows
                    if row["dataset"] == dataset and row["split"] == split
                ]
                for split in ("smoke", "pilot", "confirmatory")
            }
            for dataset in config["datasets"]
        },
    }
    atomic_write_json(lock_path, lock)
    atomic_write_json("environment/dataset_revisions.json", revisions)
    print(f"[data] wrote {len(all_rows)} rows -> {manifest_path}")
    print(f"[data] froze source revisions and IDs -> {lock_path}")


if __name__ == "__main__":
    main()

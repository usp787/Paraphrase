"""Create or validate the blinded human semantic-equivalence audit sheet."""

from __future__ import annotations

import argparse
import csv
import math
import random

from src.common import atomic_write_json, load_jsonl, read_yaml, resolve_path, utc_now


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def build(config: dict, force: bool) -> None:
    output = resolve_path(config["semantic_audit_path"])
    if output.exists() and not force:
        raise FileExistsError(
            f"{output} exists; use --force only if no audit work must be preserved"
        )
    rows = [row for row in load_jsonl(config["paraphrases_path"]) if row["split"] == "confirmatory"]
    analysis = config["analysis"]
    count = max(
        int(analysis["audit_min_pairs"]), math.ceil(float(analysis["audit_fraction"]) * len(rows))
    )
    if len(rows) < count:
        raise ValueError(f"Need {count} accepted confirmatory pairs but found {len(rows)}")
    selected = random.Random(int(analysis["bootstrap_seed"])).sample(rows, count)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "audit_id",
        "dataset",
        "item_id",
        "form_id",
        "transformation_type",
        "original_text",
        "paraphrase_text",
        "equivalence",  # equivalent | not_equivalent | uncertain
        "transformation_label_correct",  # yes | no | uncertain
        "notes",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            writer.writerow(
                {
                    "audit_id": f"audit-{index:04d}",
                    "dataset": row["dataset"],
                    "item_id": row["item_id"],
                    "form_id": row["form_id"],
                    "transformation_type": row["transformation_type"],
                    "original_text": row["original_text"],
                    "paraphrase_text": row["paraphrase_text"],
                    "equivalence": "",
                    "transformation_label_correct": "",
                    "notes": "",
                }
            )
    print(f"[audit] wrote {count} blinded pairs -> {output}")


def validate(config: dict) -> bool:
    path = resolve_path(config["semantic_audit_path"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    allowed_equivalence = {"equivalent", "not_equivalent", "uncertain"}
    allowed_labels = {"yes", "no", "uncertain"}
    invalid = [row["audit_id"] for row in rows if row["equivalence"] not in allowed_equivalence]
    invalid += [
        row["audit_id"] for row in rows if row["transformation_label_correct"] not in allowed_labels
    ]
    if invalid:
        print(f"[audit] incomplete or invalid rows: {sorted(set(invalid))[:20]}")
        return False
    decided = [row for row in rows if row["equivalence"] != "uncertain"]
    equivalent = [row for row in decided if row["equivalence"] == "equivalent"]
    rate = len(equivalent) / len(decided) if decided else 0.0
    interval = wilson_interval(len(equivalent), len(decided))
    threshold = float(config["failure_gates"]["min_human_equivalence_rate"])
    unresolved = sum(row["equivalence"] == "uncertain" for row in rows)
    print(
        f"[audit] equivalent={len(equivalent)}/{len(decided)} ({rate:.3%}); "
        f"95% Wilson CI=({interval[0]:.3%}, {interval[1]:.3%}); "
        f"uncertain={unresolved}; required={threshold:.1%}"
    )
    atomic_write_json(
        "outputs/checkpoints/semantic_audit_report.json",
        {
            "created_at": utc_now(),
            "audited_pairs": len(rows),
            "decided_pairs": len(decided),
            "equivalent_pairs": len(equivalent),
            "uncertain_pairs": unresolved,
            "equivalence_rate": rate,
            "wilson_ci95": interval,
            "required_rate": threshold,
            "passed": rate >= threshold and unresolved == 0,
        },
    )
    return rate >= threshold and unresolved == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_main.yaml")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = read_yaml(args.config)
    if args.validate:
        raise SystemExit(0 if validate(config) else 2)
    build(config, args.force)


if __name__ == "__main__":
    main()

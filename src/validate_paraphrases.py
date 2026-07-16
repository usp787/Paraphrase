"""Check accepted paraphrase completeness, gate provenance, and style confounds."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from src.common import atomic_write_json, load_jsonl, read_yaml, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_main.yaml")
    parser.add_argument("--split", choices=["smoke", "pilot", "confirmatory"], required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    config = read_yaml(args.config)
    manifest = [row for row in load_jsonl(config["manifest_path"]) if row["split"] == args.split]
    rows = [row for row in load_jsonl(config["paraphrases_path"]) if row["split"] == args.split]
    form_ids = [
        form["form_id"] for form in config["surface_forms"] if form["form_id"] != "original"
    ]
    expected = {(row["dataset"], row["item_id"], form) for row in manifest for form in form_ids}
    keys = [(row["dataset"], row["item_id"], row["form_id"]) for row in rows]
    counts = Counter(keys)
    missing = sorted(expected - set(keys))
    unexpected = sorted(set(keys) - expected)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    failed_provenance = [
        key for key, row in zip(keys, rows, strict=True) if not row["automatic_gates"]["passed"]
    ]
    ratios: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        ratios[row["transformation_type"]].append(
            len(row["paraphrase_text"].split()) / max(1, len(row["original_text"].split()))
        )
    mean_ratios = {key: sum(values) / len(values) for key, values in ratios.items()}
    systematic_shortening = {key: ratio for key, ratio in mean_ratios.items() if ratio < 0.85}
    report = {
        "created_at": utc_now(),
        "split": args.split,
        "expected_pairs": len(expected),
        "accepted_pairs": len(rows),
        "missing_count": len(missing),
        "missing_examples": missing[:25],
        "unexpected_count": len(unexpected),
        "duplicate_count": len(duplicates),
        "failed_gate_provenance_count": len(failed_provenance),
        "mean_paraphrase_to_original_word_ratio": mean_ratios,
        "systematic_shortening_below_0_85": systematic_shortening,
        "passed": not (
            missing or unexpected or duplicates or failed_provenance or systematic_shortening
        ),
    }
    output = args.report or f"outputs/checkpoints/paraphrase_validation_{args.split}.json"
    atomic_write_json(output, report)
    print(
        f"[validate] expected={len(expected)} accepted={len(rows)} missing={len(missing)} "
        f"duplicates={len(duplicates)} shortening_flags={len(systematic_shortening)}"
    )
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

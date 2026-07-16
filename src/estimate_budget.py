"""Estimate the maximum item count that fits a 7h45 Slurm round."""

from __future__ import annotations

import argparse
import math

from src.common import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True, help="completed smoke or pilot JSONL")
    parser.add_argument("--responses-per-item", type=int, required=True)
    parser.add_argument("--effective-output-tokens-per-second", type=float)
    args = parser.parse_args()
    rows = load_jsonl(args.generations)
    if not rows:
        raise ValueError("Pilot generation file is empty")
    mean_output = sum(int(row["output_tokens"]) for row in rows) / len(rows)
    throughput = args.effective_output_tokens_per_second
    if throughput is None:
        # gpu_seconds is amortized batch wall time, so summing it reconstructs
        # approximately one-GPU elapsed generation time.
        seconds = sum(float(row["gpu_seconds"]) for row in rows)
        throughput = sum(int(row["output_tokens"]) for row in rows) / seconds
    n_max = math.floor(0.70 * 28_800 * throughput / (args.responses_per_item * mean_output))
    print(f"mean_output_tokens={mean_output:.2f}")
    print(f"effective_output_tokens_per_second={throughput:.2f}")
    print(f"responses_per_item={args.responses_per_item}")
    print(f"N_max={n_max}")


if __name__ == "__main__":
    main()

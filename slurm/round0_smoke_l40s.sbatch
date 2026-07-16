#!/bin/bash
#SBATCH --job-name=para-r0-smoke
#SBATCH --array=0-1%2
# The 2026-07-16 monitor showed one idle L40S node and no idle H100. Qwen3-8B
# BF16 with the configured 8k context fits in 48 GB; outputs resume after timeout.
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/round0_smoke_%A_%a.out
#SBATCH --error=logs/round0_smoke_%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail
PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/slurm/common.sh"
record_environment "round0_smoke_${SLURM_ARRAY_TASK_ID}"
run_py src/preflight.py --stage smoke

case "$SLURM_ARRAY_TASK_ID" in
  0) MODE=non_thinking ;;
  1) MODE=thinking ;;
  *) echo "Unexpected array index" >&2; exit 2 ;;
esac

RUN_ID="main_smoke_${MODE}_practical"
run_py src/run_inference.py \
  --experiment main --split smoke --mode "$MODE" --protocol practical
run_py src/score_outputs.py \
  --input "outputs/generations/${RUN_ID}.jsonl" \
  --output "outputs/scores/${RUN_ID}.jsonl" \
  --parquet "outputs/scores/${RUN_ID}.parquet"
echo "Smoke score layers written. Resubmit unchanged if the one-hour job timed out."

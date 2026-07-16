#!/bin/bash
#SBATCH --job-name=para-r0
#SBATCH --array=0-3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
#SBATCH --mem=96G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/round0_%A_%a.out
#SBATCH --error=logs/round0_%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail
PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/slurm/common.sh"
record_environment "round0_${SLURM_ARRAY_TASK_ID}"
run_py src/preflight.py --stage pilot

case "$SLURM_ARRAY_TASK_ID" in
  0) MODE=non_thinking; PROTOCOL=practical ;;
  1) MODE=thinking; PROTOCOL=practical ;;
  2) MODE=non_thinking; PROTOCOL=controlled ;;
  3) MODE=thinking; PROTOCOL=controlled ;;
  *) echo "Unexpected array index" >&2; exit 2 ;;
esac

if [[ "$PROTOCOL" == "practical" ]]; then
  run_py src/run_inference.py --experiment main --split smoke --mode "$MODE" --protocol "$PROTOCOL"
fi
RUN_ID="main_pilot_${MODE}_${PROTOCOL}"
run_py src/run_inference.py --experiment main --split pilot --mode "$MODE" --protocol "$PROTOCOL"
run_py src/score_outputs.py \
  --input "outputs/generations/${RUN_ID}.jsonl" \
  --output "outputs/scores/${RUN_ID}.jsonl" \
  --parquet "outputs/scores/${RUN_ID}.parquet"
echo "Score layers written. Use slurm/adjudicate_scores.sbatch for any queued disagreements."

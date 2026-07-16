#!/bin/bash
#SBATCH --job-name=para-r2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
#SBATCH --mem=96G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/round2_%j.out
#SBATCH --error=logs/round2_%j.err
#SBATCH --open-mode=append

set -euo pipefail
PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/slurm/common.sh"
record_environment round2
run_py src/preflight.py --stage confirmatory
run_py src/run_inference.py --experiment main --split confirmatory --mode thinking --protocol practical
run_py src/score_outputs.py --input outputs/generations/main_confirmatory_thinking_practical.jsonl \
  --output outputs/scores/main_confirmatory_thinking_practical.jsonl \
  --parquet outputs/scores/main_confirmatory_thinking_practical.parquet
echo "Score layers written. Adjudicate queued disagreements before analysis."

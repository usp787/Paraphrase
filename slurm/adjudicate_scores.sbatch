#!/bin/bash
#SBATCH --job-name=para-judge
# Confirmatory adjudication can include thousands of outputs, so it is a longer
# H200 workload rather than an invalid multi-hour H100 sharing request.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/judge_%j.out
#SBATCH --error=logs/judge_%j.err
#SBATCH --open-mode=append

set -euo pipefail
PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/slurm/common.sh"
record_environment adjudication

# Example: SCORE_STEM=main_confirmatory_non_thinking_practical sbatch ...
: "${SCORE_STEM:?Export SCORE_STEM without the outputs/scores/ prefix or extension}"
SCORES="outputs/scores/${SCORE_STEM}.jsonl"
GENERATIONS="outputs/generations/${SCORE_STEM}.jsonl"
run_py src/judge_scores.py --scores "$SCORES"
run_py src/score_outputs.py --force --input "$GENERATIONS" --output "$SCORES" \
  --parquet "outputs/scores/${SCORE_STEM}.parquet"
run_py src/preflight.py --stage results --input "$SCORES"

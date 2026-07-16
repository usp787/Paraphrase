#!/bin/bash
#SBATCH --job-name=para-analysis
#SBATCH --partition=sharing
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --open-mode=append

set -euo pipefail
PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/slurm/common.sh"
record_environment analysis
run_py src/analyze_results.py \
  --scores outputs/scores/main_confirmatory_non_thinking_practical.jsonl \
  --scores outputs/scores/main_confirmatory_thinking_practical.jsonl \
  --scores outputs/scores/scop_confirmatory_non_thinking_practical.jsonl
run_py src/analyze_ppcv.py

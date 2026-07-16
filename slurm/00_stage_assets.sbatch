#!/bin/bash
#SBATCH --job-name=para-stage
# Hub resolution/download and dataset preparation do not execute the models.
# Use Explorer's CPU queue instead of waiting for an H100/H200.
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/stage_%j.out
#SBATCH --error=logs/stage_%j.err
#SBATCH --open-mode=append

set -euo pipefail
PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/slurm/common.sh"
record_environment stage

# Use a narrow group to limit scratch/network pressure: paraphrase before form
# construction, primary before Round 0, ppcv before Round 4, external before Round 5.
GROUP="${GROUP:-primary}"
run_py src/stage_assets.py --group "$GROUP"

if [[ "${PREPARE_DATA:-1}" == "1" ]]; then
  if [[ -f data/manifest.jsonl ]]; then
    echo "data/manifest.jsonl already exists; preserving frozen split"
  else
    run_py src/prepare_data.py
  fi
fi

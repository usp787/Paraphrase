#!/bin/bash
#SBATCH --job-name=para-stage
#SBATCH --partition=sharing
#SBATCH --gres=gpu:h100:1
# Explorer caps H100 jobs in sharing at one hour. Stage one model group per
# submission; Hugging Face downloads resume from the scratch cache.
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/stage_%j.out
#SBATCH --error=logs/stage_%j.err
#SBATCH --open-mode=append

set -euo pipefail
PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/slurm/common.sh"
record_environment stage

# Use a narrow group so staging fits the sharing limit: paraphrase before form
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

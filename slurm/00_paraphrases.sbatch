#!/bin/bash
#SBATCH --job-name=para-forms
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
#SBATCH --mem=96G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/paraphrases_%j.out
#SBATCH --error=logs/paraphrases_%j.err
#SBATCH --open-mode=append

set -euo pipefail
PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/slurm/common.sh"
record_environment paraphrases

SPLIT="${SPLIT:-smoke}"
for ATTEMPT in 1 2 3 4; do
  run_py src/generate_paraphrases.py --stage generate --split "$SPLIT" --attempt "$ATTEMPT"
  run_py src/generate_paraphrases.py --stage judge --split "$SPLIT"
done
run_py src/validate_paraphrases.py --split "$SPLIT"

if [[ "$SPLIT" == "confirmatory" && ! -f data/semantic_audit.csv ]]; then
  run_py src/build_semantic_audit.py
  echo "Human semantic audit created. Complete data/semantic_audit.csv before Round 1."
fi

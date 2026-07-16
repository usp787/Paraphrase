#!/bin/bash
#SBATCH --job-name=para-forms
# Confirmatory form generation is long enough to justify the H200 queue. Smoke
# and pilot use 00_paraphrases_short.sbatch on a one-hour sharing GPU instead.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
# Keep the guideline's 07:45 allocation for confirmatory generation. JSONL and
# Hugging Face caches make an unchanged resubmission resumable if needed.
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

SPLIT="${SPLIT:-confirmatory}"
if [[ "$SPLIT" != "confirmatory" ]]; then
  echo "Use slurm/00_paraphrases_short.sbatch for SPLIT=$SPLIT." >&2
  exit 2
fi
for ATTEMPT in 1 2 3 4; do
  run_py src/generate_paraphrases.py --stage generate --split "$SPLIT" --attempt "$ATTEMPT"
  run_py src/generate_paraphrases.py --stage judge --split "$SPLIT"
done
run_py src/validate_paraphrases.py --split "$SPLIT"

if [[ "$SPLIT" == "confirmatory" && ! -f data/semantic_audit.csv ]]; then
  run_py src/build_semantic_audit.py
  echo "Human semantic audit created. Complete data/semantic_audit.csv before Round 1."
fi

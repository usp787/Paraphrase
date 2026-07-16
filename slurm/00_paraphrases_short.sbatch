#!/bin/bash
#SBATCH --job-name=para-forms-short
# The 2026-07-16 monitor showed an idle L40S and a fully allocated H100 node.
# Re-check the monitor; `sbatch --gres=gpu:a100:1 ...` is a safe override.
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/paraphrases_short_%j.out
#SBATCH --error=logs/paraphrases_short_%j.err
#SBATCH --open-mode=append

set -euo pipefail
PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/slurm/common.sh"
record_environment paraphrases_short

SPLIT="${SPLIT:-smoke}"
case "$SPLIT" in
  smoke|pilot) ;;
  *)
    echo "Short GPU profile only accepts SPLIT=smoke or SPLIT=pilot." >&2
    echo "Use slurm/00_paraphrases.sbatch for confirmatory generation." >&2
    exit 2
    ;;
esac

for ATTEMPT in 1 2 3 4; do
  run_py src/generate_paraphrases.py --stage generate --split "$SPLIT" --attempt "$ATTEMPT"
  run_py src/generate_paraphrases.py --stage judge --split "$SPLIT"
done
run_py src/validate_paraphrases.py --split "$SPLIT"

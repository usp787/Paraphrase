#!/bin/bash
#SBATCH --job-name=para-r4
#SBATCH --array=0-1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
#SBATCH --mem=120G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/round4_%A_%a.out
#SBATCH --error=logs/round4_%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail
PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/slurm/common.sh"
record_environment "round4_${SLURM_ARRAY_TASK_ID}"
run_py src/preflight.py --stage confirmatory

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  run_py src/run_ppcv.py --variant fidelity
else
  run_py src/run_ppcv.py --variant lite
fi

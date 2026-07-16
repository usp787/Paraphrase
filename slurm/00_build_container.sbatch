#!/bin/bash
#SBATCH --job-name=para-build
#SBATCH --partition=sharing
#SBATCH --gres=gpu:h100:1
# Explorer caps H100 jobs in sharing at one hour. The build is mostly image and
# package I/O; cached layers make a resubmission cheap if the first pull times out.
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err

set -euo pipefail
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
command -v apptainer >/dev/null 2>&1 || { echo "Apptainer module not found" >&2; exit 2; }

PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SCRATCH_ROOT="${SCRATCH:-/scratch/$USER}"
SIF="${PARAPHRASE_SIF:-$SCRATCH_ROOT/paraphrase_robustness/container/paraphrase_vllm_0.11.0.sif}"
mkdir -p "$(dirname "$SIF")" "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

# Explorer-style clusters commonly allow --fakeroot. If yours does not, build
# the same definition with the university's remote builder and set PARAPHRASE_SIF.
apptainer build --fakeroot "$SIF" environment/paraphrase.def
apptainer exec --nv "$SIF" python3 -c "import torch, vllm; print('cuda', torch.cuda.is_available(), 'vllm', vllm.__version__)"

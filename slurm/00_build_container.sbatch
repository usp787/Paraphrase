#!/bin/bash
#SBATCH --job-name=para-build
# Container construction is CPU/network/storage work. Do not hold a scarce GPU.
# Explorer's short partition is the general CPU queue (48-hour maximum).
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32G
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
# Import-test the CUDA-enabled image without reserving or exposing a physical GPU.
# The first GPU smoke job performs the real driver/device compatibility check.
apptainer exec "$SIF" python3 -c \
  "import torch, vllm; print('torch_cuda_build', torch.version.cuda, 'device_visible', torch.cuda.is_available(), 'vllm', vllm.__version__)"

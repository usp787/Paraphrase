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
if ! command -v apptainer >/dev/null 2>&1; then
  echo "Apptainer executable not found on $(hostname)." >&2
  echo "Explorer normally provides the 'apptainer' command directly; no module is required." >&2
  echo "Check with: command -v apptainer && apptainer --version" >&2
  exit 127
fi

PROJECT_ROOT="${PARAPHRASE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SCRATCH_ROOT="${SCRATCH:-/scratch/$USER}"
EXPERIMENT_SCRATCH="${PARAPHRASE_SCRATCH:-$SCRATCH_ROOT/paraphrase_robustness}"
SIF="${PARAPHRASE_SIF:-$EXPERIMENT_SCRATCH/container/paraphrase_vllm_0.11.0.sif}"

# Apptainer needs enough host-side scratch space for an unpacked image while
# building. Honor an existing cluster/user setting, but always create it first.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH_ROOT/apptainer/tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH_ROOT/apptainer/cache}"
mkdir -p \
  "$APPTAINER_TMPDIR" \
  "$APPTAINER_CACHEDIR" \
  "$(dirname "$SIF")" \
  "$PROJECT_ROOT/logs"
if [[ ! -w "$APPTAINER_TMPDIR" || ! -w "$APPTAINER_CACHEDIR" ]]; then
  echo "Apptainer scratch directories are not writable:" >&2
  echo "  APPTAINER_TMPDIR=$APPTAINER_TMPDIR" >&2
  echo "  APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR" >&2
  exit 2
fi
cd "$PROJECT_ROOT"

echo "apptainer=$(command -v apptainer)"
apptainer --version
echo "APPTAINER_TMPDIR=$APPTAINER_TMPDIR"
echo "APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR"
df -h "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# Explorer-style clusters commonly allow --fakeroot. If yours does not, build
# the same definition with the university's remote builder and set PARAPHRASE_SIF.
apptainer build --fakeroot "$SIF" environment/paraphrase.def
# Import-test the CUDA-enabled image without reserving or exposing a physical GPU.
# Disable the host user site here too, so this check tests the image rather than
# an incompatible package under $HOME/.local.  The first GPU smoke job performs
# the real driver/device compatibility check.
apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 "$SIF" python3 -c \
  "import torch, vllm; print('torch_cuda_build', torch.version.cuda, 'device_visible', torch.cuda.is_available(), 'vllm', vllm.__version__, 'vllm_path', vllm.__file__); assert vllm.__version__ == '0.11.0'"

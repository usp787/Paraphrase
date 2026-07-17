#!/bin/bash
# Shared Apptainer launcher. Source this after `set -euo pipefail`.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_ROOT="${SCRATCH:-/scratch/$USER}"
EXPERIMENT_SCRATCH="${PARAPHRASE_SCRATCH:-$SCRATCH_ROOT/paraphrase_robustness}"
SIF="${PARAPHRASE_SIF:-$EXPERIMENT_SCRATCH/container/paraphrase_vllm_0.11.0.sif}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH_ROOT/apptainer/tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH_ROOT/apptainer/cache}"

mkdir -p \
  "$PROJECT_ROOT/logs" \
  "$EXPERIMENT_SCRATCH/hf_cache" \
  "$EXPERIMENT_SCRATCH/tmp" \
  "$APPTAINER_TMPDIR" \
  "$APPTAINER_CACHEDIR"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Apptainer executable not found on $(hostname)." >&2
  echo "Explorer normally provides the command directly; do not load an apptainer module." >&2
  exit 127
fi

# Only expose host NVIDIA devices when Slurm actually granted a GPU. This lets
# build/staging/analysis run on ordinary CPU nodes without an erroneous --nv.
_HAS_GPU_ALLOCATION=0
for _gpu_value in \
  "${SLURM_JOB_GPUS:-}" "${SLURM_STEP_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"; do
  if [[ -n "$_gpu_value" && "$_gpu_value" != "NoDevFiles" ]]; then
    _HAS_GPU_ALLOCATION=1
  fi
done
_APPTAINER_GPU_ARGS=()
if [[ "$_HAS_GPU_ALLOCATION" == "1" ]]; then
  module load cuda/12.8.0 2>/dev/null || true
  _APPTAINER_GPU_ARGS=(--nv)
fi
if [[ ! -f "$SIF" ]]; then
  echo "Missing container: $SIF" >&2
  echo "Submit slurm/00_build_container.sbatch or export PARAPHRASE_SIF." >&2
  exit 2
fi

export APPTAINERENV_HF_HOME="$EXPERIMENT_SCRATCH/hf_cache"
export APPTAINERENV_HF_DATASETS_CACHE="$EXPERIMENT_SCRATCH/hf_cache/datasets"
export APPTAINERENV_TMPDIR="$EXPERIMENT_SCRATCH/tmp"
export APPTAINERENV_PYTHONPATH="$PROJECT_ROOT"
# Explorer bind-mounts $HOME into the container by default.  Without this,
# Python may import incompatible packages from $HOME/.local instead of the SIF.
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
export APPTAINERENV_MPLBACKEND=Agg
if [[ "$_HAS_GPU_ALLOCATION" == "1" ]]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
else
  unset APPTAINERENV_CUDA_VISIBLE_DEVICES
fi
for _slurm_name in \
  SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_PARTITION SLURM_JOB_NODELIST \
  SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_SUBMIT_DIR; do
  if [[ -n "${!_slurm_name:-}" ]]; then
    export "APPTAINERENV_${_slurm_name}=${!_slurm_name}"
  fi
done
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
fi

_ACTIVE_CHILD=""
_STOP_REQUESTED=0
_forward_signal() {
  local signal_name="$1"
  _STOP_REQUESTED=1
  if [[ -n "$_ACTIVE_CHILD" ]] && kill -0 "$_ACTIVE_CHILD" 2>/dev/null; then
    kill -s "$signal_name" "$_ACTIVE_CHILD" 2>/dev/null || true
  fi
}
trap '_forward_signal USR1' USR1
trap '_forward_signal TERM' TERM

run_py() {
  local status
  set +e
  apptainer exec "${_APPTAINER_GPU_ARGS[@]}" --cleanenv \
    --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" \
    --pwd "$PROJECT_ROOT" \
    "$SIF" python3 -u "$@" &
  _ACTIVE_CHILD=$!
  wait "$_ACTIVE_CHILD"
  status=$?
  if kill -0 "$_ACTIVE_CHILD" 2>/dev/null; then
    wait "$_ACTIVE_CHILD"
    status=$?
  fi
  _ACTIVE_CHILD=""
  set -e
  if [[ "$_STOP_REQUESTED" == "1" ]]; then
    echo "Scheduler stop signal handled; resubmit the same job to resume." >&2
    return 99
  fi
  return "$status"
}

record_environment() {
  # Fail early with a readable error if a host package leaks into the image.
  run_py -c '
from pathlib import Path
import vllm

expected = "0.11.0"
location = Path(vllm.__file__).resolve()
print(f"container_vllm={vllm.__version__} location={location}", flush=True)
if vllm.__version__ != expected:
    raise RuntimeError(
        f"Expected vLLM {expected} from the Apptainer image, but loaded "
        f"{vllm.__version__} from {location}"
    )
if ".local" in location.parts:
    raise RuntimeError(f"Host user-site package leaked into container: {location}")
'
  run_py src/record_environment.py --label "$1"
}

#!/bin/bash
# Shared Apptainer launcher. Source this after `set -euo pipefail`.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_ROOT="${SCRATCH:-/scratch/$USER}"
EXPERIMENT_SCRATCH="${PARAPHRASE_SCRATCH:-$SCRATCH_ROOT/paraphrase_robustness}"
SIF="${PARAPHRASE_SIF:-$EXPERIMENT_SCRATCH/container/paraphrase_vllm_0.11.0.sif}"

mkdir -p "$PROJECT_ROOT/logs" "$EXPERIMENT_SCRATCH/hf_cache" "$EXPERIMENT_SCRATCH/tmp"

if ! command -v apptainer >/dev/null 2>&1; then
  module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
fi
module load cuda/12.8.0 2>/dev/null || true
if ! command -v apptainer >/dev/null 2>&1; then
  echo "Apptainer is unavailable. Load the university's Apptainer module first." >&2
  exit 2
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
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
export APPTAINERENV_MPLBACKEND=Agg
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
  apptainer exec --nv --cleanenv \
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
  run_py src/record_environment.py --label "$1"
}

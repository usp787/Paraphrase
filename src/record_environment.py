"""Capture package, GPU, Slurm, and command metadata for a cluster round."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys

from src.common import atomic_write_json, resolve_path, utc_now


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, text=True, capture_output=True).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    freeze = command_output([sys.executable, "-m", "pip", "freeze"])
    resolve_path("environment/pip_freeze.txt").write_text(freeze + "\n", encoding="utf-8")
    slurm = {key: value for key, value in os.environ.items() if key.startswith("SLURM_")}
    payload = {
        "captured_at": utc_now(),
        "label": args.label,
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "slurm": slurm,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_home": os.environ.get("HF_HOME"),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID", "local")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    suffix = f"_{task_id}" if task_id else ""
    atomic_write_json(f"environment/slurm_job_metadata_{args.label}_{job_id}{suffix}.json", payload)


if __name__ == "__main__":
    main()

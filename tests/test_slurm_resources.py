import re
from pathlib import Path


def _directive(text: str, name: str) -> str | None:
    match = re.search(rf"^#SBATCH\s+--{re.escape(name)}=(\S+)", text, re.MULTILINE)
    return match.group(1) if match else None


def test_h100_sharing_jobs_never_exceed_one_hour():
    root = Path(__file__).resolve().parents[1] / "slurm"
    offenders = []
    for path in sorted(root.glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        is_h100_sharing = (
            _directive(text, "partition") == "sharing" and _directive(text, "gres") == "gpu:h100:1"
        )
        if is_h100_sharing and _directive(text, "time") != "01:00:00":
            offenders.append((path.name, _directive(text, "time")))
    assert offenders == []


def test_jobs_over_one_hour_use_h200_gpu_partition():
    root = Path(__file__).resolve().parents[1] / "slurm"
    offenders = []
    for path in sorted(root.glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        walltime = _directive(text, "time")
        if walltime and walltime > "01:00:00":
            resources = (_directive(text, "partition"), _directive(text, "gres"))
            if resources != ("gpu", "gpu:h200:1"):
                offenders.append(path.name)
    assert offenders == []

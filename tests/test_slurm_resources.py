import re
from pathlib import Path


def _directive(text: str, name: str) -> str | None:
    match = re.search(rf"^#SBATCH\s+--{re.escape(name)}=(\S+)", text, re.MULTILINE)
    return match.group(1) if match else None


def _minutes(walltime: str) -> int:
    hours, minutes, seconds = (int(part) for part in walltime.split(":"))
    assert seconds == 0
    return hours * 60 + minutes


def test_sharing_jobs_never_exceed_one_hour():
    root = Path(__file__).resolve().parents[1] / "slurm"
    offenders = []
    for path in sorted(root.glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        walltime = _directive(text, "time")
        if _directive(text, "partition") == "sharing" and walltime and _minutes(walltime) > 60:
            offenders.append((path.name, walltime))
    assert offenders == []


def test_long_gpu_jobs_use_h200_gpu_partition():
    root = Path(__file__).resolve().parents[1] / "slurm"
    offenders = []
    for path in sorted(root.glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        walltime = _directive(text, "time")
        gres = _directive(text, "gres")
        if walltime and gres and _minutes(walltime) > 60:
            resources = (_directive(text, "partition"), _directive(text, "gres"))
            if resources != ("gpu", "gpu:h200:1"):
                offenders.append(path.name)
    assert offenders == []


def test_cpu_only_jobs_do_not_reserve_gpus():
    root = Path(__file__).resolve().parents[1] / "slurm"
    cpu_jobs = {
        "00_build_container.sbatch",
        "00_stage_assets.sbatch",
        "analyze.sbatch",
    }
    offenders = []
    for name in sorted(cpu_jobs):
        text = (root / name).read_text(encoding="utf-8")
        if _directive(text, "partition") != "short" or _directive(text, "gres") is not None:
            offenders.append(name)
    assert offenders == []


def test_quick_l40s_profiles_fit_sharing_limit():
    root = Path(__file__).resolve().parents[1] / "slurm"
    quick_jobs = {
        "00_paraphrases_short.sbatch",
        "round0_smoke_l40s.sbatch",
    }
    offenders = []
    for name in sorted(quick_jobs):
        text = (root / name).read_text(encoding="utf-8")
        resources = (
            _directive(text, "partition"),
            _directive(text, "gres"),
            _directive(text, "time"),
        )
        if resources != ("sharing", "gpu:l40s:1", "01:00:00"):
            offenders.append((name, resources))
    assert offenders == []

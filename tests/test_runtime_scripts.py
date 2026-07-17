from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_apptainer_build_prepares_host_scratch_before_building():
    text = (ROOT / "slurm" / "00_build_container.sbatch").read_text(encoding="utf-8")
    assert 'export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH_ROOT/apptainer/tmp}"' in text
    assert (
        'export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH_ROOT/apptainer/cache}"' in text
    )
    assert text.index('mkdir -p \\\n  "$APPTAINER_TMPDIR"') < text.index(
        'apptainer build --fakeroot "$SIF"'
    )


def test_explorer_runtime_does_not_try_nonexistent_container_modules():
    for relative in ("slurm/00_build_container.sbatch", "slurm/common.sh"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "module load apptainer" not in text
        assert "module load singularity" not in text
        assert "command -v apptainer" in text


def test_apptainer_runtime_cannot_import_host_user_site_packages():
    common = (ROOT / "slurm" / "common.sh").read_text(encoding="utf-8")
    assert "export APPTAINERENV_PYTHONNOUSERSITE=1" in common
    assert 'expected = "0.11.0"' in common
    assert 'if ".local" in location.parts:' in common

    definition = (ROOT / "environment" / "paraphrase.def").read_text(encoding="utf-8")
    assert "export PYTHONNOUSERSITE=1" in definition

    build = (ROOT / "slurm" / "00_build_container.sbatch").read_text(encoding="utf-8")
    assert "apptainer exec --cleanenv --env PYTHONNOUSERSITE=1" in build


def test_readme_uses_container_python3_entrypoint():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'apptainer exec "$PARAPHRASE_SIF" python ' not in text
    assert 'apptainer exec "$PARAPHRASE_SIF" python3 ' in text


def test_datasets_and_pyarrow_pins_are_resolver_compatible():
    requirements = (ROOT / "environment" / "requirements.runtime.txt").read_text(encoding="utf-8")
    assert "datasets==4.5.0" in requirements
    assert "pyarrow==21.0.0" in requirements
    assert "pyarrow==19.0.1" not in requirements

    definition = (ROOT / "environment" / "paraphrase.def").read_text(encoding="utf-8")
    assert "import datasets, math_verify, pyarrow" in definition

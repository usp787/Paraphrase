# Budgeted Paraphrase Robustness in Reasoning LLMs

This repository is a runnable, inference-first implementation of
[`EXPERIMENT_GUIDELINE.md`](EXPERIMENT_GUIDELINE.md). It measures whether
meaning-preserving surface changes produce real item-level instability after
sampling noise and answer-extraction artifacts are controlled, then compares
self-consistency, paraphrase ensembling (SCoP), and a reduced PPCV intervention
under explicit cost budgets.

The implementation is complete; experiment results are not. GPU jobs must run on
the university cluster after the human semantic audit is completed.

## What is implemented

- Qwen3-8B thinking versus non-thinking with the official mode-specific decoding
  settings and a same-decoding pilot ablation.
- Frozen, disjoint 5/40/200 smoke/pilot/confirmatory splits for both
  `apple/GSM-Symbolic` and `HuggingFaceH4/MATH-500`.
- Original plus seven typed paraphrases, with deterministic gates, an independent
  semantic judge, retained rejection provenance, and a blinded human audit.
- Main measurement: five forms x three seeds per item.
- Fixed eight-answer allocations: SC-8, SCoP-2x4, SCoP-4x2, and SCoP-8x1.
- Reduced PPCV fidelity (50 GSM-Symbolic items) and PPCV-lite (100 items), including
  teacher-forced token discrepancy, critical-position selection, alternative-token
  rollouts, and similarity-weighted cross-paraphrase selection.
- Raw exact, normalized parser, and canonical mathematical-equivalence scoring;
  only disagreements/undecidable cases enter a blinded LLM adjudication queue.
- Strict robustness, worst-form accuracy, answer/correctness flips, stochastic
  variance decomposition, calibration, error-detection AUROC, paired bootstrap,
  McNemar/Holm tests, efficiency metrics, and an accuracy-cost plot.
- Append-only JSONL, configuration fingerprints, immutable Hub revisions,
  scheduler-signal handling, and item-level PPCV/batch-level inference resume.

The preregistered decisions are restated in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Cluster contract

The Slurm resource pattern is taken from the accessible
`C:\Users\usp78\Desktop\on_policy_distillation` project:

- container build, Hub staging, dataset preparation, and analysis:
  `--partition=short` with no GPU request;
- short, resumable inference: `--partition=sharing`, `--gres=gpu:l40s:1`, and
  Explorer's one-hour sharing maximum;
- long rounds: `--partition=gpu`, `--gres=gpu:h200:1`, and the guideline's
  `07:45:00` walltime;
- CUDA module: `cuda/12.8.0` when available;
- model/dataset cache on `/scratch/$USER`, not the home quota;
- append-mode logs and resumable output.

These defaults reflect the supplied GPU Monitor snapshot from 2026-07-16 15:09:
the single four-GPU H100 node was fully allocated, one L40S node was idle, and
the queue reported 100 H200 requests. Availability changes continuously. Check
again immediately before submission:

```bash
sinfo -p gpu -O "NodeList,Gres:30,GresUsed:30"
squeue -u "$USER" -o "%.18i %.12P %.24j %.2t %.10M %.6D %R"
```

Explorer documents `short` as the general CPU queue, `sharing` as a one-hour
CPU/GPU queue, `gpu-short` as a two-hour GPU queue, and `gpu` as an eight-hour
maximum single-GPU queue. Command-line `sbatch` options override a script's
header. For example, if the monitor later shows an A100 free instead of an L40S:

```bash
sbatch --gres=gpu:a100:1 slurm/round0_smoke_l40s.sbatch
```

Keep `07:45:00` for a confirmatory run unless the pilot demonstrates that a
shorter request is safe. A measured shorter walltime can then be supplied with
`sbatch --time=04:00:00 ...` without editing the frozen experiment config.

That reference project uses Conda, not Apptainer. This repository adds an
Apptainer runtime while preserving its verified Slurm, module, H200, and scratch
conventions. If your local module or partition names differ, change only the
Slurm headers or export the overrides documented below; do not change experiment
configs to solve a scheduler mismatch.

## Repository layout

```text
configs/       frozen experimental choices and model settings
data/          generated manifest, accepted paraphrases, human audit
environment/   Apptainer definition, package lock, captured revisions/metadata
outputs/       append-only generations, scored rows, resume checkpoints
reports/       pilot/main reports and figures
slurm/         one setup job and one job per experimental round
src/           data, paraphrase, inference, PPCV, scoring, and analysis code
tests/         dependency-light unit tests
```

Generated data, model containers, outputs, and logs are intentionally ignored by
Git. Empty artifact directories are retained with `.gitkeep` files.

## One-time cluster setup

Run all commands from the repository root on the cluster. `logs/` must exist
before `sbatch`; it is already present in this repository.

1. Build the image on scratch:

   ```bash
   sbatch slurm/00_build_container.sbatch
   ```

   This is a CPU-only `short` job; Apptainer image construction does not need a
   GPU. The script assumes the cluster supports `apptainer build --fakeroot`. If
   it does not, build
   `environment/paraphrase.def` with the university remote builder and export its
   path for every submission:

   ```bash
   export PARAPHRASE_SIF=/scratch/$USER/path/to/paraphrase.sif
   ```

   Do not submit staging or GPU jobs until the build reports `COMPLETED` with
   `ExitCode 0:0`. Verify the image after the build:

   ```bash
   export PARAPHRASE_SIF="${PARAPHRASE_SIF:-/scratch/$USER/paraphrase_robustness/container/paraphrase_vllm_0.11.0.sif}"
   test -s "$PARAPHRASE_SIF" && apptainer inspect "$PARAPHRASE_SIF" >/dev/null && echo "container ready"
   ```

   Explorer supplies the `apptainer` executable directly; there is no
   `apptainer` or `singularity` module to load. The build script creates
   `$APPTAINER_TMPDIR` and `$APPTAINER_CACHEDIR` under scratch before building.
   The message about using a root-mapped namespace because the user is absent
   from `/etc/subuid` is informational. If the build later fails inside `%post`
   with a permission error, that is a separate fakeroot-policy problem and must
   be reported to Research Computing or handled with a remote/prebuilt image.

2. Resolve/download paraphraser and judge commits, then freeze the data split.
   Staging is a CPU/network job divided into narrow groups to reduce scratch and
   network pressure. Hugging Face downloads resume from the cache if resubmitted:

   ```bash
   sbatch --export=ALL,GROUP=paraphrase,PREPARE_DATA=1 slurm/00_stage_assets.sbatch
   ```

3. Construct only the smoke forms first. The job tries up to four candidates for
   any rejected pair and is safe to resubmit:

   ```bash
   sbatch --export=ALL,SPLIT=smoke slurm/00_paraphrases_short.sbatch
   ```

   This uses one L40S for at most one hour. Resubmit unchanged if it needs another
   slice. Do not generate pilot or confirmatory forms until the preceding stage
   passes.

4. Stage Qwen3-8B before the smoke inference:

   ```bash
   sbatch --export=ALL,GROUP=primary,PREPARE_DATA=0 slurm/00_stage_assets.sbatch
   ```

`src/stage_assets.py` writes immutable 40-character Hub commits to
`environment/model_revisions.json`. Confirmatory preflight rejects mutable or
missing revisions.

## Experimental rounds

Every confirmatory inference job requests `07:45:00`, receives `USR1` five
minutes before timeout,
flushes its active batch/item, and can be resubmitted unchanged.

### Round 0 — smoke, pilot, and controlled decoding

```bash
sbatch slurm/round0_smoke_l40s.sbatch
```

This two-task array runs the practical thinking/non-thinking smoke test on the
short L40S profile. After both smoke tasks pass, construct the pilot forms. Wait
for that job to pass, then submit the four H200 pilot tasks (practical plus the
same-decoding controlled ablation), capped at two simultaneous array tasks:

```bash
sbatch --export=ALL,SPLIT=pilot slurm/00_paraphrases_short.sbatch
sbatch slurm/round0_smoke_pilot.sbatch
```

Use the measured throughput to verify the confirmatory size before looking at
confirmatory outcomes:

```bash
apptainer exec "$PARAPHRASE_SIF" python3 src/estimate_budget.py \
  --generations outputs/generations/main_pilot_non_thinking_practical.jsonl \
  --responses-per-item 15
```

Record the go/no-go decision in [`reports/pilot_report.md`](reports/pilot_report.md).
If throughput requires a smaller confirmatory set, change the split plan and
regenerate the manifest before any confirmatory inference. Never resize after
viewing confirmatory accuracy.

Only after the pilot decision is frozen, generate confirmatory forms on the H200:

```bash
sbatch slurm/00_paraphrases.sbatch
```

When that job completes, finish `data/semantic_audit.csv` without looking at
model results. Allowed values are:

- `equivalence`: `equivalent`, `not_equivalent`, or `uncertain`;
- `transformation_label_correct`: `yes`, `no`, or `uncertain`.

Resolve every uncertain confirmatory case. This command must pass before Round 1:

```bash
apptainer exec "$PARAPHRASE_SIF" python3 src/build_semantic_audit.py --validate
```

### Rounds 1–3 — main comparison and fixed-budget allocation

```bash
sbatch slurm/round1_main_nonthinking.sbatch
sbatch slurm/round2_main_thinking.sbatch
sbatch slurm/round3_scop.sbatch
```

These jobs score outputs but do not spend their remaining wall clock loading a
second 7B judge. For every score stem, submit a separate adjudication job:

```bash
sbatch --export=ALL,SCORE_STEM=main_confirmatory_non_thinking_practical slurm/adjudicate_scores.sbatch
sbatch --export=ALL,SCORE_STEM=main_confirmatory_thinking_practical slurm/adjudicate_scores.sbatch
sbatch --export=ALL,SCORE_STEM=scop_confirmatory_non_thinking_practical slurm/adjudicate_scores.sbatch
```

The adjudication job re-scores from the immutable generation JSONL, merges blinded
decisions, and enforces truncation, parser/judge, answer-format, and duplicate-key
failure gates.

Round 3 is reported twice: once at the fixed eight-answer budget and once with
each item's observed SC-8 generated-token spend as a cap. The cost-matched row
order is frozen in `PREREGISTRATION.md` and never uses correctness.

### Round 4 — PPCV

Stage the primary and embedding models in a CPU-only job, then submit the
two-task array (fidelity and lite are separate H200 allocations):

```bash
sbatch --export=ALL,GROUP=ppcv,PREPARE_DATA=0 slurm/00_stage_assets.sbatch
sbatch slurm/round4_ppcv.sbatch
```

PPCV output is item-level JSONL with nested full rollouts. The fidelity variant
includes SC-48; PPCV-lite is compared against Round 3 at measured token/GPU cost.

### Round 5 — external validity

Stage only the external-validity model in its own CPU-only job, then:

```bash
sbatch --export=ALL,GROUP=external,PREPARE_DATA=0 slurm/00_stage_assets.sbatch
sbatch slurm/round5_external.sbatch
sbatch --export=ALL,SCORE_STEM=external_confirmatory_reasoning_practical slurm/adjudicate_scores.sbatch
```

### Analysis

```bash
sbatch slurm/analyze.sbatch
```

Analysis is CPU-only on `short`; it never reserves a GPU.

Outputs:

- `reports/main_results.json`
- `reports/main_results.md`
- `reports/figures/accuracy_cost_pareto.png`
- `reports/ppcv_results.json`
- `reports/ppcv_report.md`

GSM-Symbolic and MATH-500 are analyzed separately. Conditional disagreement/error
subsets must remain explicitly labeled and must not replace the random-sample result.

## Resume and immutability behavior

The main/SCoP key is `(dataset, item_id, form_id, mode, seed, method)`. PPCV
checkpoints after each item. Generation JSONL is append-only and fsynced after
each batch. A checkpoint stores the expected count and config fingerprint.

- Same command after timeout: completed keys are skipped.
- Same output with changed config: rejected.
- Existing frozen manifest: preserved unless `--force` is explicit.
- Duplicate accepted paraphrase or generation keys: rejected.
- Missing/uncertain human audit: confirmatory run rejected.

Do not edit configs after confirmatory generation begins. If a scientifically
necessary change is made, use a new run ID and report it as a separate experiment.

## Local verification

GPU and Hub operations are cluster-only. Dependency-light logic can be checked
locally:

```bash
python -m pytest
python -m compileall -q src tests
ruff check src tests
```

## Important caveats

- The accepted GSM-Symbolic dataset is non-commercial (`CC BY-NC-ND 4.0`); verify
  that the intended dissemination complies with its license.
- PPCV is a preprint-level target and this is a reduced Qwen3-8B reproduction of
  a reported Qwen3-32B experiment. A null or cost-negative result is valid.
- The automatic number/unit/symbol gates are conservative. They are rejection
  filters, not proof of equivalence; the independent judge and blinded human audit
  remain mandatory.
- LoRA is intentionally absent. The guideline permits training only after the
  inference study identifies a stable failure that cheaper calibration, SCoP, or
  conditional PPCV cannot recover.

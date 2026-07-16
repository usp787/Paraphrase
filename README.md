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

- long rounds: `--partition=gpu`, `--gres=gpu:h200:1`;
- short/staging rounds: `--partition=sharing`, `--gres=gpu:h100:1`, with
  Explorer's maximum wall time of exactly one hour;
- CUDA module: `cuda/12.8.0` when available;
- model/dataset cache on `/scratch/$USER`, not the home quota;
- append-mode logs and resumable output.

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

   This is a one-hour H100 sharing job. The script assumes the cluster supports
   `apptainer build --fakeroot`. If it does not, build
   `environment/paraphrase.def` with the university remote builder and export its
   path for every submission:

   ```bash
   export PARAPHRASE_SIF=/scratch/$USER/path/to/paraphrase.sif
   ```

2. Resolve/download paraphraser and judge commits, then freeze the data split.
   Staging is divided into narrow groups so each H100 submission stays within one
   hour. Hugging Face downloads resume from the scratch cache if resubmitted:

   ```bash
   sbatch --export=ALL,GROUP=paraphrase,PREPARE_DATA=1 slurm/00_stage_assets.sbatch
   ```

3. Construct forms in increasing-cost order. Each job tries up to four candidates
   for any rejected pair and is safe to resubmit:

   ```bash
   sbatch --export=ALL,SPLIT=smoke slurm/00_paraphrases.sbatch
   sbatch --export=ALL,SPLIT=pilot slurm/00_paraphrases.sbatch
   sbatch --export=ALL,SPLIT=confirmatory slurm/00_paraphrases.sbatch
   ```

4. Complete `data/semantic_audit.csv` without looking at model results. Allowed
   values are:

   - `equivalence`: `equivalent`, `not_equivalent`, or `uncertain`;
   - `transformation_label_correct`: `yes`, `no`, or `uncertain`.

   Resolve all uncertain confirmatory cases. This command must pass:

   ```bash
   apptainer exec "$PARAPHRASE_SIF" python src/build_semantic_audit.py --validate
   ```

5. Stage Qwen3-8B before requesting the first H200 round:

   ```bash
   sbatch --export=ALL,GROUP=primary,PREPARE_DATA=0 slurm/00_stage_assets.sbatch
   ```

`src/stage_assets.py` writes immutable 40-character Hub commits to
`environment/model_revisions.json`. Confirmatory preflight rejects mutable or
missing revisions.

## Experimental rounds

Every long job requests `07:45:00`, receives `USR1` five minutes before timeout,
flushes its active batch/item, and can be resubmitted unchanged.

### Round 0 — smoke, pilot, and controlled decoding

```bash
sbatch slurm/round0_smoke_pilot.sbatch
```

This four-task array runs practical thinking/non-thinking pilots plus the
same-decoding controlled ablation. Use the measured throughput to verify the
confirmatory size before looking at confirmatory outcomes:

```bash
apptainer exec "$PARAPHRASE_SIF" python src/estimate_budget.py \
  --generations outputs/generations/main_pilot_non_thinking_practical.jsonl \
  --responses-per-item 15
```

Record the go/no-go decision in [`reports/pilot_report.md`](reports/pilot_report.md).
If throughput requires a smaller confirmatory set, change the split plan and
regenerate the manifest before any confirmatory inference. Never resize after
viewing confirmatory accuracy.

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

Stage the primary and embedding models in a one-hour H100 job, then submit the
two-task array (fidelity and lite are separate H200 allocations):

```bash
sbatch --export=ALL,GROUP=ppcv,PREPARE_DATA=0 slurm/00_stage_assets.sbatch
sbatch slurm/round4_ppcv.sbatch
```

PPCV output is item-level JSONL with nested full rollouts. The fidelity variant
includes SC-48; PPCV-lite is compared against Round 3 at measured token/GPU cost.

### Round 5 — external validity

Stage only the external-validity model in its own one-hour H100 job, then:

```bash
sbatch --export=ALL,GROUP=external,PREPARE_DATA=0 slurm/00_stage_assets.sbatch
sbatch slurm/round5_external.sbatch
sbatch --export=ALL,SCORE_STEM=external_confirmatory_reasoning_practical slurm/adjudicate_scores.sbatch
```

### Analysis

```bash
sbatch slurm/analyze.sbatch
```

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

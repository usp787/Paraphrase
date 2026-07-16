# Reproduction Guideline: Budgeted Paraphrase Robustness in Reasoning LLMs

Last updated: 2026-07-16

## 1. Executive decision

The best reproduction target for this hardware is not another broad demonstration that LLMs are prompt-sensitive. That result is already well established. The more useful and current experiment is:

> Measure whether a modern hybrid reasoning model is genuinely brittle to meaning-preserving paraphrases after controlling for sampling and scoring artifacts, then compare self-consistency, paraphrase ensembling, and a reduced PPCV method under the same inference budget.

The primary model is `Qwen/Qwen3-8B`, evaluated in thinking and non-thinking modes. This gives a valuable within-model comparison because the weights are fixed and only the reasoning mode changes. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` is an optional external-validity model.

The primary tasks are mathematical reasoning problems with objective answer verification:

- GSM-Symbolic: 200 confirmatory items, plus a small pilot split.
- MATH-500: 200 confirmatory items, plus a small pilot split.

The core experiment is inference-only. Do not begin with LoRA. Recent peer-reviewed evidence shows that naive format-augmented LoRA can raise accuracy without reliably reducing prompt sensitivity. Training should be a later experiment only if the inference study reveals a stable, practically meaningful failure mode.

## 2. Why this target reflects the state of the field

The literature now has four distinct stages:

1. Measurement: semantically preserving changes can alter generated answers.
2. Exploitation: sampling across paraphrases can outperform sampling repeatedly from one prompt.
3. Reliability correction: some apparent sensitivity is caused by brittle answer extraction or log-likelihood scoring.
4. Fine-grained intervention: paraphrases can identify unstable tokens inside a reasoning trajectory and guide alternative rollouts.

This experiment deliberately covers stages 2 through 4 while retaining a strong measurement protocol.

### 2.1 Local-paper qualification

| Local paper | Reliability and relevance | Use in this experiment |
|---|---|---|
| ReCode, ACL 2023 | Peer-reviewed and strong, but code-generation-specific | Metric inspiration only; optional later code-domain validation |
| Paraphrase and Solve / SCoP, NAACL 2024 | Peer-reviewed, released code, directly relevant | Main inference baseline: distribute samples across paraphrases |
| GSM-Symbolic, ICLR 2025 | Peer-reviewed and reliable | Source of controllable math problems; symbolic variants are not themselves paraphrases |
| PBSS, arXiv 2025 | Interesting diagnostic, but the local version contains a placeholder code link and emphasizes embedding drift rather than correctness | Do not use as the main anchor |
| Mapping from Meaning, AAAI 2025 | Peer-reviewed, code released, strong fixed-budget and calibration design | Compare different allocations of a fixed sample budget across prompt forms |
| PPCV, ICLR 2026 submission / arXiv preprint | Newest and technically interesting, but not yet a reliable accepted result and no official code was located | Reproduce cautiously; label it as a preprint reproduction and test efficiency claims independently |

### 2.2 Essential external papers

- [Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1006/): motivates raw-parser versus canonical-verifier comparisons.
- [When Punctuation Matters, Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1109/): motivates postponing naive LoRA and evaluating cost as well as accuracy.
- [Brittlebench, arXiv 2026](https://arxiv.org/abs/2603.13285): supplies the variance-decomposition framing and perturbation taxonomy; it is current but still a preprint.
- [Measuring LLMs' Sensitivity to Paraphrased Opinion Prompts, WASSA 2026](https://aclanthology.org/2026.wassa-1.5/): suggests that alignment and reasoning design may matter more than model size, but its task is narrow.
- [Qwen3-8B official model card](https://huggingface.co/Qwen/Qwen3-8B): documents the same-model thinking switch and recommended decoding settings.
- [DeepSeek-R1-Distill-Qwen-7B official model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B): documents the Qwen2.5-Math base and reasoning distillation lineage.

## 3. Research questions and preregistered hypotheses

### RQ1 - Is there real paraphrase brittleness after scoring is repaired?

H1: Canonical answer verification will reduce measured sensitivity relative to raw exact match, but non-zero within-item answer and correctness variation will remain.

### RQ2 - Does reasoning mode change paraphrase robustness?

H2: Qwen3 thinking mode will change the fraction of variance attributable to surface form and improve strict robust accuracy relative to non-thinking mode. Treat direction and effect size as empirical; thinking may improve accuracy while still increasing stochastic variance or token cost.

### RQ3 - At fixed cost, where should inference samples be spent?

H3: Distributing a fixed sample budget across paraphrases will improve worst-form accuracy, error detection, or calibration relative to spending all samples on the original form.

### RQ4 - Does PPCV provide value beyond ordinary paraphrase ensembling?

H4: PPCV will be most helpful on items whose baseline trajectories are incorrect or disagree across paraphrases. Its advantage may disappear when compared at equal generated-token or wall-clock cost.

### RQ5 - Is any conclusion model-specific?

H5: The qualitative result on a held-out subset will transfer to DeepSeek-R1-Distill-Qwen-7B, although the effect size may differ.

## 4. Experimental variables

### 4.1 Models

Primary:

- `Qwen/Qwen3-8B`, thinking mode.
- `Qwen/Qwen3-8B`, non-thinking mode.

External confirmation:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` on a stratified 150-200 item subset.

Optional scale check only after the main result:

- A 32B model on 50-100 items. Do not make this part of the first-pass design. An H200 can hold a 32B dense model in BF16, but the long output traces and repeated rollouts, not weight memory, are likely to dominate the eight-hour window.

### 4.2 Datasets

Use disjoint deterministic splits created once and stored in a manifest:

| Split | GSM-Symbolic | MATH-500 | Purpose |
|---|---:|---:|---|
| Smoke | 5 | 5 | Parsing, chat template, resume behavior |
| Pilot | 40 | 40 | Throughput, token length, power and failure audit |
| Confirmatory | 200 | 200 | Main preregistered comparison |

If the measured throughput cannot support 400 confirmatory items, reduce the sample size using the formula in Section 9. Do not silently reduce the number after seeing outcome accuracy.

### 4.3 Surface forms

For the main robustness measurement, use the original plus four validated paraphrases:

1. lexical substitution;
2. syntactic restructuring;
3. clause or discourse reordering;
4. constrained free paraphrase.

For the fixed-budget SCoP experiment, generate seven paraphrases so there are eight total surface forms. Label every form by transformation type and generator.

Use a paraphraser that is not one of the evaluated target models, such as Mistral-7B-Instruct. Keep a small anchor set of released SCoP paraphrases if compatible data are available. This helps reveal whether LLM-generated paraphrases have been standardized into easier, model-preferred language.

## 5. Paraphrase construction and semantic validation

### 5.1 Generation contract

The paraphraser must:

- preserve every number, unit, named entity role, variable relationship, and requested unknown;
- avoid solving the problem or adding hints;
- avoid deleting apparently irrelevant details;
- change only the linguistic realization;
- return one paraphrase and no explanation.

Store the generator model, model revision, prompt hash, seed, sampling parameters, and output text.

### 5.2 Automatic gates

Reject and regenerate a paraphrase if any gate fails:

- the multiset of numbers differs from the original;
- units or mathematical symbols are added or removed;
- the requested target quantity changes;
- text is too similar to the original for its assigned type;
- an independent semantic-equivalence judge marks it non-equivalent or uncertain.

Embedding similarity may be recorded, but do not treat a cosine threshold as proof of semantic equivalence.

### 5.3 Human audit

Before the confirmatory run, manually audit at least 10% of problem-form pairs, with a minimum of 80 pairs. Blind the auditor to model results. Record:

- equivalent;
- not equivalent;
- uncertain;
- transformation type is correctly labeled.

Require at least 95% equivalent among non-uncertain audited pairs. Resolve all uncertain cases used in the confirmatory set. Report the audit rate and confidence interval.

## 6. Inference protocols

### 6.1 Main brittleness measurement

For each item `i`, form `p`, mode `m`, and seed `s`, generate an answer:

- 5 forms per item: original plus 4 paraphrases;
- 3 independent seeds per form;
- one eight-hour job for non-thinking mode;
- one eight-hour job for thinking mode.

Use vLLM with continuous batching. Pin the model revision and all library versions.

Main practical settings:

| Setting | Thinking | Non-thinking |
|---|---:|---:|
| temperature | 0.6 | 0.7 |
| top-p | 0.95 | 0.8 |
| top-k | 20 | 20 |
| max new tokens, GSM-Symbolic | 2,048 | 1,024 |
| max new tokens, MATH-500 | 4,096 | 2,048 |

These follow the model's intended operating modes but do not isolate reasoning mode from decoding configuration. Therefore, run a controlled ablation on the 80-item pilot split using the same `temperature=0.6`, `top_p=0.95`, and `top_k=20` in both modes. Report the practical and controlled comparisons separately.

Track truncation. If more than 5% of outputs hit the maximum length, raise the cap or narrow the dataset before the confirmatory run.

### 6.2 Fixed-budget SC versus SCoP

Use a total generation budget of eight answers per problem:

| Method | Number of forms `n_p` | Samples per form `n_s` | Total answers |
|---|---:|---:|---:|
| SC-8 | 1 | 8 | 8 |
| SCoP-2x4 | 2 | 4 | 8 |
| SCoP-4x2 | 4 | 2 | 8 |
| SCoP-8x1 | 8 | 1 | 8 |

Aggregate normalized final answers by majority vote. Resolve ties by summed normalized log probability if available; otherwise use a deterministic rule fixed before evaluation.

Sample count is not the final cost metric. Also report total input tokens, generated tokens, GPU seconds, and peak memory. Paraphrases can change input length and reasoning length.

### 6.3 PPCV fidelity check

The local PPCV paper is a preprint, so separate reproduction fidelity from efficiency modification.

On 50 randomly selected GSM-Symbolic confirmatory items:

- use Qwen3-8B non-thinking mode;
- generate `N=4` paraphrases;
- obtain one initial trajectory;
- teacher-force the initial trajectory under each paraphrased question;
- mark positions where the predicted top-1 token differs from the actual trajectory token;
- score candidate position `i` using the maximum top-1 probability minus the probability of the actual token across paraphrases;
- choose the highest-scoring position;
- include the original token and sample `K=10` top alternatives;
- truncate at that position and generate rollouts for the original and four paraphrases;
- select the answer with the highest similarity-weighted cross-paraphrase consistency;
- compare with SC-48, matching the paper's approximate rollout count.

This is a reduced-scale reproduction because the paper reports Qwen3-32B for its Qwen experiment. State that clearly.

### 6.4 PPCV-lite efficiency check

On 100 randomly selected items and a separately reported disagreement subset:

- `N=2` paraphrases;
- `K=3` alternative tokens, including the original token;
- one critical position only;
- compare against SC and SCoP using the same measured generated-token or wall-clock budget.

The PPCV paper's appendix reports only a slight accuracy reduction from top-10 to top-3 with lower latency. This claim is precisely what the efficiency check should attempt to verify.

Do not run PPCV only on baseline failures and present the result as an overall accuracy gain. Report:

1. performance on a random sample;
2. performance conditional on baseline disagreement or error;
3. the cost of identifying that conditional subset.

## 7. Answer verification and artifact controls

### 7.1 Three scoring layers

Score every output three ways:

1. raw exact match;
2. normalized parser match;
3. canonical mathematical equivalence.

For GSM-Symbolic, normalize commas, currency symbols, signs, percentages, fractions, and the final numeric answer.

For MATH-500, extract the final boxed answer, normalize LaTeX, and use symbolic equivalence where possible. Log parser failures rather than counting them silently as incorrect.

Use an LLM judge only for cases where raw, normalized, and symbolic scoring disagree or cannot decide. Blind the judge to prompt form, method, and model. Manually audit at least 100 judged cases or all judged cases if fewer than 100.

### 7.2 Why this control is essential

Average accuracy can remain almost unchanged even when individual examples flip from correct to incorrect and vice versa. Conversely, rigid extraction can manufacture apparent flips. Report both aggregate accuracy and paired, item-level instability after canonical verification.

## 8. Metrics

Let `Y[i,p,s]` be canonical correctness for item `i`, surface form `p`, and sampling seed `s`.

### 8.1 Accuracy and strict robustness

- Mean accuracy across all forms and seeds.
- Original-form accuracy.
- Mean paraphrase accuracy.
- Robust accuracy: fraction of items correct under every form using majority correctness over seeds.
- Worst-form accuracy: average over items of the minimum form-level solve rate.
- Answer-flip rate: fraction of items for which canonical answers differ across forms.
- Correctness-flip rate: fraction of items with at least one correct and at least one incorrect form.

### 8.2 Variance decomposition

Extend Brittlebench to stochastic generation:

```text
V_run  = E_item,form Var_seed(Y)
V_form = E_item Var_form(E_seed[Y])
V_item = Var_item(E_form,seed[Y])
Pi_form = V_form / (V_run + V_form + V_item)
```

Report all three components. This prevents sampling noise from being mislabeled as paraphrase brittleness.

### 8.3 Calibration and error detection

Use answer frequency within the eight-sample budget as confidence. Report:

- Brier score;
- expected calibration error with fixed bins and an adaptive-bin sensitivity check;
- AUROC for detecting an incorrect final answer using disagreement or entropy;
- paraphrase-attributable uncertainty versus within-form sampling uncertainty.

### 8.4 Efficiency

For every method, report:

- accuracy per million generated tokens;
- robust accuracy per GPU hour;
- average and p95 latency per item;
- total input and output tokens;
- peak GPU memory;
- truncation and failure rates.

Plot the Pareto frontier of canonical accuracy versus generated tokens. A method that is more accurate only because it spends five times the tokens is not an efficiency win.

## 9. Fitting every round into eight hours

Use the smoke and pilot runs to measure effective batched output throughput. Then compute the maximum item count for a job:

```text
N_max = floor(
  0.70 * 28,800 seconds * effective_output_tokens_per_second
  / (responses_per_item * mean_output_tokens)
)
```

The 0.70 factor reserves time for model loading, tokenization, long-tail generations, evaluation, and checkpoint flushing.

Operational rules:

- Request `07:45:00`, leaving cluster cleanup margin.
- Stage model weights and datasets before the H200 allocation starts when cluster policy permits.
- Write append-only JSONL after every completed batch.
- Flush a progress manifest every 25 items.
- Make every job resumable by unique key `(dataset, item_id, form_id, mode, seed, method)`.
- Use a scheduler signal five minutes before timeout if supported.
- Never wait until job end to write results.
- Record throughput separately for thinking and non-thinking modes.

### Recommended round schedule

| Round | Maximum scope | Go/no-go output |
|---|---|---|
| 0 | 10 smoke items plus 80 pilot items | Valid parsers, no duplicated keys, throughput and token-length estimate |
| 1 | Qwen3-8B non-thinking, main measurement | Canonical brittleness metrics and valid resume behavior |
| 2 | Qwen3-8B thinking, main measurement | Same metrics, plus cost-normalized mode comparison |
| 3 | SC-8 versus SCoP budget allocation | Best fixed-budget allocation and calibration result |
| 4 | PPCV 50-item fidelity check plus PPCV-lite if time remains | Reproduction direction, failure audit, efficiency comparison |
| 5 | DeepSeek-R1-Distill-Qwen-7B on 150-200 stratified items | External-validity check |
| 6 optional | Consistency-training or 32B scale check | Run only if Rounds 1-5 identify a stable open question |

## 10. Statistical analysis

- Freeze item IDs, paraphrases, scoring code, hypotheses, and primary metrics before confirmatory inference.
- Use paired bootstrap resampling over items with 10,000 resamples for 95% confidence intervals.
- Use McNemar's test for paired binary final-answer comparisons.
- Bootstrap the complete variance decomposition by item.
- Correct multiple pairwise method comparisons with Holm's method.
- Report effect sizes and confidence intervals, not only p-values.
- Analyze GSM-Symbolic and MATH-500 separately before reporting a weighted aggregate.
- Treat the disagreement or baseline-error subset as conditional analysis, not as the main test set.

Do not define "hard" items from the same samples later used to measure the treatment gain. If a hard subset is needed, identify it using pilot seeds and evaluate it with held-out seeds.

## 11. Failure gates

Stop and repair the pipeline before using more GPU time if any of these occur:

- semantic audit falls below 95%;
- parser or judge undecided rate exceeds 2%;
- more than 5% of outputs are truncated;
- duplicate or missing result keys occur;
- resume produces different configs for the same key;
- more than 10% of outputs violate the required answer format;
- paraphrases systematically shorten prompts or remove details, indicating an easier-language confound;
- one generator's paraphrases dominate all improvements, suggesting generator style bias.

## 12. Decision rules for interpreting the result

### Finding A - Mostly an evaluation artifact

If raw exact-match sensitivity is large but canonical correctness-flip rate and `Pi_form` become small, the main conclusion is that the evaluation pipeline was brittle. This still reproduces an important 2025 development and prevents a false robustness claim.

### Finding B - Real per-item brittleness with stable average accuracy

If mean accuracy barely changes but canonical answer or correctness flips remain common, report instance-level instability as the primary finding. Do not summarize it as "accuracy was unchanged, therefore the model is robust."

### Finding C - Thinking improves accuracy but not robustness

If thinking raises mean accuracy while `Pi_form`, flip rate, or cost-normalized robust accuracy does not improve, conclude that stronger reasoning and paraphrase invariance are distinct properties.

### Finding D - SCoP wins at fixed sample count but loses at fixed tokens

Conclude that paraphrase diversification is useful but its benefit is partly purchased by longer prompts or traces. Report both cost regimes.

### Finding E - PPCV wins only on disagreement cases

The practical result is a router: run a cheap paraphrase-consistency probe first and invoke PPCV only when disagreement is detected. This is more useful than applying PPCV to every request.

## 13. Optional training extension

Do not include this in the first reproduction. If a stable failure remains, train a Qwen3-8B LoRA with a consistency objective across paired paraphrases, not only ordinary augmented supervised examples.

Requirements:

- train paraphrases and test paraphrases must come from different generators or transformation templates;
- retain the original task loss;
- include a divergence penalty between answer distributions or hidden states for equivalent forms;
- evaluate clean accuracy, robust accuracy, and cross-domain transfer;
- compare against the untrained model and ordinary augmentation-only LoRA;
- use one short epoch and checkpoint by tokens, not steps.

This extension is justified only if the inference study shows that robustness cannot be recovered cheaply by calibration, SCoP, or conditional PPCV.

## 14. Reproducibility artifacts to save

Recommended output structure:

```text
configs/
  model_qwen3_8b.yaml
  experiment_main.yaml
  experiment_scop.yaml
  experiment_ppcv.yaml
data/
  manifest.jsonl
  paraphrases.jsonl
  semantic_audit.csv
outputs/
  generations/*.jsonl
  scores/*.parquet
  checkpoints/*.json
reports/
  pilot_report.md
  main_results.md
  figures/
environment/
  pip_freeze.txt
  model_revisions.json
  slurm_job_metadata.json
```

Every generation record should contain:

```text
run_id, dataset, item_id, form_id, transformation_type,
paraphrase_generator, target_model, model_revision, mode,
seed, sampling_config, prompt_hash, raw_output, parsed_answer,
canonical_answer, correctness, input_tokens, output_tokens,
latency_seconds, truncated, parser_status, judge_status
```

## 15. Minimum publishable result package

The experiment is complete when it contains:

1. a validated paraphrase set and semantic-audit report;
2. thinking versus non-thinking instance-level brittleness with confidence intervals;
3. raw versus canonical scoring to quantify evaluation artifacts;
4. SC versus SCoP at equal sample and equal token budgets;
5. a transparent PPCV reproduction or failure report;
6. one external-validity model or a clearly stated reason it was not run;
7. accuracy-cost Pareto plots and all configs needed to resume or rerun.

The most credible outcome is not necessarily a positive PPCV result. A careful finding that PPCV does not beat a simpler paraphrase ensemble at equal cost would be scientifically useful because PPCV is currently a preprint-level claim.

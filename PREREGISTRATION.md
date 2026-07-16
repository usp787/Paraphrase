# Preregistration

This file freezes the primary analysis before confirmatory inference. The complete
rationale and decision rules are in `EXPERIMENT_GUIDELINE.md`.

## Research questions and hypotheses

1. Canonical verification will reduce apparent sensitivity relative to raw exact
   match, but item-level answer and correctness variation will remain.
2. Qwen3 thinking mode will change surface-form-attributable variance and strict
   robust accuracy. Direction and magnitude are empirical; accuracy, robustness,
   sampling variance, and token cost are separate endpoints.
3. At eight answers per item, distributing samples across paraphrases will improve
   worst-form accuracy, error detection, or calibration relative to SC-8.
4. PPCV will help mainly on baseline errors/disagreements and may not beat ordinary
   paraphrase ensembling at equal generated-token or wall-clock cost.
5. A stratified DeepSeek-R1-Distill-Qwen-7B subset will test whether the qualitative
   main result transfers beyond Qwen3-8B.

## Frozen primary design

- Datasets: 200 confirmatory GSM-Symbolic and 200 confirmatory MATH-500 items.
- Main forms: original plus lexical, syntactic, discourse, and constrained-free.
- Samples: three preregistered seeds per item/form/mode.
- Modes: Qwen3-8B thinking and non-thinking; practical official decoding.
- Controlled ablation: 80-item pilot only, identical `temperature=0.6`,
  `top_p=0.95`, `top_k=20` in both modes.
- Primary correctness: canonical mathematical equivalence after blinded
  adjudication of parser-layer disagreement or undecidable cases.
- Equal-token SCoP sensitivity: use each item's observed SC-8 generated-token
  spend as its cap; admit competing-method rows in deterministic form/sample/seed
  order without consulting correctness.

## Primary metrics

- mean, original-form, and mean-paraphrase accuracy;
- strict robust accuracy and worst-form accuracy;
- canonical answer-flip and correctness-flip rates;
- `V_run`, `V_form`, `V_item`, and `Pi_form`;
- SC/SCoP final accuracy, Brier, fixed/adaptive ECE, incorrect-answer AUROC;
- total input/output tokens, GPU seconds, latency, peak memory, truncation;
- accuracy per generated-token budget and robust accuracy per GPU hour.

## Statistical tests

- paired item bootstrap with 10,000 resamples and 95% intervals;
- McNemar test for paired binary comparisons;
- item-bootstrap variance decomposition;
- Holm correction across pairwise method comparisons;
- separate dataset results before any weighted aggregate.

## Locked interpretation constraints

- Do not resize after viewing confirmatory accuracy.
- Do not define hard items from the evaluation seeds used to measure a treatment.
- Do not present a baseline-failure-only PPCV analysis as overall improvement.
- Do not call a sample-count win an efficiency win without token/GPU cost.
- Do not interpret stable mean accuracy as robustness when item-level flips remain.

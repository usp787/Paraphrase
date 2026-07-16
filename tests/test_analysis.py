from src.analyze_results import analyze


def test_analysis_reports_scoring_layers_and_efficiency():
    rows = []
    for item_id in ("a", "b"):
        for form_id in ("original", "lexical"):
            for seed in (1, 2, 3):
                correct = not (item_id == "b" and form_id == "lexical")
                rows.append(
                    {
                        "dataset": "gsm_symbolic",
                        "mode": "non_thinking",
                        "method": "main",
                        "item_id": item_id,
                        "form_id": form_id,
                        "seed": seed,
                        "correctness": correct,
                        "parsed_answer": "1" if correct else "2",
                        "raw_exact_match": correct,
                        "normalized_match": correct,
                        "canonical_match": correct,
                        "judge_required": False,
                        "answer_format_ok": True,
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "gpu_seconds": 0.5,
                        "latency_seconds": 0.5,
                        "peak_gpu_memory_gib": 10.0,
                        "truncated": False,
                        "raw_output": "answer",
                        "finish_reason": "stop",
                    }
                )
    config = {"analysis": {"bootstrap_resamples": 20, "bootstrap_seed": 4, "ece_bins": 5}}
    result = analyze(rows, config)
    metrics = result["groups"]["gsm_symbolic/non_thinking/main"]
    assert metrics["raw_accuracy"] == metrics["mean_accuracy"]
    assert metrics["accuracy_per_million_generated_tokens"] > 0
    assert metrics["generation_failure_rate"] == 0

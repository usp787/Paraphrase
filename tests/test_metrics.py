import pytest

from src.metrics import bootstrap_variance_ci, fixed_budget_metrics, main_robustness_metrics


def main_rows():
    rows = []
    answers = {
        ("a", "original"): [True, True, True],
        ("a", "lexical"): [True, True, False],
        ("b", "original"): [True, False, False],
        ("b", "lexical"): [True, True, True],
    }
    for (item, form), values in answers.items():
        for seed, value in enumerate(values):
            rows.append(
                {
                    "item_id": item,
                    "form_id": form,
                    "correctness": value,
                    "parsed_answer": "1" if value else "2",
                    "seed": seed,
                }
            )
    return rows


def test_main_robustness_and_variance():
    metrics, per_item = main_robustness_metrics(main_rows())
    assert metrics["robust_accuracy"] == pytest.approx(0.5)
    assert metrics["correctness_flip_rate"] == pytest.approx(0.5)
    assert 0 <= metrics["pi_form"] <= 1
    assert per_item["a"]["robust"] is True
    intervals = bootstrap_variance_ci(main_rows(), resamples=20, seed=3)
    assert set(intervals) == {"v_run", "v_form", "v_item", "pi_form"}


def test_fixed_budget_majority_and_cost():
    rows = [
        {
            "item_id": "a",
            "parsed_answer": "1" if index < 6 else "2",
            "correctness": index < 6,
            "normalized_log_probability": -0.1,
            "input_tokens": 10,
            "output_tokens": 20,
            "gpu_seconds": 0.5,
        }
        for index in range(8)
    ]
    metrics, _ = fixed_budget_metrics(rows)
    assert metrics["final_accuracy"] == 1.0
    assert metrics["mean_output_tokens_per_item"] == 160

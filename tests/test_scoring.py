from src.scoring import extract_raw_answer, normalize_answer, score_record


def base_record(output):
    return {
        "dataset": "gsm_symbolic",
        "canonical_answer": "1/2",
        "gold_answer": "reasoning #### 1/2",
        "raw_output": output,
    }


def test_nested_box_extraction():
    answer, method = extract_raw_answer(r"Therefore, \boxed{\frac{1}{2}}.", "math_500")
    assert answer == r"\frac{1}{2}"
    assert method == "boxed"


def test_normalization_numeric_forms():
    assert normalize_answer("50%") == "0.5"
    assert normalize_answer("2/4") == "1/2"
    assert normalize_answer("1,250") == "1250"


def test_score_tracks_all_layers():
    scored = score_record(base_record(r"Work. Final: \boxed{1/2}"))
    assert scored["raw_exact_match"] is True
    assert scored["normalized_match"] is True
    assert scored["canonical_match"] is True
    assert scored["answer_format_ok"] is True

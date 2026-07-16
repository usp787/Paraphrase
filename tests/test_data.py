from src.prepare_data import deterministic_splits, gsm_final_answer, normalize_row


def test_gsm_answer_and_normalization():
    row = {"id": 4, "instance": 2, "question": "How many?", "answer": "work\n#### 1,250"}
    spec = {
        "question_field": "question",
        "answer_field": "answer",
        "source_id_fields": ["id", "instance"],
    }
    normalized = normalize_row("gsm_symbolic", row, 0, spec, "a" * 40)
    assert gsm_final_answer(row["answer"]) == "1250"
    assert normalized["canonical_answer"] == "1250"
    assert normalized["item_id"] == "gsm_symbolic:4:2"


def test_deterministic_splits_are_disjoint():
    rows = [{"item_id": f"id-{index}"} for index in range(20)]
    plan = {"smoke": 2, "pilot": 3, "confirmatory": 5}
    first = deterministic_splits([dict(row) for row in rows], plan, 7, "demo")
    second = deterministic_splits([dict(row) for row in rows], plan, 7, "demo")
    assert [(row["item_id"], row["split"]) for row in first] == [
        (row["item_id"], row["split"]) for row in second
    ]
    assert len({row["item_id"] for row in first}) == 10

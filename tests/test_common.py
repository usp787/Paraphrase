import json

import pytest

from src.common import append_jsonl, completed_keys, load_jsonl, record_key


def result(seed=1):
    return {
        "dataset": "gsm_symbolic",
        "item_id": "item-1",
        "form_id": "original",
        "mode": "thinking",
        "seed": seed,
        "method": "main",
    }


def test_jsonl_append_and_duplicate_detection(tmp_path):
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, [result(), result(2)])
    assert len(load_jsonl(path)) == 2
    seen, duplicates = completed_keys(path)
    assert len(seen) == 2
    assert duplicates == []
    append_jsonl(path, [result()])
    _, duplicates = completed_keys(path)
    assert duplicates == [record_key(result())]


def test_invalid_jsonl_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(result()) + "\n{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_jsonl(path)

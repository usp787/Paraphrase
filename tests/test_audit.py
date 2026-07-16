import pytest

from src.build_semantic_audit import wilson_interval


def test_wilson_interval_contains_observed_rate():
    lower, upper = wilson_interval(95, 100)
    assert lower < 0.95 < upper
    assert lower == pytest.approx(0.8882, abs=0.001)

import pytest
from evaluation.aggregation import aggregate_tool_score


def test_max_aggregation():
    scores = [0.1, 0.9, 0.3]
    assert aggregate_tool_score(scores, method="max") == pytest.approx(0.9)


def test_mean_aggregation():
    scores = [0.2, 0.4, 0.6]
    assert aggregate_tool_score(scores, method="mean") == pytest.approx(0.4)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        aggregate_tool_score([0.5], method="median")

import math
import pytest
from evaluation.metrics import compute_metrics, aggregate_fold_metrics


def test_perfect_classifier():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    m = compute_metrics(y_true, y_score)
    assert m["f1"] == pytest.approx(1.0)
    assert m["auc_roc"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)


def test_all_wrong():
    y_true = [0, 0, 1, 1]
    y_score = [0.9, 0.8, 0.2, 0.1]
    m = compute_metrics(y_true, y_score)
    assert m["f1"] == pytest.approx(0.0)
    assert m["auc_roc"] == pytest.approx(0.0)


def test_single_class_returns_nan_for_auc():
    y_true = [0, 0, 0]
    y_score = [0.1, 0.2, 0.3]
    m = compute_metrics(y_true, y_score)
    assert math.isnan(m["auc_roc"])
    assert math.isnan(m["auprc"])


def test_aggregate_fold_metrics():
    folds = [
        {"f1": 0.8, "auc_roc": 0.9, "auprc": 0.85, "accuracy": 0.8},
        {"f1": 0.9, "auc_roc": 0.95, "auprc": 0.90, "accuracy": 0.9},
    ]
    summary = aggregate_fold_metrics(folds)
    assert summary["f1_mean"] == pytest.approx(0.85)
    assert "f1_std" in summary
    assert "auc_roc_mean" in summary

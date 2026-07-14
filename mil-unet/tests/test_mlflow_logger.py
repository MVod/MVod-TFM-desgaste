import pytest
from pathlib import Path
from tracking.mlflow_logger import RunLogger


def test_logger_creates_experiment(tmp_path):
    logger = RunLogger(
        experiment_name="test_exp",
        tracking_uri=str(tmp_path / "mlruns"),
    )
    assert logger.experiment_name == "test_exp"


def test_log_fold_does_not_raise(tmp_path):
    logger = RunLogger("test_exp", tracking_uri=str(tmp_path / "mlruns"))
    logger.log_fold(
        fold_id=1,
        test_tool_id="000001",
        params={"branch": "A", "lr": 0.001, "epochs": 5},
        metrics={"f1": 0.85, "auc_roc": 0.90, "auprc": 0.88, "accuracy": 0.85},
    )


def test_log_summary_does_not_raise(tmp_path):
    logger = RunLogger("test_exp", tracking_uri=str(tmp_path / "mlruns"))
    fold_metrics = [
        {"f1": 0.8, "auc_roc": 0.9, "auprc": 0.85, "accuracy": 0.8},
        {"f1": 0.9, "auc_roc": 0.95, "auprc": 0.90, "accuracy": 0.9},
    ]
    logger.log_summary(fold_metrics=fold_metrics, params={"branch": "A"})


def test_log_fold_with_model_path(tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    logger = RunLogger("test_exp", tracking_uri=str(tmp_path / "mlruns"))
    logger.log_fold(
        fold_id=0,
        test_tool_id="000002",
        params={"branch": "B"},
        metrics={"f1": 1.0, "auc_roc": 1.0, "auprc": 1.0, "accuracy": 1.0},
        model_path=model_path,
    )

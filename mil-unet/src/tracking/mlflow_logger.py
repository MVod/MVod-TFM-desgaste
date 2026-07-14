from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import mlflow
from evaluation.metrics import aggregate_fold_metrics


class RunLogger:
    """Wraps MLflow to log per-fold and summary runs for a branch."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "./mlruns",
    ) -> None:
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_fold(
        self,
        fold_id: int,
        test_tool_id: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model_path: Optional[Path] = None,
        artifact_paths: Optional[List[Path]] = None,
    ) -> None:
        """Logs one LOO-CV fold as an MLflow run."""
        run_name = f"fold_{fold_id:02d}_tool_{test_tool_id}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("run_type", "fold")
            mlflow.set_tag("fold_id", str(fold_id))
            mlflow.set_tag("test_tool_id", test_tool_id)
            mlflow.log_params(params)
            mlflow.log_params({"fold_id": fold_id, "test_tool_id": test_tool_id})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not math.isnan(v)})
            if model_path is not None and Path(model_path).exists():
                mlflow.log_artifact(str(model_path), artifact_path="model")
            for path in (artifact_paths or []):
                if Path(path).exists():
                    mlflow.log_artifact(str(path), artifact_path="artifacts")

    def log_summary(
        self,
        fold_metrics: List[Dict[str, float]],
        params: Dict[str, Any],
        all_true: Optional[List[int]] = None,
        all_scores: Optional[List[float]] = None,
    ) -> None:
        """Logs aggregated LOO-CV summary (mean +/- std) as a separate MLflow run."""
        summary = aggregate_fold_metrics(fold_metrics, all_true=all_true, all_scores=all_scores)
        with mlflow.start_run(run_name="loocv_summary"):
            mlflow.log_params(params)
            mlflow.log_params({"loocv_folds": len(fold_metrics)})
            mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float)) and not math.isnan(v)})
            mlflow.set_tag("run_type", "summary")

import math
from typing import Dict, List
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def compute_metrics(
    y_true: List[int],
    y_score: List[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes F1, accuracy, AUC-ROC, AUPRC at tool level.
    Returns NaN for AUC metrics when only one class is present.
    """
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    if len(set(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    else:
        auc_roc = math.nan
        auprc = math.nan

    return {
        "f1": float(f1),
        "accuracy": float(accuracy),
        "auc_roc": auc_roc,
        "auprc": auprc,
        "threshold": threshold,
    }


def aggregate_fold_metrics(
    fold_metrics: List[Dict[str, float]],
    all_true: List[int] | None = None,
    all_scores: List[float] | None = None,
) -> Dict[str, float]:
    """
    Mean +/- std across LOO-CV folds. NaN values excluded.
    If all_true and all_scores are provided, computes global AUC-ROC and AUPRC
    across all folds (required when each fold has only 1 sample).
    """
    keys = [k for k in fold_metrics[0] if k != "threshold"]
    summary: Dict[str, float] = {}
    for k in keys:
        values = [m[k] for m in fold_metrics if not math.isnan(m.get(k, math.nan))]
        summary[f"{k}_mean"] = float(np.mean(values)) if values else math.nan
        summary[f"{k}_std"] = float(np.std(values)) if values else math.nan

    if all_true is not None and all_scores is not None and len(set(all_true)) > 1:
        summary["auc_roc_mean"] = float(roc_auc_score(all_true, all_scores))
        summary["auc_roc_std"] = math.nan
        summary["auprc_mean"] = float(average_precision_score(all_true, all_scores))
        summary["auprc_std"] = math.nan
        # F1 at optimal threshold (post-hoc, informative for TFM comparison)
        best_f1, best_t = 0.0, 0.5
        for t in sorted(set(all_scores)):
            y_pred = [1 if s >= t else 0 for s in all_scores]
            f1 = f1_score(all_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        summary["f1_optimal_mean"] = best_f1
        summary["f1_optimal_threshold"] = best_t

    return summary

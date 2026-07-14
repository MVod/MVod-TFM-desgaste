from typing import List


def aggregate_tool_score(image_scores: List[float], method: str = "max") -> float:
    """
    Aggregates per-image scores into a single tool-level score.
    method='max': worn tool needs only 1 anomalous image (Branch A, B).
    method='mean': deviation averaged over all images (Branch E, F).
    """
    if method == "max":
        return max(image_scores)
    if method == "mean":
        return sum(image_scores) / len(image_scores)
    raise ValueError(f"Unknown aggregation method: {method!r}. Use 'max' or 'mean'.")

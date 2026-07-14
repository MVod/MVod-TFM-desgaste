from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from branches.base import BaseBranch
from evaluation.aggregation import aggregate_tool_score
from preprocessing.m2_preprocessing import load_tool_roi_cached
from preprocessing.registration import register_mask
from preprocessing.scale_calibration import get_px_per_mm
from preprocessing.synthetic_mask import detect_thread_boundary, generate_ideal_mask


class ProfileDeviationScorer:
    """XOR deviation between actual binary profile and ideal ISO mask."""

    def compute(self, actual: np.ndarray, ideal: np.ndarray) -> float:
        """
        Returns |actual XOR ideal| / |ideal|.
        0.0 = perfect match, approaches 1.0 for complete mismatch.
        Returns 0.0 if ideal mask is empty (no reference area).
        """
        actual_bin = (actual > 127).astype(np.uint8)
        ideal_bin = (ideal > 127).astype(np.uint8)
        deviation = np.abs(actual_bin.astype(np.int16) - ideal_bin.astype(np.int16))
        ideal_area = int(ideal_bin.sum())
        if ideal_area == 0:
            return 0.0
        return float(deviation.sum()) / float(ideal_area)


class BranchC(BaseBranch):
    """
    Branch C — Direct ISO profile comparison.

    No neural network. Compares the binary Otsu tool mask directly against
    a registered synthetic ISO ideal mask. train() is a no-op.
    """

    def __init__(self) -> None:
        self._config: Dict = {}

    def train(
        self,
        train_tool_ids: List[str],
        tool_index: Dict,
        config: Dict,
    ) -> None:
        self._config = config

    def predict(
        self,
        tool_id: str,
        tool_index: Dict,
        config: Dict,
    ) -> Tuple[float, List[float]]:
        paths, _ = tool_index[tool_id]
        px_per_mm = get_px_per_mm(config)
        scorer = ProfileDeviationScorer()
        image_scores: List[float] = []

        for img_path in paths:
            tool_mask, tool_roi = load_tool_roi_cached(img_path)
            H, W = tool_mask.shape

            boundary = detect_thread_boundary(tool_roi)
            center_row = H - 1 - boundary
            ideal = np.flipud(
                generate_ideal_mask(
                    px_per_mm=px_per_mm,
                    image_shape=(H, W),
                    pitch_mm=config["pitch_mm"],
                    thread_height_mm=config["thread_height_mm"],
                    crest_width_mm=config["crest_width_mm"],
                    valley_width_mm=config["valley_width_mm"],
                    center_row=center_row,
                )
            )

            registered, _ = register_mask(tool_roi, ideal, tool_mask)
            score = scorer.compute(tool_mask, registered)
            image_scores.append(score)

            if config.get("vis_dir"):
                from pathlib import Path as _P
                from visualization import save_mask_c
                save_mask_c(
                    img_path, tool_roi, tool_mask, registered,
                    score, _P(config["vis_dir"]), tool_id,
                )

        tool_score = aggregate_tool_score(
            image_scores, config.get("aggregation", "mean")
        )
        return tool_score, image_scores

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._config, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self._config = pickle.load(f)

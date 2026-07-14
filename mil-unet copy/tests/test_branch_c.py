import sys
from pathlib import Path
import numpy as np
import pytest
from src.branches.branch_c_profile import BranchC, ProfileDeviationScorer


def test_branch_c_implements_base():
    from branches.base import BaseBranch
    assert issubclass(BranchC, BaseBranch)


def test_profile_deviation_scorer_perfect_match():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[30:70, :] = 255
    scorer = ProfileDeviationScorer()
    assert scorer.compute(mask, mask) == pytest.approx(0.0)


def test_profile_deviation_scorer_mismatch():
    predicted = np.zeros((100, 200), dtype=np.uint8)
    predicted[40:60, :] = 255
    ideal = np.zeros((100, 200), dtype=np.uint8)
    ideal[30:70, :] = 255
    scorer = ProfileDeviationScorer()
    ratio = scorer.compute(predicted, ideal)
    assert 0.0 < ratio <= 1.0


def test_branch_c_train_is_noop(tmp_data_dir):
    from src.evaluation.loocv import build_tool_index
    config = _minimal_config(tmp_data_dir)
    branch = BranchC()
    tool_index = build_tool_index(tmp_data_dir)
    branch.train(list(tool_index.keys()), tool_index, config)


def test_branch_c_predict_returns_valid_score(tmp_data_dir):
    from src.evaluation.loocv import build_tool_index
    config = _minimal_config(tmp_data_dir)
    branch = BranchC()
    tool_index = build_tool_index(tmp_data_dir)
    branch.train(list(tool_index.keys()), tool_index, config)
    tool_score, image_scores = branch.predict("000001_0", tool_index, config)
    assert isinstance(tool_score, float)
    assert tool_score >= 0.0
    assert len(image_scores) == len(tool_index["000001_0"][0])
    assert all(isinstance(s, float) for s in image_scores)


def test_branch_c_save_load_roundtrip(tmp_data_dir, tmp_path):
    from src.evaluation.loocv import build_tool_index
    config = _minimal_config(tmp_data_dir)
    branch = BranchC()
    tool_index = build_tool_index(tmp_data_dir)
    branch.train(list(tool_index.keys()), tool_index, config)
    path = tmp_path / "branch_c_final.pkl"
    branch.save(path)
    assert path.exists()
    branch2 = BranchC()
    branch2.load(path)
    tool_score, _ = branch2.predict("000001_0", tool_index, config)
    assert isinstance(tool_score, float)


def _minimal_config(tmp_data_dir):
    return {
        "data_dir": str(tmp_data_dir),
        "seed": 42,
        "aggregation": "mean",
        "scale_factor_path": None,
        "px_per_mm": 5.0,
        "pitch_mm": 1.5,
        "thread_height_mm": 0.76,
        "crest_width_mm": 0.19,
        "valley_width_mm": 0.19,
    }

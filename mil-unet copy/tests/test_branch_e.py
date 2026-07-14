import numpy as np
import torch
import pytest
import segmentation_models_pytorch as smp
from src.branches.branch_e_unet import BranchE, DeviationScorer


def test_unet_forward_shape():
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,   # no download in CI
        in_channels=1,
        classes=1,
        activation="sigmoid",
    )
    model.eval()
    x = torch.zeros(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_deviation_scorer():
    predicted = np.zeros((100, 200), dtype=np.uint8)
    predicted[40:60, :] = 255  # narrower than ideal

    ideal = np.zeros((100, 200), dtype=np.uint8)
    ideal[30:70, :] = 255  # wider ideal

    scorer = DeviationScorer()
    ratio = scorer.compute(predicted, ideal)
    assert 0.0 < ratio <= 1.0


def test_deviation_scorer_perfect_match():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[30:70, :] = 255
    scorer = DeviationScorer()
    ratio = scorer.compute(mask, mask)
    assert ratio == pytest.approx(0.0)


def test_branch_e_implements_base():
    from branches.base import BaseBranch
    assert issubclass(BranchE, BaseBranch)


def test_branch_e_predict_shape(tmp_data_dir, tmp_path):
    config = {
        "backbone": "efficientnet-b0",
        "image_size": [64, 64],
        "batch_size": 1,
        "lr": 0.001,
        "epochs": 1,
        "aggregation": "mean",
        "seed": 42,
        "scale_factor_path": None,   # use default scale for test
        "pitch_mm": 1.5,
        "thread_height_mm": 0.76,
        "crest_width_mm": 0.46,
        "valley_width_mm": 0.19,
        "px_per_mm": 5.0,   # inject scale directly for test
    }
    from src.evaluation.loocv import build_tool_index
    tool_index = build_tool_index(tmp_data_dir)
    train_ids = [t for t in tool_index if t != "000001_0"]
    branch = BranchE()
    branch.train(train_ids, tool_index, config)
    tool_score, image_scores = branch.predict("000001_0", tool_index, config)
    assert 0.0 <= tool_score <= 1.0
    assert len(image_scores) == len(tool_index["000001_0"][0])

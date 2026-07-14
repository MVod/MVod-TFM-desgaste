import torch
import pytest
from src.branches.branch_a_mil import MILClassifier, BranchA


def test_mil_classifier_forward_shape():
    model = MILClassifier(dropout=0.0)
    model.eval()
    x = torch.zeros(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


def test_mil_classifier_output_in_range():
    model = MILClassifier(dropout=0.0)
    model.eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_branch_a_implements_base():
    from branches.base import BaseBranch
    assert issubclass(BranchA, BaseBranch)


def test_branch_a_predict_returns_tuple(tmp_data_dir, tmp_path):
    config = {
        "image_size": [32, 32],   # tiny for fast CI
        "batch_size": 2,
        "lr_phase1": 0.001,
        "lr_phase2": 0.0001,
        "epochs_phase1": 1,
        "epochs_phase2": 1,
        "dropout": 0.0,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "augmentation": False,
        "mil_pooling": "max",
        "seed": 42,
    }
    from src.evaluation.loocv import build_tool_index
    tool_index = build_tool_index(tmp_data_dir)
    train_ids = [t for t in tool_index if t != "000001_0"]
    branch = BranchA()
    branch.train(train_ids, tool_index, config)
    tool_score, image_scores = branch.predict("000001_0", tool_index, config)
    assert 0.0 <= tool_score <= 1.0
    assert len(image_scores) == len(tool_index["000001_0"][0])


def test_branch_a_save_load(tmp_data_dir, tmp_path):
    config = {
        "image_size": [32, 32],
        "batch_size": 2,
        "lr_phase1": 0.001,
        "lr_phase2": 0.0001,
        "epochs_phase1": 1,
        "epochs_phase2": 1,
        "dropout": 0.0,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "augmentation": False,
        "mil_pooling": "max",
        "seed": 42,
    }
    from src.evaluation.loocv import build_tool_index
    tool_index = build_tool_index(tmp_data_dir)
    train_ids = [t for t in tool_index if t != "000001_0"]
    branch = BranchA()
    branch.train(train_ids, tool_index, config)
    # Get score before saving
    score_before, _ = branch.predict("000001_0", tool_index, config)
    save_path = tmp_path / "branch_a.pt"
    branch.save(save_path)
    assert save_path.exists()

    branch2 = BranchA()
    branch2.load(save_path)
    score, _ = branch2.predict("000001_0", tool_index, config)
    assert 0.0 <= score <= 1.0
    assert abs(score - score_before) < 1e-5  # deterministic round-trip


def test_branch_a_predict_vis_writes_file(tmp_data_dir, tmp_path):
    """When vis_dir is set, predict() writes a PNG to output/vis/A/."""
    config = {
        "image_size": [32, 32],
        "batch_size": 2,
        "lr_phase1": 0.001,
        "lr_phase2": 0.0001,
        "epochs_phase1": 1,
        "epochs_phase2": 1,
        "dropout": 0.0,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "augmentation": False,
        "mil_pooling": "max",
        "seed": 42,
        "vis_dir": str(tmp_path / "vis"),
        "px_per_mm": 348.0,
    }
    from src.evaluation.loocv import build_tool_index
    tool_index = build_tool_index(tmp_data_dir)
    train_ids = [t for t in tool_index if t != "000001_0"]
    branch = BranchA()
    branch.train(train_ids, tool_index, config)
    branch.predict("000001_0", tool_index, config)
    vis_dir = tmp_path / "vis" / "A"
    assert vis_dir.exists(), "output/vis/A/ directory not created"
    pngs = list(vis_dir.glob("*.png"))
    assert len(pngs) > 0, "No PNG files written to output/vis/A/"


def test_branch_a_predict_no_vis_does_not_create_dir(tmp_data_dir, tmp_path):
    """When vis_dir is NOT set, predict() creates no vis directory."""
    config = {
        "image_size": [32, 32],
        "batch_size": 2,
        "lr_phase1": 0.001,
        "lr_phase2": 0.0001,
        "epochs_phase1": 1,
        "epochs_phase2": 1,
        "dropout": 0.0,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "augmentation": False,
        "mil_pooling": "max",
        "seed": 42,
    }
    from src.evaluation.loocv import build_tool_index
    tool_index = build_tool_index(tmp_data_dir)
    train_ids = [t for t in tool_index if t != "000001_0"]
    branch = BranchA()
    branch.train(train_ids, tool_index, config)
    branch.predict("000001_0", tool_index, config)
    assert not (tmp_path / "vis" / "A").exists()

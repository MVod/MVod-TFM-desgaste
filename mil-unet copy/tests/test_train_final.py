# jordi/tests/test_train_final.py
import sys
import yaml
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def minimal_config(tmp_path, tmp_data_dir):
    cfg = {
        "data_dir": str(tmp_data_dir),
        "model_dir": str(tmp_path / "models"),
        "seed": 42,
        "backbone": "resnet18",
        "image_size": [224, 224],
        "coreset_ratio": 0.1,
        "k_neighbors": 1,
        "aggregation": "max",
        "scale_factor_path": None,
        "mlflow_uri": "./mlruns",
        "save_models": False,
    }
    return cfg


def test_train_branch_final_calls_train_and_save(tmp_path, minimal_config):
    mock_branch = MagicMock()

    with patch("train_final.load_branch", return_value=mock_branch), \
         patch("train_final.model_path", return_value=tmp_path / "model.pkl"):
        from train_final import train_branch_final
        train_branch_final("B", minimal_config)

    mock_branch.train.assert_called_once()
    mock_branch.save.assert_called_once()



def test_train_branch_final_a_skips_precompute(tmp_path, minimal_config):
    mock_branch = MagicMock()

    with patch("train_final.load_branch", return_value=mock_branch), \
         patch("train_final.model_path", return_value=tmp_path / "model.pkl"):
        from train_final import train_branch_final
        train_branch_final("A", minimal_config)

    mock_branch.precompute_features.assert_not_called()


def test_train_branch_final_creates_model_dir(tmp_path, minimal_config):
    out_path = tmp_path / "new_dir" / "model.pkl"
    mock_branch = MagicMock()

    with patch("train_final.load_branch", return_value=mock_branch), \
         patch("train_final.model_path", return_value=out_path):
        from train_final import train_branch_final
        train_branch_final("B", minimal_config)

    assert out_path.parent.exists()

import numpy as np
import cv2
import pytest
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path):
    """
    Minimal fake dataset: 4 tools (2 normal, 2 worn), 3 images each.
    Image name follows Imagen_XXXXXX_RYYY convention.
    Images are 100x200 grayscale with a dark band simulating the tool.
    """
    for label in ("normal", "worn"):
        (tmp_path / label).mkdir()

    configs = [
        ("000001", "normal"),
        ("000002", "normal"),
        ("000003", "worn"),
        ("000004", "worn"),
    ]
    for tool_id, label in configs:
        for part in ("RM01", "RM02", "RM03"):
            img = np.ones((100, 200), dtype=np.uint8) * 240  # bright background
            img[30:70, :] = 60  # dark tool band
            path = tmp_path / label / f"Imagen_{tool_id}_{part}.png"
            cv2.imwrite(str(path), img)
    return tmp_path


@pytest.fixture
def gray_tool_image():
    """100x200 grayscale image: white background, dark tool band in center."""
    img = np.ones((100, 200), dtype=np.uint8) * 240
    img[30:70, :] = 60
    return img


@pytest.fixture
def tool_roi_mask():
    """Binary mask matching gray_tool_image: 255 where tool, 0 elsewhere."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[30:70, :] = 255
    return mask

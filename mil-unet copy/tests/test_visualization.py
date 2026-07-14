import numpy as np
import pytest
from pathlib import Path


def make_roi(h=80, w=120) -> np.ndarray:
    roi = np.ones((h, w), dtype=np.uint8) * 60
    roi[20:60, :] = 30
    return roi


def make_mask(h=80, w=120, filled=True) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if filled:
        m[20:60, 40:80] = 255
    return m



def test_save_4panel_writes_file(tmp_path):
    from visualization import _save_4panel
    roi = make_roi()
    pred = make_mask()
    ideal = make_mask()
    _save_4panel(roi, pred, ideal, 0.75, tmp_path, "A", "000001_0", "img_stem")
    out = tmp_path / "A" / "000001_0_img_stem.png"
    assert out.exists()
    import cv2
    img = cv2.imread(str(out))
    assert img is not None
    assert img.shape[1] == roi.shape[1] * 4  # 4 panels side by side



def test_save_gradcam_a_writes_file(tmp_path):
    from visualization import save_gradcam_a
    roi = make_roi()
    gradcam_map = np.random.rand(14, 14).astype(np.float32)
    ideal = make_mask()
    fake_path = Path("Imagen_000001_RM01.png")
    save_gradcam_a(fake_path, roi, gradcam_map, ideal, 0.8, tmp_path, "000001_0")
    out = tmp_path / "A" / "000001_0_Imagen_000001_RM01.png"
    assert out.exists()


def test_save_mask_c_writes_file(tmp_path):
    from visualization import save_mask_c
    roi = make_roi()
    tool_mask = make_mask()
    registered_ideal = make_mask()
    fake_path = Path("Imagen_000001_RM01.png")
    save_mask_c(fake_path, roi, tool_mask, registered_ideal, 0.3, tmp_path, "000001_0")
    out = tmp_path / "C" / "000001_0_Imagen_000001_RM01.png"
    assert out.exists()
    import cv2
    img = cv2.imread(str(out))
    assert img is not None
    assert img.shape[1] == roi.shape[1] * 4

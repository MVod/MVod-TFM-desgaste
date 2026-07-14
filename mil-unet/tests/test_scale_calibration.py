import json
import numpy as np
import cv2
import pytest
from pathlib import Path
from preprocessing.scale_calibration import calibrate_scale, save_scale_factor, load_scale_factor


@pytest.fixture
def synthetic_threaded_image(tmp_path):
    """
    200x400 image with 3 dark vertical bands simulating valleys at x=80,160,240.
    Valley distance = 80 px. With valley_distance_mm=1.0 → expected scale = 80 px/mm.
    """
    img = np.ones((200, 400), dtype=np.uint8) * 240  # bright background
    img[60:140, :] = 120  # dark tool band
    for x in (80, 160, 240):
        img[60:140, x-3:x+3] = 20  # very dark valleys
    path = tmp_path / "ref_image.png"
    cv2.imwrite(str(path), img)
    return path


def test_calibrate_scale_returns_float(synthetic_threaded_image):
    px_per_mm = calibrate_scale(synthetic_threaded_image, valley_distance_mm=1.0,
                                 min_peak_distance_px=50)
    assert isinstance(px_per_mm, float)
    assert px_per_mm > 0


def test_calibrate_scale_approximate_value(synthetic_threaded_image):
    px_per_mm = calibrate_scale(synthetic_threaded_image, valley_distance_mm=1.0,
                                 min_peak_distance_px=50)
    assert abs(px_per_mm - 80.0) < 5.0


def test_save_and_load_scale_factor(tmp_path):
    path = tmp_path / "scale_factor.json"
    save_scale_factor(348.5, path)
    loaded = load_scale_factor(path)
    assert abs(loaded - 348.5) < 1e-6


def test_load_scale_factor_reads_json(tmp_path):
    path = tmp_path / "scale_factor.json"
    path.write_text(json.dumps({"px_per_mm": 350.0}))
    assert load_scale_factor(path) == pytest.approx(350.0)


def test_get_px_per_mm_from_config_key():
    from preprocessing.scale_calibration import get_px_per_mm
    assert get_px_per_mm({"px_per_mm": 500.0}) == 500.0


def test_get_px_per_mm_from_file(tmp_path):
    from preprocessing.scale_calibration import get_px_per_mm, save_scale_factor
    p = tmp_path / "scale.json"
    save_scale_factor(686.0, p)
    assert get_px_per_mm({"scale_factor_path": str(p)}) == 686.0


def test_get_px_per_mm_default_fallback():
    from preprocessing.scale_calibration import get_px_per_mm
    assert get_px_per_mm({}) == 348.0

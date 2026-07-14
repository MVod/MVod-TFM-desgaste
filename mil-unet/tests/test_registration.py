import numpy as np
import cv2
import pytest
from preprocessing.registration import register_mask


@pytest.fixture
def aligned_pair():
    H, W = 100, 200
    gray = np.ones((H, W), dtype=np.uint8) * 240
    gray[30:70, :] = 60
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    roi_mask[30:70, :] = 255
    ideal_mask = np.zeros((H, W), dtype=np.uint8)
    ideal_mask[30:70, :] = 255
    return gray, roi_mask, ideal_mask


def test_register_mask_returns_ndarray(aligned_pair):
    gray, roi_mask, ideal_mask = aligned_pair
    registered, confidence = register_mask(gray, ideal_mask, roi_mask)
    assert isinstance(registered, np.ndarray)
    assert registered.shape == ideal_mask.shape
    assert registered.dtype == np.uint8


def test_register_mask_returns_confidence_in_range(aligned_pair):
    gray, roi_mask, ideal_mask = aligned_pair
    _, confidence = register_mask(gray, ideal_mask, roi_mask)
    assert 0.0 <= confidence <= 1.0


def test_register_mask_output_is_binary(aligned_pair):
    gray, roi_mask, ideal_mask = aligned_pair
    registered, _ = register_mask(gray, ideal_mask, roi_mask)
    assert set(np.unique(registered)).issubset({0, 255})


def test_register_mask_fallback_on_bad_image():
    """Uniform image -> low confidence -> fallback returns input mask."""
    H, W = 100, 200
    gray = np.ones((H, W), dtype=np.uint8) * 128
    roi_mask = np.ones((H, W), dtype=np.uint8) * 255
    ideal_mask = np.zeros((H, W), dtype=np.uint8)
    ideal_mask[40:60, :] = 255
    registered, confidence = register_mask(gray, ideal_mask, roi_mask,
                                            min_confidence=0.9)
    assert np.array_equal(registered, ideal_mask)

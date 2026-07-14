"""
Tests for M2 — Digital Illumination Correction
Run with: uv run pytest tests/test_m2.py -v
"""

import numpy as np
import pytest
from pathlib import Path

from preprocessing.m2_preprocessing import (
    separate_tool_background,
    homomorphic_filter,
    inpaint_highlights,
    compute_snr,
    run_m2,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_image():
    """Simulate a backlit tool image: dark tool on bright background."""
    img = np.full((256, 256), 240, dtype=np.uint8)   # bright background
    img[64:192, 64:192] = 80                          # dark tool region
    img[100:130, 100:130] = 255                       # simulated highlight
    return img


@pytest.fixture
def uniform_tool():
    """Uniform dark tool, no background."""
    return np.full((128, 128), 80, dtype=np.uint8)


# ── M2-0 tests ────────────────────────────────────────────────────────────────

def test_separate_tool_background_returns_correct_shapes(synthetic_image):
    mask, roi = separate_tool_background(synthetic_image)
    assert mask.shape == synthetic_image.shape
    assert roi.shape == synthetic_image.shape


def test_tool_mask_is_binary(synthetic_image):
    mask, _ = separate_tool_background(synthetic_image)
    unique = np.unique(mask)
    assert set(unique).issubset({0, 255})


def test_tool_roi_background_is_zeroed(synthetic_image):
    mask, roi = separate_tool_background(synthetic_image)
    assert roi[mask == 0].max() == 0


def test_tool_mask_detects_dark_region(synthetic_image):
    mask, _ = separate_tool_background(synthetic_image)
    # Center of tool region should be detected
    assert mask[128, 128] == 255


# ── M2-1 tests ────────────────────────────────────────────────────────────────

def test_homomorphic_filter_output_shape(uniform_tool):
    result = homomorphic_filter(uniform_tool)
    assert result.shape == uniform_tool.shape


def test_homomorphic_filter_output_dtype(uniform_tool):
    result = homomorphic_filter(uniform_tool)
    assert result.dtype == np.uint8


def test_homomorphic_filter_output_range(uniform_tool):
    result = homomorphic_filter(uniform_tool)
    assert result.min() >= 0
    assert result.max() <= 255


# ── M2-2 tests ────────────────────────────────────────────────────────────────

def test_inpaint_highlights_reduces_bright_pixels(synthetic_image):
    # Extract tool region
    _, tool_roi = separate_tool_background(synthetic_image)
    inpainted, mask = inpaint_highlights(tool_roi, threshold=0.95)
    bright_before = (tool_roi > 240).sum()
    bright_after = (inpainted > 240).sum()
    assert bright_after <= bright_before


def test_inpaint_mask_is_binary(synthetic_image):
    _, tool_roi = separate_tool_background(synthetic_image)
    _, mask = inpaint_highlights(tool_roi)
    unique = np.unique(mask)
    assert set(unique).issubset({0, 255})


# ── SNR tests ─────────────────────────────────────────────────────────────────

def test_compute_snr_returns_float(uniform_tool):
    snr = compute_snr(uniform_tool)
    assert isinstance(snr, float)


def test_compute_snr_with_mask(synthetic_image):
    mask, _ = separate_tool_background(synthetic_image)
    snr = compute_snr(synthetic_image, mask)
    assert snr > 0


# ── Integration test ──────────────────────────────────────────────────────────

def test_run_m2_returns_all_keys(tmp_path, synthetic_image):
    import cv2
    img_path = tmp_path / "test_image.png"
    cv2.imwrite(str(img_path), synthetic_image)

    result = run_m2(img_path)
    expected_keys = {"original", "tool_mask", "tool_roi", "corrected", "highlight_mask", "final"}
    assert expected_keys == set(result.keys())


def test_run_m2_output_shapes_match(tmp_path, synthetic_image):
    import cv2
    img_path = tmp_path / "test_image.png"
    cv2.imwrite(str(img_path), synthetic_image)

    result = run_m2(img_path)
    h, w = synthetic_image.shape
    for key in ("original", "tool_mask", "tool_roi", "corrected", "final"):
        assert result[key].shape == (h, w), f"Shape mismatch for key: {key}"

import numpy as np
import pytest
from preprocessing.synthetic_mask import generate_ideal_mask


def test_output_shape():
    mask = generate_ideal_mask(px_per_mm=10.0, image_shape=(100, 200))
    assert mask.shape == (100, 200)
    assert mask.dtype == np.uint8


def test_output_is_binary():
    mask = generate_ideal_mask(px_per_mm=10.0, image_shape=(100, 200))
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255})


def test_mask_has_material_and_background():
    mask = generate_ideal_mask(px_per_mm=10.0, image_shape=(100, 200))
    assert (mask == 255).any(), "No material pixels found"
    assert (mask == 0).any(), "No background pixels found"


def test_crest_rows_are_255_at_center():
    """At a crest column, rows from center upward should be 255."""
    px_per_mm = 20.0
    H, W = 200, 400
    mask = generate_ideal_mask(
        px_per_mm=px_per_mm,
        image_shape=(H, W),
        pitch_mm=1.5,
        thread_height_mm=0.5,
        crest_width_mm=0.4,
        valley_width_mm=0.2,
        center_row=100,
    )
    assert mask[100, W // 2] == 255


def test_body_is_always_filled():
    """Rows below center_row + thread_height should always be 255 (tool body)."""
    px_per_mm = 10.0
    H, W = 100, 200
    center_row = 50
    thread_height_px = int(0.76 * px_per_mm)
    mask = generate_ideal_mask(
        px_per_mm=px_per_mm,
        image_shape=(H, W),
        center_row=center_row,
    )
    body_row = center_row + thread_height_px + 2
    if body_row < H:
        assert mask[body_row, W // 2] == 255, "Body below valleys must be filled"

from typing import Optional, Tuple
import numpy as np
import cv2


def detect_thread_boundary(tool_roi: np.ndarray, search_start_pct: float = 0.2) -> int:
    """
    Find the row where the solid tap body transitions into the thread profile zone.

    In tool_roi, background pixels are zeroed. Tool body rows have high mean
    (all dark tool pixels). Thread zone rows have progressively lower mean because
    background gaps between teeth are zeroed. The boundary is where row-mean
    drops below the midpoint between the body level and the zeroed background level.

    Args:
        tool_roi: Grayscale image (H, W) uint8, background zeroed by roi_mask.
        search_start_pct: Skip this top fraction (avoids image edge artefacts).

    Returns:
        Row index of the boundary.  Pass to generate_ideal_mask as:
            center_row = H - 1 - boundary_row
        then np.flipud the result.
    """
    H = tool_roi.shape[0]
    row_mean = tool_roi.mean(axis=1).astype(np.float32)

    top_q = H // 4
    body_level = float(row_mean[:top_q].mean())       # solid body: mean ~ 87-94
    bg_level   = float(row_mean[3 * H // 4:].mean())  # zeroed bkg: mean ~ 8-10

    # Need meaningful contrast to detect the transition
    if body_level - bg_level < 10:
        return H // 2

    # Threshold at 25% drop from body level — detects the VALLEY start,
    # where tool pixels first begin to be replaced by background gaps.
    # Using 50% (mid_level) would land at mid-transition, too deep.
    threshold = body_level - 0.25 * (body_level - bg_level)

    s0 = int(H * search_start_pct)
    s1 = 3 * H // 4
    below = np.where(row_mean[s0:s1] < threshold)[0]
    if len(below):
        return s0 + int(below[0])
    return H // 2


def generate_ideal_mask(
    px_per_mm: float,
    image_shape: Tuple[int, int],
    pitch_mm: float = 1.5,
    thread_height_mm: float = 0.76,
    crest_width_mm: float = 0.19,
    valley_width_mm: float = 0.19,
    n_threads: int = 5,
    center_row: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a binary mask of the ideal ISO metric thread profile.

    Thread profile is a horizontal band of trapezoid teeth.
    Each tooth: valley -> left flank (60 deg) -> flat crest -> right flank -> valley.
    Tool body below the valleys is always filled (255).

    Args:
        px_per_mm: Pixels per millimetre (from scale_calibration).
        image_shape: (H, W) output mask size.
        pitch_mm: Thread pitch [mm].
        thread_height_mm: Height from valley to crest [mm].
        crest_width_mm: Flat crest width [mm].
        valley_width_mm: Flat valley width [mm].
        n_threads: Number of thread teeth to render.
        center_row: Row where valleys sit. Defaults to H // 2.

    Returns:
        uint8 mask: 255 = thread material, 0 = background.
    """
    H, W = image_shape
    if center_row is None:
        center_row = H // 2

    pitch_px = pitch_mm * px_per_mm
    height_px = thread_height_mm * px_per_mm
    crest_px = crest_width_mm * px_per_mm
    valley_px = valley_width_mm * px_per_mm
    # Horizontal run of a 60-degree flank: h / tan(60 deg)
    flank_px = height_px / np.tan(np.radians(60.0))

    mask = np.zeros((H, W), dtype=np.uint8)
    total_width_px = n_threads * pitch_px
    start_x = W / 2.0 - total_width_px / 2.0

    for col in range(W):
        x_rel = (col - start_x) % pitch_px

        half_valley = valley_px / 2.0
        if x_rel < half_valley or x_rel >= pitch_px - half_valley:
            tooth_h = 0.0
        elif x_rel < half_valley + flank_px:
            tooth_h = ((x_rel - half_valley) / flank_px) * height_px
        elif x_rel < half_valley + flank_px + crest_px:
            tooth_h = height_px
        elif x_rel < half_valley + 2 * flank_px + crest_px:
            tooth_h = (1.0 - (x_rel - half_valley - flank_px - crest_px) / flank_px) * height_px
        else:
            tooth_h = 0.0

        top_row = int(center_row - tooth_h)
        top_row = max(0, min(top_row, H))
        mask[top_row:H, col] = 255

    return mask

from typing import Tuple
import numpy as np
import cv2
from scipy.signal import fftconvolve


def register_mask(
    image: np.ndarray,
    ideal_mask: np.ndarray,
    roi_mask: np.ndarray,
    min_confidence: float = 0.3,
) -> Tuple[np.ndarray, float]:
    """
    Aligns the ideal thread profile mask to the actual thread position in the image.

    Step 1 — Vertical shift: cross-correlates row-mean Sobel Y profiles.
    Step 2 — Horizontal phase: cross-correlates column-mean profiles in the thread
              zone.  Pitch is detected via autocorrelation of the full-width mask
              (robust against flat crest plateaus where find_peaks fails).

    Args:
        image: Grayscale image (H, W) uint8.
        ideal_mask: Binary ideal profile mask (H, W) uint8 {0, 255}.
        roi_mask: Otsu tool ROI mask (H, W) uint8 {0, 255}.
        min_confidence: Minimum normalised correlation peak to trust vertical result.

    Returns:
        (registered_mask, confidence): registered_mask uint8, confidence in [0, 1].
    """
    H, W = image.shape

    # ── Step 1: vertical registration ────────────────────────────────────────
    roi_float = image.astype(np.float32)
    roi_float[roi_mask == 0] = 0.0
    grad_y = np.abs(cv2.Sobel(roi_float, cv2.CV_32F, 0, 1, ksize=3))
    image_profile = grad_y.mean(axis=1)  # (H,)

    mask_float = ideal_mask.astype(np.float32) / 255.0
    mask_profile = mask_float.mean(axis=1)  # (H,)

    corr = fftconvolve(image_profile, mask_profile[::-1], mode="full")
    peak_idx = int(np.argmax(corr))
    offset_y = peak_idx - (H - 1)

    corr_max = float(corr.max())
    corr_mean = float(np.abs(corr).mean())
    if corr_mean > 0:
        confidence = min(max((corr_max - corr_mean) / (corr_max + 1e-8), 0.0), 1.0)
    else:
        confidence = 0.0

    if confidence < min_confidence:
        registered = ideal_mask.copy()
    else:
        M = np.float32([[1, 0, 0], [0, 1, offset_y]])
        registered = cv2.warpAffine(
            ideal_mask, M, (W, H),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        ).astype(np.uint8)

    # ── Step 2: horizontal phase correction ──────────────────────────────────
    registered = _align_horizontal(image, registered)

    return registered, confidence


def _align_horizontal(
    image: np.ndarray,
    ideal_mask: np.ndarray,
    col_start_pct: float = 0.05,
    col_end_pct: float = 0.80,
) -> np.ndarray:
    """
    Correct horizontal phase of a periodic thread mask by cross-correlating
    column-mean profiles in the thread zone.

    The thread zone is identified as rows where the ideal mask has partial coverage
    (neither all-white nor all-black).  Cross-correlation is restricted to one pitch
    period (auto-detected from mask peak spacing) so np.roll wraps correctly.

    col_start_pct / col_end_pct: restrict analysis to this column range to exclude
    tap-head or shank artefacts that appear on the left/right edges of some images.
    """
    H, W = ideal_mask.shape

    # Identify thread zone (rows with partial column coverage)
    row_coverage = ideal_mask.mean(axis=1)  # 0-255
    partial = np.where((row_coverage > 10) & (row_coverage < 245))[0]
    if len(partial) < 10:
        return ideal_mask  # no identifiable thread zone

    zone_top = int(partial[0])
    zone_bot = int(partial[-1]) + 1

    # Detect pitch from FULL-WIDTH mask via autocorrelation.
    # find_peaks fails on the flat crest plateaus (all columns in a crest have
    # identical height → no strict local maximum).  Autocorrelation of a periodic
    # signal has a peak at lag = period, regardless of plateau width.
    mask_full = (ideal_mask[zone_top:zone_bot, :].astype(np.float32) / 255.0).mean(axis=0)
    mask_full_n = (mask_full - mask_full.mean()) / (mask_full.std() + 1e-8)
    auto = fftconvolve(mask_full_n, mask_full_n[::-1], mode="full")
    auto_center = len(mask_full_n) - 1
    min_pitch, max_pitch = W // 8, W // 2
    auto_search = auto[auto_center + min_pitch: auto_center + max_pitch]
    if len(auto_search) == 0:
        return ideal_mask
    pitch_px = min_pitch + int(np.argmax(auto_search))

    # Restrict to central columns for cross-correlation (exclude tap-head / shank on edges)
    ca = int(W * col_start_pct)
    cb = int(W * col_end_pct)

    # Column mean profile in thread zone (central columns only)
    img_col  = image[zone_top:zone_bot, ca:cb].astype(np.float32).mean(axis=0)
    mask_col = (ideal_mask[zone_top:zone_bot, ca:cb].astype(np.float32) / 255.0).mean(axis=0)

    # Normalize and cross-correlate
    img_col_n  = (img_col  - img_col.mean())  / (img_col.std()  + 1e-8)
    mask_col_n = (mask_col - mask_col.mean()) / (mask_col.std() + 1e-8)
    corr = fftconvolve(img_col_n, mask_col_n[::-1], mode="full")
    center = len(mask_col_n) - 1
    half = pitch_px // 2
    lo, hi = max(0, center - half), min(len(corr), center + half)
    search = corr[lo:hi]
    if search.max() <= 0:
        return ideal_mask

    shift_x = int(np.argmax(search)) - half
    if shift_x == 0:
        return ideal_mask

    # Translate (no wrap) — np.roll would create a spurious extra tooth at the
    # wrap boundary when W is not a multiple of pitch_px.
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    result = cv2.warpAffine(
        ideal_mask, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    ).astype(np.uint8)

    # Tile-fill any edge black strip left by warpAffine clipping.
    # The thread pattern is periodic, so col x can borrow from (x ± pitch_px).
    col_max = result.max(axis=0)  # 0 = column is completely black

    # Right edge (shift_x < 0 leaves a black strip on the right)
    right_ok = W
    while right_ok > 0 and col_max[right_ok - 1] == 0:
        right_ok -= 1
    gap_r = W - right_ok
    if 0 < gap_r < pitch_px and right_ok - pitch_px >= 0:
        result[:, right_ok:] = result[:, right_ok - pitch_px:right_ok - pitch_px + gap_r]

    # Left edge (shift_x > 0 leaves a black strip on the left)
    left_ok = 0
    while left_ok < W and col_max[left_ok] == 0:
        left_ok += 1
    gap_l = left_ok
    if 0 < gap_l < pitch_px and left_ok + pitch_px <= W:
        result[:, :gap_l] = result[:, left_ok + pitch_px - gap_l:left_ok + pitch_px]

    return result

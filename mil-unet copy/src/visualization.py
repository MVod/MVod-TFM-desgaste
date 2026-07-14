"""
visualization.py — Save prediction overlays for Streamlit inspection.

Called from branch predict() when config["vis_dir"] is set.
Output format: [ROI | predicted_mask | ideal_mask | diff] for all branches.
Branch A uses raw GradCAM++ heatmap (jet) in panel 2 instead of a binary mask.
Filename: {vis_dir}/{branch}/{tool_id}_{img_stem}.png
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def _diff_two_color(pred_mask: np.ndarray, ideal_mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """Two-color diff BGR image.
    Red  (0,0,255): pred=material, ideal=empty  → wear filling the valley.
    Blue (255,0,0): pred=empty,    ideal=material → wear eating the tooth.
    Black:          agreement.
    """
    pred  = (cv2.resize(pred_mask,  (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(bool)
    ideal = (cv2.resize(ideal_mask, (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(bool)
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[pred & ~ideal] = (0, 0, 255)    # red
    out[~pred & ideal] = (255, 0, 0)    # blue
    return out


def _save_4panel(
    tool_roi: np.ndarray,
    pred_mask: np.ndarray,
    ideal_mask: np.ndarray,
    score: float,
    vis_dir: Path,
    branch: str,
    tool_id: str,
    img_stem: str,
) -> None:
    """Write [ROI | pred_mask | ideal_mask | diff] as a single PNG."""
    H, W = tool_roi.shape[:2]

    def to_bgr(m: np.ndarray) -> np.ndarray:
        m_r = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m_bin = (m_r > 127).astype(np.uint8) * 255
        return cv2.cvtColor(m_bin, cv2.COLOR_GRAY2BGR)

    roi_bgr = cv2.cvtColor(tool_roi, cv2.COLOR_GRAY2BGR)
    pred_bgr = to_bgr(pred_mask)
    ideal_bgr = to_bgr(ideal_mask)
    diff_color = _diff_two_color(pred_mask, ideal_mask, H, W)

    panel = np.hstack([roi_bgr, pred_bgr, ideal_bgr, diff_color])
    cv2.putText(panel, f"score={score:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out = _ensure(vis_dir / branch) / f"{tool_id}_{img_stem}.png"
    cv2.imwrite(str(out), panel)


def save_gradcam_a(
    img_path: Path,
    tool_roi: np.ndarray,
    gradcam_map: np.ndarray,
    ideal_mask: np.ndarray,
    score: float,
    vis_dir: Path,
    tool_id: str,
) -> None:
    """Branch A: GradCAM++ raw heatmap (jet colormap) → 4-panel."""
    H, W = tool_roi.shape[:2]
    resized = cv2.resize(gradcam_map.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    roi_bgr = cv2.cvtColor(tool_roi, cv2.COLOR_GRAY2BGR)

    def to_bgr(m: np.ndarray) -> np.ndarray:
        m_r = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m_bin = (m_r > 127).astype(np.uint8) * 255
        return cv2.cvtColor(m_bin, cv2.COLOR_GRAY2BGR)

    ideal_bgr = to_bgr(ideal_mask)
    overlay = cv2.addWeighted(ideal_bgr, 0.5, heatmap_bgr, 0.5, 0)

    panel = np.hstack([roi_bgr, heatmap_bgr, ideal_bgr, overlay])
    cv2.putText(panel, f"score={score:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out = _ensure(vis_dir / "A") / f"{tool_id}_{img_path.stem}.png"
    cv2.imwrite(str(out), panel)


def save_mask_b(
    img_path: Path,
    tool_roi: np.ndarray,
    pred_mask: np.ndarray,
    ideal_mask: np.ndarray,
    deviation: float,
    vis_dir: Path,
    tool_id: str,
) -> None:
    """Branch B: U-Net segmentation mask → 4-panel."""
    _save_4panel(tool_roi, pred_mask, ideal_mask, deviation, vis_dir, "B", tool_id, img_path.stem)


def save_mask_c(
    img_path: Path,
    tool_roi: np.ndarray,
    tool_mask: np.ndarray,
    registered_ideal: np.ndarray,
    deviation: float,
    vis_dir: Path,
    tool_id: str,
) -> None:
    """Branch C: binary Otsu profile vs registered ISO ideal → 4-panel."""
    _save_4panel(tool_roi, tool_mask, registered_ideal, deviation, vis_dir, "C", tool_id, img_path.stem)

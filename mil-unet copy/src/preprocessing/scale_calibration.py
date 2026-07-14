import json
from pathlib import Path
from typing import Dict
import numpy as np
import cv2
from scipy.signal import find_peaks
from preprocessing.m2_preprocessing import separate_tool_background


def calibrate_scale(
    image_path: Path,
    valley_distance_mm: float = 1.25,
    min_peak_distance_px: int = 400,
) -> float:
    """
    Derives px/mm by measuring valley-to-valley distance in a reference normal image.
    Valleys appear as periodic dips in the horizontal projection of the tool ROI.
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    tool_mask, tool_roi = separate_tool_background(gray)
    proj = tool_roi.astype(np.float32).mean(axis=0)

    inv_proj = float(proj.max()) - proj
    peaks, _ = find_peaks(
        inv_proj,
        distance=min_peak_distance_px,
        height=inv_proj.mean(),
    )

    if len(peaks) < 2:
        raise ValueError(
            f"Detected only {len(peaks)} valley(s) in {image_path.name}. "
            "Need >=2. Try reducing min_peak_distance_px."
        )

    mean_valley_distance_px = float(np.diff(peaks).mean())
    return mean_valley_distance_px / valley_distance_mm


def save_scale_factor(px_per_mm: float, path: Path) -> None:
    """Saves px/mm to a JSON file for reuse across runs."""
    Path(path).write_text(json.dumps({"px_per_mm": px_per_mm}, indent=2))


def load_scale_factor(path: Path) -> float:
    """Loads px/mm from JSON produced by save_scale_factor."""
    return float(json.loads(Path(path).read_text())["px_per_mm"])


def get_px_per_mm(config: Dict) -> float:
    """Resolve px/mm from config. Checks 'px_per_mm' key, then 'scale_factor_path', then defaults to 348.0."""
    if "px_per_mm" in config:
        return float(config["px_per_mm"])
    scale_path = config.get("scale_factor_path")
    if scale_path and Path(scale_path).exists():
        return load_scale_factor(Path(scale_path))
    return 348.0

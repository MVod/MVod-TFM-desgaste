"""
M2 — Digital Illumination Correction
Steps:
  M2-0: Tool / background separation (Otsu + morphology)
  M2-1: Homomorphic filtering on tool ROI
  M2-2: (Optional) Highlight inpainting on tool ROI
"""

import cv2
import numpy as np
from pathlib import Path


# ── ROI cache (avoids repeated disk I/O + Otsu across LOO-CV folds) ───────────

_roi_cache: dict = {}


def load_tool_roi_cached(img_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load image and run M2-0, caching result by path."""
    key = str(img_path)
    if key not in _roi_cache:
        gray = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        _roi_cache[key] = separate_tool_background(gray)
    return _roi_cache[key]


# ── M2-0: Tool / Background Separation ────────────────────────────────────────

def separate_tool_background(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate dark tool from bright background using Otsu thresholding.

    Args:
        gray: Grayscale image (H, W) uint8

    Returns:
        tool_mask: Binary mask — 255 where tool, 0 where background
        tool_roi:  Original image with background zeroed out
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing to fill internal gaps in the tool region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tool_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    tool_roi = cv2.bitwise_and(gray, gray, mask=tool_mask)
    return tool_mask, tool_roi


# ── M2-1: Homomorphic Filtering ───────────────────────────────────────────────

def homomorphic_filter(gray_roi: np.ndarray, sigma: float = 30.0,
                       gamma_low: float = 0.5, gamma_high: float = 1.5) -> np.ndarray:
    """
    Attenuate low-frequency illumination gradient while enhancing reflectance.

    Args:
        gray_roi:   Tool ROI grayscale image (background should be 0 or masked)
        sigma:      Gaussian sigma in frequency domain (controls cutoff)
        gamma_low:  Gain for low frequencies (illumination) — < 1 to suppress
        gamma_high: Gain for high frequencies (reflectance) — > 1 to enhance

    Returns:
        Corrected image as uint8, same shape as input
    """
    rows, cols = gray_roi.shape

    # Log transform (avoid log(0))
    img_log = np.log1p(gray_roi.astype(np.float64))

    # FFT
    img_fft = np.fft.fft2(img_log)
    img_fft_shift = np.fft.fftshift(img_fft)

    # Gaussian high-pass filter in frequency domain
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    d_sq = (y - crow) ** 2 + (x - ccol) ** 2
    H = (gamma_high - gamma_low) * (1 - np.exp(-d_sq / (2 * sigma ** 2))) + gamma_low

    # Apply filter and inverse FFT
    filtered = np.fft.ifft2(np.fft.ifftshift(img_fft_shift * H)).real

    # Exponential inverse and normalise to uint8
    result = np.expm1(filtered)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


# ── M2-2: Highlight Inpainting (optional) ─────────────────────────────────────

def inpaint_highlights(image: np.ndarray, threshold: float = 0.95,
                       radius: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect and inpaint specular highlights on the tool surface.
    Should be applied AFTER homomorphic filtering.
    Call only if highlights are confirmed present on tool ROI.

    Args:
        image:     Grayscale tool ROI uint8
        threshold: Normalised intensity threshold (0–1) for highlight detection
        radius:    Inpainting radius

    Returns:
        inpainted: Corrected image
        mask:      Highlight mask (255 = highlight detected)
    """
    thresh_val = int(threshold * 255)
    _, mask = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
    return inpainted, mask


# ── Full M2 pipeline ───────────────────────────────────────────────────────────

def run_m2(image_path: str | Path,
           homomorphic_sigma: float = 30.0,
           apply_inpainting: bool = False,
           highlight_threshold: float = 0.95) -> dict:
    """
    Run the full M2 correction pipeline on a single image.

    Args:
        image_path:           Path to input image
        homomorphic_sigma:    Sigma for homomorphic filter (default 30.0)
        apply_inpainting:     Whether to apply highlight inpainting (default False)
        highlight_threshold:  Threshold for highlight detection (0–1)

    Returns:
        dict with keys:
            'original'      : original grayscale image
            'tool_mask'     : binary tool mask
            'tool_roi'      : masked tool ROI
            'corrected'     : homomorphic-filtered tool ROI
            'highlight_mask': highlight mask (or None if not applied)
            'final'         : final corrected image
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # M2-0
    tool_mask, tool_roi = separate_tool_background(gray)

    # M2-1
    corrected = homomorphic_filter(tool_roi, sigma=homomorphic_sigma)

    # Re-apply mask after filtering (background may have been altered)
    corrected = cv2.bitwise_and(corrected, corrected, mask=tool_mask)

    # M2-2 (optional)
    highlight_mask = None
    final = corrected
    if apply_inpainting:
        final, highlight_mask = inpaint_highlights(corrected, threshold=highlight_threshold)

    return {
        "original": gray,
        "tool_mask": tool_mask,
        "tool_roi": tool_roi,
        "corrected": corrected,
        "highlight_mask": highlight_mask,
        "final": final,
    }


# ── SNR helper ────────────────────────────────────────────────────────────────

def compute_snr(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Estimate SNR (dB) on the image or masked region.
    SNR = 20 * log10(mean / std)
    """
    roi = image[mask > 0] if mask is not None else image.ravel()
    roi = roi[roi > 0]  # exclude background zeros
    if roi.std() == 0:
        return float("inf")
    return float(20 * np.log10(roi.mean() / roi.std()))


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="M2 — Illumination correction pipeline")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--sigma", type=float, default=30.0, help="Homomorphic filter sigma")
    parser.add_argument("--inpaint", action="store_true", help="Apply highlight inpainting")
    parser.add_argument("--output", default="output", help="Output folder")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_m2(args.image, homomorphic_sigma=args.sigma, apply_inpainting=args.inpaint)

    stem = Path(args.image).stem
    cv2.imwrite(str(out_dir / f"{stem}_tool_mask.png"), result["tool_mask"])
    cv2.imwrite(str(out_dir / f"{stem}_tool_roi.png"), result["tool_roi"])
    cv2.imwrite(str(out_dir / f"{stem}_corrected.png"), result["corrected"])
    cv2.imwrite(str(out_dir / f"{stem}_final.png"), result["final"])

    snr_before = compute_snr(result["tool_roi"], result["tool_mask"])
    snr_after  = compute_snr(result["final"],    result["tool_mask"])
    print(f"SNR before: {snr_before:.2f} dB")
    print(f"SNR after:  {snr_after:.2f} dB")
    print(f"SNR gain:   {snr_after - snr_before:+.2f} dB")
    print(f"Outputs saved to: {out_dir}")
    sys.exit(0)

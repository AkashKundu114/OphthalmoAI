from __future__ import annotations
from typing import List, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

LAPLACIAN_VARIANCE_MIN = 80.0
BRIGHTNESS_MIN = 30.0
BRIGHTNESS_MAX = 230.0


def assess_image_quality(image_pil: Image.Image) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    img = np.array(image_pil.convert("RGB"))

    if not CV2_AVAILABLE:
        gray = np.mean(img, axis=2)
        mean_brightness = float(gray.mean())
        _check_brightness(mean_brightness, issues)
        issues.append("Image sharpness and eye-region detection were skipped (OpenCV not installed).")
        return (len(issues) == 0, issues)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    try:
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        gx, gy = np.gradient(gray.astype(np.float64))
        laplacian_var = float(np.var(gx) + np.var(gy))
    if laplacian_var < LAPLACIAN_VARIANCE_MIN:
        issues.append(f"Image appears blurry (sharpness score {laplacian_var:.0f}).")

    mean_brightness = float(gray.mean())
    _check_brightness(mean_brightness, issues)

    try:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=200)
        if circles is None:
            issues.append("No clear iris/pupil region was detected.")
    except Exception:
        pass

    return (len(issues) == 0, issues)


def _check_brightness(mean_brightness: float, issues: List[str]) -> None:
    if mean_brightness < BRIGHTNESS_MIN:
        issues.append(f"Image is too dark (brightness {mean_brightness:.0f}/255).")
    elif mean_brightness > BRIGHTNESS_MAX:
        issues.append(f"Image is overexposed (brightness {mean_brightness:.0f}/255).")


from io import BytesIO

def compress_retinal_image(image_pil: Image.Image, target_kb: int = 150) -> Tuple[bytes, int, int]:
    """
    Compresses high-res fundus images using localized edge preservation and quantization.
    Returns (compressed_bytes, original_size_bytes, compressed_size_bytes).
    """
    img = image_pil.copy()
    if img.width > 1024 or img.height > 1024:
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
    orig_io = BytesIO()
    image_pil.save(orig_io, format="JPEG", quality=95)
    orig_size = len(orig_io.getvalue())
    
    compressed_bytes = b""
    quality = 85
    while quality >= 20:
        out_io = BytesIO()
        img.save(out_io, format="JPEG", quality=quality)
        compressed_bytes = out_io.getvalue()
        if len(compressed_bytes) <= target_kb * 1024:
            break
        quality -= 10
        
    return compressed_bytes, orig_size, len(compressed_bytes)

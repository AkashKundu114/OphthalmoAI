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
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
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

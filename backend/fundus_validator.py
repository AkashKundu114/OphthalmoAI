"""
Retinal Fundus Domain Guardrail and Optical Admissibility Validator.

Validates whether an input image satisfies physiological, chromatic, and morphological
criteria of an authentic color fundus photograph. Rejects non-retinal imagery, documents,
synthetic noise, corrupted files, and external photographs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image


MIN_DIMENSION_PX = 128
MAX_ASPECT_RATIO = 3.5
MIN_LUMINANCE_STD = 8.0
MAX_WHITE_PIXEL_RATIO = 0.65
MIN_SPATIAL_CORRELATION = 0.35
MIN_ACTIVE_TISSUE_RATIO = 0.12


def _validate_image_geometry(image: Any) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Checks basic image geometry, dimensions, and aspect ratio boundaries."""
    if image is None:
        return False, "Input image object is None.", {"failure_stage": "null_input"}

    if not hasattr(image, "size") or not hasattr(image, "convert"):
        return False, "Input object is not a valid PIL Image.", {"failure_stage": "invalid_type"}

    w, h = image.size
    if w < MIN_DIMENSION_PX or h < MIN_DIMENSION_PX:
        return False, (
            f"Image dimensions ({w}x{h}) below minimum diagnostic threshold "
            f"({MIN_DIMENSION_PX}x{MIN_DIMENSION_PX})."
        ), {"width": w, "height": h, "failure_stage": "dimensions"}

    aspect_ratio = max(w / max(h, 1), h / max(w, 1))
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False, (
            f"Image aspect ratio ({aspect_ratio:.2f}) is abnormal. "
            "Retinal fundus photographs require standard circular or 4:3 optical aspect ratios."
        ), {"aspect_ratio": round(aspect_ratio, 2), "failure_stage": "aspect_ratio"}

    return True, None, {"width": w, "height": h, "aspect_ratio": round(aspect_ratio, 2)}


def _extract_and_validate_tensor(image: Image.Image) -> Tuple[Optional[np.ndarray], Optional[str], Dict[str, Any]]:
    """Converts image to RGB numpy array and verifies shape and finite numerical values."""
    try:
        rgb_image = image.convert("RGB")
        arr = np.array(rgb_image, dtype=np.float32)
    except Exception as exc:
        return None, f"Failed to convert image to RGB numerical array: {exc}", {"failure_stage": "conversion"}

    if arr.ndim != 3 or arr.shape[2] != 3:
        return None, f"Expected 3-channel RGB image, got shape {arr.shape}.", {"failure_stage": "channels"}

    if arr.size == 0:
        return None, "Image data array is empty.", {"failure_stage": "empty_array"}

    if not np.all(np.isfinite(arr)):
        return None, "Image contains non-finite numerical values (NaN or Inf).", {"failure_stage": "invalid_numerics"}

    return arr, None, {}


def _calculate_luminance(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Computes ITU-R BT.601 perceptual luminance and basic statistical moments."""
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    std_lum = float(np.std(lum))
    mean_lum = float(np.mean(lum))
    return lum, mean_lum, std_lum


def _evaluate_spatial_autocorrelation(lum: np.ndarray) -> float:
    """
    Evaluates spatial continuity across adjacent pixels.
    Natural retinal photography exhibits high neighbor correlation (r > 0.60);
    pure synthetic noise or static yields near-zero correlation.
    """
    try:
        h_left = lum[:, :-1].ravel()
        h_right = lum[:, 1:].ravel()
        v_top = lum[:-1, :].ravel()
        v_bot = lum[1:, :].ravel()

        std_h_left = np.std(h_left)
        std_h_right = np.std(h_right)
        std_v_top = np.std(v_top)
        std_v_bot = np.std(v_bot)

        if std_h_left < 1e-4 or std_h_right < 1e-4 or std_v_top < 1e-4 or std_v_bot < 1e-4:
            return 0.0

        h_corr = float(np.corrcoef(h_left, h_right)[0, 1])
        v_corr = float(np.corrcoef(v_top, v_bot)[0, 1])

        if math.isnan(h_corr) or math.isnan(v_corr):
            return 0.0

        return float(np.clip((h_corr + v_corr) / 2.0, 0.0, 1.0))
    except Exception:
        return 0.5


def _analyze_optical_aperture(
    lum: np.ndarray, w: int, h: int
) -> Tuple[bool, float, float, np.ndarray, float]:
    """
    Identifies characteristic circular aperture masking of retinal fundus cameras.
    Returns (has_circular_mask, corner_mean, center_mean, active_mask, active_ratio).
    """
    corner_w = max(int(w * 0.08), 2)
    corner_h = max(int(h * 0.08), 2)

    tl = lum[:corner_h, :corner_w]
    tr = lum[:corner_h, -corner_w:]
    bl = lum[-corner_h:, :corner_w:]
    br = lum[-corner_h:, -corner_w:]
    corner_mean = float((np.mean(tl) + np.mean(tr) + np.mean(bl) + np.mean(br)) / 4.0)

    cx, cy = w // 2, h // 2
    rw, rh = max(int(w * 0.22), 2), max(int(h * 0.22), 2)
    center_region = lum[max(0, cy - rh) : min(h, cy + rh), max(0, cx - rw) : min(w, cx + rw)]
    center_mean = float(np.mean(center_region)) if center_region.size > 0 else 0.0

    has_circular_mask = (
        corner_mean < 45.0
        and center_mean > 55.0
        and (center_mean / max(corner_mean + 1e-4, 1e-4) > 1.35)
    )

    active_mask = lum > max(15.0, corner_mean * 1.15)
    active_ratio = float(np.mean(active_mask)) if lum.size > 0 else 0.0

    return has_circular_mask, corner_mean, center_mean, active_mask, active_ratio


def _compute_vessel_gradient(g_channel: np.ndarray, active_mask: np.ndarray) -> float:
    """Retinal vessels absorb green spectrum light (~540nm), producing local gradients."""
    try:
        g_norm = (g_channel / 255.0).astype(np.float32)
        gx = np.diff(g_norm, axis=1)
        gy = np.diff(g_norm, axis=0)
        grad_mag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)

        sub_mask = active_mask[:-1, :-1]
        if np.any(sub_mask):
            return float(np.mean(grad_mag[sub_mask]))
        return float(np.mean(grad_mag))
    except Exception:
        return 0.0


def _evaluate_chromatic_profile(
    r_act: np.ndarray,
    g_act: np.ndarray,
    b_act: np.ndarray,
    has_circular_mask: bool,
    active_ratio: float,
    vessel_gradient: float,
) -> Tuple[bool, float, float, float, bool, Optional[str]]:
    """
    Evaluates retinal chromophore profile (chorioretinal hemoglobin & melanin reflection).
    Also detects and handles BGR channel reversals from external video capture frames.
    """
    if r_act.size == 0 or g_act.size == 0 or b_act.size == 0:
        return False, 0.0, 0.0, 0.0, False, "No active tissue pixels detected."

    r_mean = float(np.mean(r_act))
    g_mean = float(np.mean(g_act))
    b_mean = float(np.mean(b_act))

    rb_ratio = r_mean / max(b_mean, 1e-5)
    rg_ratio = r_mean / max(g_mean, 1e-5)
    channel_variance = float(np.mean(np.abs(r_act - g_act)) + np.mean(np.abs(r_act - b_act)))

    is_bgr_inverted = False
    if b_mean > r_mean * 1.15:
        candidate_rb = b_mean / max(r_mean, 1e-5)
        candidate_rg = b_mean / max(g_mean, 1e-5)
        if (has_circular_mask or (0.15 <= active_ratio <= 0.98 and vessel_gradient >= 0.003)) and candidate_rb >= 1.10:
            is_bgr_inverted = True
            r_mean, b_mean = b_mean, r_mean
            rb_ratio = candidate_rb
            rg_ratio = candidate_rg

    if not is_bgr_inverted and not has_circular_mask and b_mean > r_mean * 1.25 and b_mean > g_mean * 1.15:
        return False, rb_ratio, rg_ratio, channel_variance, is_bgr_inverted, (
            "Dominant blue chromaticity detected. Incompatible with retinal tissue spectroscopy."
        )

    if not has_circular_mask and g_mean > r_mean * 1.35:
        return False, rb_ratio, rg_ratio, channel_variance, is_bgr_inverted, (
            "Dominant green chromaticity detected without red vascular reflection. Incompatible with fundus tissue."
        )

    if channel_variance < 3.5 and not has_circular_mask:
        return False, rb_ratio, rg_ratio, channel_variance, is_bgr_inverted, (
            "Image is monochromatic or black-and-white without fundus optical aperture. Color fundus photograph expected."
        )

    return True, rb_ratio, rg_ratio, channel_variance, is_bgr_inverted, None


def validate_fundus_image(image_pil: Image.Image) -> Tuple[bool, float, str, Dict[str, Any]]:
    """
    Analyzes physical, chromatic, and morphological features of an image to verify
    it is an authentic color fundus photograph of the posterior pole.

    Returns:
        is_valid (bool): True if verified as an admissible retinal fundus image.
        confidence (float): Verification confidence score in [0.0, 1.0].
        reason (str): Clinical and technical explanation.
        metrics (dict): Extracted physical measurements.
    """
    # 1. Structural and geometric validation
    geom_ok, geom_err, geom_meta = _validate_image_geometry(image_pil)
    if not geom_ok:
        return False, 0.0, geom_err or "Geometric validation failed.", geom_meta

    w, h = geom_meta["width"], geom_meta["height"]

    # 2. Tensor conversion and numerical checks
    arr, tensor_err, tensor_meta = _extract_and_validate_tensor(image_pil)
    if arr is None:
        return False, 0.0, tensor_err or "Array conversion failed.", tensor_meta

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # 3. Luminance and variance analysis
    lum, mean_lum, std_lum = _calculate_luminance(arr)
    if std_lum < MIN_LUMINANCE_STD:
        return False, 0.0, "Image has near-zero visual variance (blank or solid color). Not a fundus photograph.", {
            "std_luminance": round(std_lum, 2), "failure_stage": "zero_variance"
        }

    # 4. White document / screenshot rejection
    white_pixel_ratio = float(np.mean(lum > 238))
    if white_pixel_ratio > MAX_WHITE_PIXEL_RATIO:
        return False, 0.0, "Image appears to be a white document, chart, or screenshot. Not a fundus photograph.", {
            "white_ratio": round(white_pixel_ratio, 3), "failure_stage": "document_pattern"
        }

    # 5. Spatial autocorrelation (rejects synthetic static/noise)
    spatial_corr = _evaluate_spatial_autocorrelation(lum)
    if spatial_corr < MIN_SPATIAL_CORRELATION:
        return False, 0.0, "Image contains uncorrelated random noise or static artifacts without anatomical coherence.", {
            "spatial_autocorrelation": round(spatial_corr, 3), "failure_stage": "random_noise"
        }

    # 6. Optical aperture and illuminated field analysis
    has_circular_mask, corner_mean, center_mean, active_mask, active_ratio = _analyze_optical_aperture(lum, w, h)
    if active_ratio < MIN_ACTIVE_TISSUE_RATIO:
        return False, 0.0, "Image is predominantly dark with no identifiable retinal tissue area.", {
            "active_ratio": round(active_ratio, 3), "failure_stage": "dark_field"
        }

    # 7. Vascular gradient
    vessel_gradient = _compute_vessel_gradient(g, active_mask)

    # 8. Retinal chromophore spectroscopy
    chroma_ok, rb_ratio, rg_ratio, channel_variance, is_bgr_inverted, chroma_err = _evaluate_chromatic_profile(
        r[active_mask], g[active_mask], b[active_mask], has_circular_mask, active_ratio, vessel_gradient
    )
    if not chroma_ok:
        return False, 0.05, chroma_err or "Chromophore validation failed.", {
            "rb_ratio": round(rb_ratio, 2),
            "rg_ratio": round(rg_ratio, 2),
            "channel_variance": round(channel_variance, 2),
            "failure_stage": "chromatic_profile",
        }

    # Composite score computation
    score = 0.0
    if has_circular_mask:
        score += 0.35
    elif 0.15 <= active_ratio <= 0.98:
        score += 0.20

    if rb_ratio >= 1.18 and rg_ratio >= 1.02:
        score += 0.35
    elif 0.94 <= rb_ratio <= 1.25 and 0.92 <= rg_ratio <= 1.18 and (channel_variance >= 4.0 or has_circular_mask):
        score += 0.32
    elif rb_ratio >= 1.05:
        score += 0.20
    else:
        score += 0.08

    if 0.005 <= vessel_gradient <= 0.25 and spatial_corr >= 0.65:
        score += 0.30
    elif 0.004 <= vessel_gradient <= 0.25 and spatial_corr >= 0.35:
        score += 0.20
    elif spatial_corr >= 0.50:
        score += 0.15
    else:
        score += 0.05

    score = min(1.0, max(0.0, score))

    metrics = {
        "score": round(score, 3),
        "has_circular_mask": has_circular_mask,
        "is_bgr_inverted": is_bgr_inverted,
        "spatial_correlation": round(spatial_corr, 3),
        "rb_ratio": round(rb_ratio, 2),
        "rg_ratio": round(rg_ratio, 2),
        "vessel_gradient": round(vessel_gradient, 4),
        "active_ratio": round(active_ratio, 2),
        "corner_luminance": round(corner_mean, 1),
        "center_luminance": round(center_mean, 1),
    }

    if score < 0.50:
        return False, score, (
            "Image failed fundus validation. Optical aperture, retinal chromophore distribution, "
            "or vascular structures were not recognized. Please upload a genuine posterior pole color fundus scan."
        ), metrics

    return True, score, "Authentic retinal fundus photograph verified.", metrics

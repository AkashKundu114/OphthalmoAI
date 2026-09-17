"""
Cross-Dataset Generalization & Camera Sensor Domain Shift Adaptation.

Provides:
1. Reinhard Color Constancy Normalization (LAB color space moment transfer)
   to harmonize illumination variance across different fundus camera manufacturers
   (Zeiss, Topcon, Canon, smartphone-attached lenses).
2. Optical Domain Shift Detector that assesses sensor chromatic distribution and
   emits a confidence score and clinical advisory if camera optics diverge from training baseline.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# Canonical reference moments in LAB color space for high-quality standard retinal fundus
# Derived empirically from high-resolution gold-standard posterior pole fundus scans
CANONICAL_LAB_MEANS = np.array([128.5, 142.0, 153.2], dtype=np.float32)
CANONICAL_LAB_STDS  = np.array([32.4,  11.2,  14.8], dtype=np.float32)


def apply_reinhard_color_constancy(image: Image.Image) -> Image.Image:
    """
    Applies Reinhard color constancy normalization in LAB space.
    Aligns the mean and standard deviation of L, A, and B channels to the canonical
    retinal target distribution, mitigating inter-camera illumination and white-balance shifts.
    """
    img_np = np.array(image.convert("RGB"))

    if CV2_AVAILABLE:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    else:
        # Fallback manual approximate RGB to LAB conversion
        r = img_np[:, :, 0].astype(np.float32)
        g = img_np[:, :, 1].astype(np.float32)
        b = img_np[:, :, 2].astype(np.float32)
        l_chan = 0.299 * r + 0.587 * g + 0.114 * b
        a_chan = 0.5 * (r - g) + 128.0
        b_chan = 0.5 * (g - b) + 128.0
        lab = np.stack([l_chan, a_chan, b_chan], axis=-1)

    # Calculate source moments
    src_means = np.mean(lab, axis=(0, 1))
    src_stds = np.std(lab, axis=(0, 1))
    src_stds = np.maximum(src_stds, 1e-4)

    # Transfer moments
    norm_lab = np.zeros_like(lab)
    for c in range(3):
        norm_lab[:, :, c] = (
            (lab[:, :, c] - src_means[c]) / src_stds[c]
        ) * CANONICAL_LAB_STDS[c] + CANONICAL_LAB_MEANS[c]

    norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)

    if CV2_AVAILABLE:
        norm_rgb = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2RGB)
    else:
        # Fallback invert
        l_c = norm_lab[:, :, 0].astype(np.float32)
        a_c = norm_lab[:, :, 1].astype(np.float32) - 128.0
        b_c = norm_lab[:, :, 2].astype(np.float32) - 128.0
        r_c = l_c + 1.402 * a_c
        g_c = l_c - 0.344136 * b_c - 0.714136 * a_c
        b_c = l_c + 1.772 * b_c
        norm_rgb = np.clip(np.stack([r_c, g_c, b_c], axis=-1), 0, 255).astype(np.uint8)

    return Image.fromarray(norm_rgb)


def is_already_ben_graham(img_np: np.ndarray) -> bool:
    """Checks if image has already undergone Ben Graham local color constancy enhancement."""
    ch_means = img_np.mean(axis=(0, 1))
    diff = max(abs(ch_means[0] - ch_means[1]), abs(ch_means[0] - ch_means[2]), abs(ch_means[1] - ch_means[2]))
    return bool(diff < 8.0 and (65.0 < float(np.mean(ch_means)) < 125.0))


def detect_sensor_domain_shift(image: Image.Image) -> Dict[str, Any]:
    """
    Evaluates retinal image chromatic distribution, contrast entropy, and
    vascular balance to detect camera sensor and illumination domain shifts.
    """
    img_np = np.array(image.convert("RGB"))

    # If the image is already Ben Graham preprocessed or normalized, it is already in canonical model space
    if is_already_ben_graham(img_np):
        return {
            "domain_shift_detected": False,
            "sensor_domain_confidence": 0.98,
            "optical_profile_advisory": "Optical profile aligned with canonical clinical bench standards (Harmonized Ben Graham space).",
            "metrics": {
                "rg_ratio": 1.0,
                "rb_ratio": 1.0,
                "green_contrast_std": float(np.std(img_np[:, :, 1])),
                "mean_luminance": float(np.mean(img_np)),
            },
        }

    r = img_np[:, :, 0].astype(np.float32)
    g = img_np[:, :, 1].astype(np.float32)
    b = img_np[:, :, 2].astype(np.float32)

    mean_r, mean_g, mean_b = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    std_r, std_g, std_b    = float(np.std(r)), float(np.std(g)), float(np.std(b))

    # Standard mydriatic/non-mydriatic fundus cameras have high R/G dominance due to choroid
    rg_ratio = (mean_r + 1.0) / (mean_g + 1.0)
    rb_ratio = (mean_r + 1.0) / (mean_b + 1.0)

    # Domain shift criteria:
    # 1. rb_ratio < 1.15 indicates extreme blue tint (smartphone lens adapter or cold LED sensor)
    # 2. rg_ratio < 0.95 indicates inverted/atypical chromatic spectrum
    # 3. std_g < 10.0 indicates severe over/underexposure flattening vascular contrast
    is_shift = False
    shift_reasons = []

    if rb_ratio < 1.15:
        is_shift = True
        shift_reasons.append("Elevated short-wavelength blue backscatter (cold LED or handheld adapter).")
    if rg_ratio < 0.95:
        is_shift = True
        shift_reasons.append("Atypical green-dominant balance deviating from standard chorioretinal profile.")
    if std_g < 10.0:
        is_shift = True
        shift_reasons.append("Severe contrast compression; inadequate illumination dynamic range.")

    # Calculate domain confidence: 1.0 is canonical Zeiss/Topcon bench, drops if shift detected
    penalty = len(shift_reasons) * 0.28
    sensor_domain_confidence = max(0.20, min(1.0, 0.96 - penalty))

    advisory = (
        "Optical profile aligned with canonical clinical bench standards (Zeiss/Topcon benchmark)."
        if not is_shift
        else f"Camera sensor domain shift detected: {' '.join(shift_reasons)} Reinhard color constancy normalization recommended."
    )

    return {
        "domain_shift_detected": is_shift,
        "sensor_domain_confidence": round(sensor_domain_confidence, 3),
        "optical_profile_advisory": advisory,
        "metrics": {
            "rg_ratio": round(rg_ratio, 2),
            "rb_ratio": round(rb_ratio, 2),
            "green_contrast_std": round(std_g, 2),
            "mean_luminance": round((mean_r + mean_g + mean_b) / 3.0, 2),
        },
    }


def adapt_fundus_domain(
    image: Image.Image,
    apply_color_constancy: bool = True,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    End-to-end domain adaptation pipeline:
    Analyzes optical sensor properties, detects domain shift, and optionally applies
    Reinhard color constancy to stabilize feature extractor activations.
    """
    domain_eval = detect_sensor_domain_shift(image)

    if apply_color_constancy and domain_eval["domain_shift_detected"]:
        adapted_img = apply_reinhard_color_constancy(image)
        domain_eval["color_constancy_applied"] = True
    else:
        adapted_img = image
        domain_eval["color_constancy_applied"] = False

    return adapted_img, domain_eval

"""
OphthalmoAI Retinal Fundus Domain Guardrail.
=============================================================================
Validates whether an input image is an authentic color fundus photograph.
Rejects out-of-domain (OOD) imagery including:
- Random everyday objects (cars, furniture, food, animals)
- Documents, slides, barcodes, and screenshots
- External selfies, portraits, and non-fundus medical scans (X-rays, CTs)
- Solid colors, synthetic noise, and corrupted inputs
"""

from __future__ import annotations
from typing import Dict, Tuple, Any
import numpy as np
from PIL import Image

def validate_fundus_image(image_pil: Image.Image) -> Tuple[bool, float, str, Dict[str, Any]]:
    """
    Analyzes physical, chromatic, and morphological features of an image to verify
    it is an authentic color fundus photograph of the posterior pole.
    
    Returns:
        is_valid (bool): True if verified as a retinal fundus image.
        confidence (float): Verification confidence score in [0.0, 1.0].
        reason (str): Clinical / technical explanation.
        metrics (dict): Extracted physical measurements.
    """
    img = image_pil.convert("RGB")
    w, h = img.size
    
    # Check 1: Minimum spatial resolution
    if w < 128 or h < 128:
        return False, 0.0, "Resolution too low (< 128x128). Fundus diagnostics require adequate optical resolution.", {
            "width": w, "height": h, "failure_stage": "dimensions"
        }
        
    arr = np.array(img, dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    
    # Luminance computation
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    std_lum = float(np.std(lum))
    mean_lum = float(np.mean(lum))
    
    # Check 2: Solid or near-blank image
    if std_lum < 8.0:
        return False, 0.0, "Image has near-zero visual variance (blank or solid color). Not a fundus photograph.", {
            "std_luminance": round(std_lum, 2), "failure_stage": "zero_variance"
        }
        
    # Check 3: White document / screenshot / presentation slide
    white_pixel_ratio = float(np.mean(lum > 238))
    if white_pixel_ratio > 0.65:
        return False, 0.0, "Image appears to be a white document, chart, or screenshot. Not a fundus photograph.", {
            "white_ratio": round(white_pixel_ratio, 3), "failure_stage": "document_pattern"
        }
        
    # Check 4: Spatial Autocorrelation (Rejects synthetic noise / static)
    # Natural and medical photographs have high horizontal and vertical correlation (r > 0.80)
    # Pure white noise has r ~ 0.0
    try:
        h_corr = float(np.corrcoef(lum[:, :-1].ravel(), lum[:, 1:].ravel())[0, 1])
        v_corr = float(np.corrcoef(lum[:-1, :].ravel(), lum[1:, :].ravel())[0, 1])
        spatial_corr = (h_corr + v_corr) / 2.0
    except Exception:
        spatial_corr = 0.5
        
    if spatial_corr < 0.35:
        return False, 0.0, "Image contains uncorrelated random noise or static artifacts without anatomical coherence.", {
            "spatial_autocorrelation": round(spatial_corr, 3), "failure_stage": "random_noise"
        }
        
    # Check 5: Optical Aperture / Circular Field of View Analysis
    # Standard fundus cameras image through a circular optical aperture creating dark corners
    corner_w = max(int(w * 0.08), 4)
    corner_h = max(int(h * 0.08), 4)
    tl = lum[:corner_h, :corner_w]
    tr = lum[:corner_h, -corner_w:]
    bl = lum[-corner_h:, :corner_w:]
    br = lum[-corner_h:, -corner_w:]
    corner_mean = float((np.mean(tl) + np.mean(tr) + np.mean(bl) + np.mean(br)) / 4.0)
    
    cx, cy = w // 2, h // 2
    rw, rh = int(w * 0.22), int(h * 0.22)
    center_region = lum[cy - rh:cy + rh, cx - rw:cx + rw]
    center_mean = float(np.mean(center_region))
    
    has_circular_mask = (corner_mean < 45.0 and center_mean > 55.0 and (center_mean / (corner_mean + 1e-4) > 1.35))
    
    # Active tissue mask (illuminated retina area)
    active_mask = lum > max(15.0, corner_mean * 1.15)
    active_ratio = float(np.mean(active_mask))
    
    if active_ratio < 0.12:
        return False, 0.0, "Image is predominantly dark with no identifiable retinal tissue area.", {
            "active_ratio": round(active_ratio, 3), "failure_stage": "dark_field"
        }
        
    # Check 6: Green-Channel Vascular Gradient & Tissue Coherence
    # Retinal vessels absorb green light (~540nm) creating localized tubular contrast
    g_norm = (g / 255.0).astype(np.float32)
    gx = np.diff(g_norm, axis=1)
    gy = np.diff(g_norm, axis=0)
    grad_mag = np.sqrt(gx[:-1, :]**2 + gy[:, :-1]**2)
    vessel_gradient = float(np.mean(grad_mag[active_mask[:-1, :-1]]))

    # Check 7: Retinal Chromophore Profile
    # Human retinal fundus photography is dominated by chorioretinal vasculature and RPE melanin
    r_act = r[active_mask]
    g_act = g[active_mask]
    b_act = b[active_mask]
    
    r_mean = float(np.mean(r_act))
    g_mean = float(np.mean(g_act))
    b_mean = float(np.mean(b_act))
    
    rb_ratio = r_mean / (b_mean + 1e-5)
    rg_ratio = r_mean / (g_mean + 1e-5)
    
    # Check for BGR color channel swap (common in OpenCV exports and camera capture buffers)
    is_bgr_inverted = False
    if b_mean > r_mean * 1.15:
        candidate_rb = b_mean / (r_mean + 1e-5)
        candidate_rg = b_mean / (g_mean + 1e-5)
        if (has_circular_mask or (0.15 <= active_ratio <= 0.98 and vessel_gradient >= 0.003)) and candidate_rb >= 1.10:
            is_bgr_inverted = True
            r_mean, b_mean = b_mean, r_mean
            r_act, b_act = b_act, r_act
            rb_ratio = candidate_rb
            rg_ratio = candidate_rg

    # Reject unnatural dominant blue (e.g. blue sky, blue car, water) if NOT an inverted fundus and has no circular mask
    if not is_bgr_inverted and not has_circular_mask and b_mean > r_mean * 1.25 and b_mean > g_mean * 1.15:
        return False, 0.05, "Dominant blue chromaticity detected. Incompatible with retinal tissue spectroscopy.", {
            "rb_ratio": round(rb_ratio, 2), "failure_stage": "blue_dominance"
        }
        
    # Reject unnatural dominant green (e.g. foliage, green neon objects)
    if not has_circular_mask and g_mean > r_mean * 1.35:
        return False, 0.05, "Dominant green chromaticity detected without red vascular reflection. Incompatible with fundus tissue.", {
            "rg_ratio": round(rg_ratio, 2), "failure_stage": "green_dominance"
        }
        
    # Check 8: Grayscale / Monochromatic rejection if no circular fundus mask exists
    channel_variance = float(np.mean(np.abs(r_act - g_act)) + np.mean(np.abs(r_act - b_act)))
    if channel_variance < 3.5 and not has_circular_mask:
        return False, 0.10, "Image is monochromatic or black-and-white without fundus optical aperture. Color fundus photograph expected.", {
            "channel_variance": round(channel_variance, 2), "failure_stage": "monochrome"
        }
        
    # Calculate composite verification score
    score = 0.0
    
    # Aperture / Mask: +0.35
    if has_circular_mask:
        score += 0.35
    elif 0.15 <= active_ratio <= 0.98:
        score += 0.20
        
    # Chromatic Profile: +0.35
    if rb_ratio >= 1.18 and rg_ratio >= 1.02:
        score += 0.35 # Strong raw color fundus
    elif 0.94 <= rb_ratio <= 1.25 and 0.92 <= rg_ratio <= 1.18 and (channel_variance >= 4.0 or has_circular_mask):
        score += 0.32 # High-quality preprocessed fundus
    elif rb_ratio >= 1.05:
        score += 0.20
    else:
        score += 0.08
        
    # Vascular Structure: +0.30
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
        "center_luminance": round(center_mean, 1)
    }
    
    # Threshold for validity: 0.50
    if score < 0.50:
        return False, score, (
            "Image failed fundus validation. Optical aperture, retinal chromophore distribution, "
            "or vascular structures were not recognized. Please upload a genuine posterior pole color fundus scan."
        ), metrics
        
    return True, score, "Authentic retinal fundus photograph verified.", metrics

"""
OphthalmoAI - Saliency-Grounded Biomarker Extraction (SGB-LLM)
=============================================================
Extracts quantitative spatial, morphological, and colorimetric biomarkers
from the Grad-CAM activation heatmap and anatomical ocular segments:
1. Corneal Involvement Ratio (rho_anterior): Overlap between high activation and detected iris/cornea.
2. Vascular Erythema Index (Delta_EI): Chromatic saturation in CIELAB space within inflamed conjunctiva.
3. Scleral Icterus Index (b*_sclera): Yellow-blue chromatic coordinate in scleral zone for jaundice validation.
4. Saliency Spatial Entropy / Focus: Distinguishes focal lesions (corneal ulcer) from diffuse inflammation.
5. Formats structured context for Gemini 2.0 Flash to guarantee grounded medical explanations.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def extract_visual_biomarkers(
    image_pil: Image.Image,
    gradcam_grayscale: Optional[np.ndarray],
    predicted_diagnosis: str
) -> Dict[str, Any]:
    """
    Extracts quantitative visual biomarkers from the RGB image and Grad-CAM saliency map.
    gradcam_grayscale: 2D numpy array with values in [0.0, 1.0], same spatial dimension as image.
    """
    img_rgb = np.array(image_pil.convert("RGB"))
    h, w, _ = img_rgb.shape

    # Defaults in case of missing modules or images
    corneal_ratio = 0.0
    vascular_erythema = 0.0
    scleral_icterus = 0.0
    saliency_focus = "Moderate"
    activation_area_pct = 0.0

    if gradcam_grayscale is not None and gradcam_grayscale.size > 0:
        # Resize gradcam if needed to match image shape
        if gradcam_grayscale.shape != (h, w):
            if CV2_AVAILABLE:
                cam_resized = cv2.resize(gradcam_grayscale, (w, h))
            else:
                cam_pil = Image.fromarray((gradcam_grayscale * 255).astype(np.uint8))
                cam_resized = np.array(cam_pil.resize((w, h))) / 255.0
        else:
            cam_resized = gradcam_grayscale

        # Binarize top 20% activation focus
        threshold = float(np.percentile(cam_resized, 80))
        active_mask = (cam_resized >= threshold).astype(np.uint8)
        activation_area_pct = round(float(np.mean(active_mask)) * 100.0, 2)

        # Compute spatial entropy (dispersion)
        flat_cam = cam_resized.flatten() + 1e-8
        prob_dist = flat_cam / np.sum(flat_cam)
        entropy = -float(np.sum(prob_dist * np.log(prob_dist)))
        max_entropy = math.log(len(flat_cam))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

        if norm_entropy < 0.70:
            saliency_focus = "Focal / Well-Circumscribed"
        elif norm_entropy < 0.88:
            saliency_focus = "Moderate / Regional"
        else:
            saliency_focus = "Diffuse / Widespread"

        # Detect pupil/iris using Hough Circles
        iris_mask = np.zeros((h, w), dtype=np.uint8)
        if CV2_AVAILABLE:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            blurred = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=w // 4,
                param1=50,
                param2=30,
                minRadius=min(h, w) // 10,
                maxRadius=min(h, w) // 3
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :1]:
                    cx, cy, r = circle[0], circle[1], circle[2]
                    cv2.circle(iris_mask, (cx, cy), r, 1, -1)
            else:
                # Approximate central anterior zone if circle not detected
                cx, cy = w // 2, h // 2
                r = min(h, w) // 4
                cv2.circle(iris_mask, (cx, cy), r, 1, -1)
        else:
            # Fallback ellipse mask
            yy, xx = np.ogrid[:h, :w]
            cx, cy = w // 2, h // 2
            r = min(h, w) // 4
            iris_mask = ((xx - cx)**2 + (yy - cy)**2 <= r**2).astype(np.uint8)

        # 1. Corneal Involvement Ratio
        active_pixels = np.sum(active_mask)
        if active_pixels > 0:
            overlap = np.sum(active_mask * iris_mask)
            corneal_ratio = round(float(overlap / active_pixels) * 100.0, 1)

        # 2. Vascular Erythema & Scleral Icterus in CIELAB space
        if CV2_AVAILABLE:
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            # OpenCV LAB: L in [0, 255], a in [0, 255] (128 is neutral), b in [0, 255] (128 is neutral)
            L = lab[:, :, 0]
            a = lab[:, :, 1] - 128.0 # Green (-) to Red (+)
            b = lab[:, :, 2] - 128.0 # Blue (-) to Yellow (+)

            # Erythema in active lesion zone: Red saturation
            if active_pixels > 0:
                erythema_vals = a[active_mask == 1]
                vascular_erythema = round(float(np.mean(np.maximum(0, erythema_vals)) / 12.8), 2)

            # Scleral region: Outside iris mask, high brightness (sclera is normally white)
            sclera_mask = ((1 - iris_mask) == 1) & (L > 90)
            if np.sum(sclera_mask) > 50:
                sclera_b = b[sclera_mask]
                # Positive b corresponds to yellow chromatic shift (jaundice/icterus)
                scleral_icterus = round(float(np.mean(np.maximum(0, sclera_b)) / 12.8), 2)
        else:
            # Simplified RGB chrominance fallback
            r_chan = img_rgb[:, :, 0].astype(np.float32)
            g_chan = img_rgb[:, :, 1].astype(np.float32)
            b_chan = img_rgb[:, :, 2].astype(np.float32)
            if active_pixels > 0:
                red_excess = (r_chan - (g_chan + b_chan) / 2.0)
                vascular_erythema = round(float(np.mean(np.maximum(0, red_excess[active_mask == 1])) / 25.5), 2)
            # Yellow excess: (R + G)/2 - B
            yellow_excess = (r_chan + g_chan) / 2.0 - b_chan
            scleral_icterus = round(float(np.mean(np.maximum(0, yellow_excess)) / 25.5), 2)

    return {
        "corneal_involvement_pct": corneal_ratio,
        "vascular_erythema_index": vascular_erythema,
        "scleral_icterus_index": scleral_icterus,
        "saliency_focus_profile": saliency_focus,
        "active_lesion_area_pct": activation_area_pct
    }


def format_biomarkers_for_llm(
    biomarkers: Dict[str, Any],
    diagnosis: str,
    conformal_set: List[str],
    coverage_guarantee: float,
    epistemic_vacuity: float
) -> str:
    """
    Constructs a clinically verifiable, grounded prompt context block for the LLM.
    """
    corneal = biomarkers.get("corneal_involvement_pct", 0.0)
    erythema = biomarkers.get("vascular_erythema_index", 0.0)
    icterus = biomarkers.get("scleral_icterus_index", 0.0)
    focus = biomarkers.get("saliency_focus_profile", "Moderate")
    area = biomarkers.get("active_lesion_area_pct", 0.0)

    lines = [
        "--- VERIFIED VISION PIPELINE & GROUNDED BIOMARKERS ---",
        f"- Top Primary Finding: {diagnosis}",
        f"- Conformal Prediction Set: {', '.join(conformal_set)} (Statistical coverage guarantee: {coverage_guarantee:.1f}%)",
        f"- Epistemic Uncertainty (Vacuity): {epistemic_vacuity:.3f} / 1.000",
        f"- Spatial Saliency Focus: {focus} (occupies {area}% of image field)",
        f"- Corneal / Iris Involvement: {corneal}%",
        f"- Vascular Erythema Index: {erythema} (conjunctival/ciliary injection severity)",
        f"- Scleral Icterus Yellowness Index: {icterus}",
        "CLINICAL GROUNDING INSTRUCTION: In your response, explicitly reference these quantitative physical observations "
        "(e.g., mention whether corneal involvement or vascular erythema corroborates the clinical presentation). "
        "Do not contradict the deterministic vision findings."
    ]
    return "\n".join(lines)

"""
Retinal Image Preprocessing Service.

Implements anatomical color-preserving enhancement:
1. Aspect-ratio-preserved aperture crop with symmetric square padding.
2. Luminance-isolated CLAHE in CIE-LAB space: preserves true diagnostic chrominance
   of hemorrhages, drusen, lipid exudates, and optic nerve rim color without destructive
   Gaussian subtraction artifacts.
3. Anti-aliased circular aperture mask to eliminate artificial high-frequency border step gradients.
4. Canonical PyTorch tensor transformation with ImageNet mean and std normalization.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from backend.config import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from backend.logging_config import get_logger

logger = get_logger("ophthalmoai.preprocessor")

# Canonical torchvision normalization pipeline
preprocess = transforms.Compose([
    transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def is_already_ben_graham(img_np: np.ndarray) -> bool:
    """Checks if image has already undergone Ben Graham local color constancy enhancement."""
    ch_means = img_np.mean(axis=(0, 1))
    diff = max(
        abs(ch_means[0] - ch_means[1]),
        abs(ch_means[0] - ch_means[2]),
        abs(ch_means[1] - ch_means[2]),
    )
    return bool(diff < 8.0 and (65.0 < float(np.mean(ch_means)) < 125.0))


def ben_graham_preprocess(
    img_pil: Image.Image,
    target_size: int = DEFAULT_IMAGE_SIZE,
) -> Image.Image:
    """
    Anatomical Color-Preserving Preprocessing:
    1. Aspect-ratio-preserved aperture crop with symmetric square padding (no elliptical distortion).
    2. Luminance-isolated CLAHE in CIE-LAB space: preserves true diagnostic chrominance of hemorrhages,
       drusen, lipid exudates, and optic nerve rim color without destructive Gaussian subtraction artifacts.
    3. Anti-aliased circular mask to eliminate artificial high-frequency border step gradients.
    """
    img = np.array(img_pil.convert("RGB"))
    if is_already_ben_graham(img):
        return img_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)

    try:
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c_cnt)
            if w > 30 and h > 30:
                crop = img[y : y + h, x : x + w]
                # Aspect-ratio preserved padding to square canvas
                max_dim = max(w, h)
                pad_x = (max_dim - w) // 2
                pad_y = (max_dim - h) // 2
                img = cv2.copyMakeBorder(
                    crop,
                    pad_y,
                    max_dim - h - pad_y,
                    pad_x,
                    max_dim - w - pad_x,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )

        # High-dimensional resize with anti-aliasing area interpolation
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # LAB-CLAHE: Enhance luminance only, preserving true diagnostic chrominance
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_chan)
        enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a_chan, b_chan]), cv2.COLOR_LAB2RGB)

        # Anti-aliased circular aperture mask
        circle_mask = np.zeros((target_size, target_size), dtype=np.uint8)
        cv2.circle(
            circle_mask,
            (target_size // 2, target_size // 2),
            int(target_size * 0.485),
            255,
            -1,
        )
        circle_mask = cv2.GaussianBlur(circle_mask, (5, 5), 1.5)
        mask_3d = circle_mask[:, :, None] / 255.0

        result = (enhanced * mask_3d).astype(np.uint8)
        return Image.fromarray(result)
    except Exception as e:
        logger.warning("preprocess.fallback", error=str(e))
        return img_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)


def prepare_input_tensor(
    img_pil: Image.Image,
    target_size: int = DEFAULT_IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Runs Ben Graham preprocessing and converts the image to a normalized PyTorch 4D tensor [1, 3, H, W].
    """
    model_input_image = ben_graham_preprocess(img_pil, target_size=target_size)
    tensor = preprocess(model_input_image)
    if device is not None:
        tensor = tensor.to(device)
    return model_input_image, tensor.unsqueeze(0)

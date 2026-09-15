"""
Tests for backend/domain_adaptation.py (Camera Sensor Domain Shift & Reinhard Color Constancy).
"""

import numpy as np
from PIL import Image

from backend.domain_adaptation import (
    adapt_fundus_domain,
    apply_reinhard_color_constancy,
    detect_sensor_domain_shift,
)


def test_reinhard_color_constancy():
    # Create test synthetic image
    arr = np.random.randint(50, 200, size=(100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    normalized = apply_reinhard_color_constancy(img)

    assert isinstance(normalized, Image.Image)
    assert normalized.size == (100, 100)
    norm_arr = np.array(normalized)
    assert norm_arr.shape == (100, 100, 3)


def test_detect_canonical_fundus_domain():
    # Canonical fundus has high Red, medium Green, lower Blue
    w, h = 100, 100
    r = np.full((h, w), 180, dtype=np.uint8)
    g = np.full((h, w), 80, dtype=np.uint8)
    b = np.full((h, w), 40, dtype=np.uint8)
    # add some noise for variance
    g += np.random.randint(0, 60, size=(h, w), dtype=np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=-1))

    res = detect_sensor_domain_shift(img)
    assert "domain_shift_detected" in res
    assert "sensor_domain_confidence" in res
    assert res["domain_shift_detected"] is False
    assert res["sensor_domain_confidence"] > 0.8


def test_detect_shifted_fundus_domain():
    # Shifted image (e.g. cold blue smartphone lens where Blue is higher than Red)
    w, h = 100, 100
    r = np.full((h, w), 100, dtype=np.uint8)
    g = np.full((h, w), 120, dtype=np.uint8)
    b = np.full((h, w), 160, dtype=np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=-1))

    res = detect_sensor_domain_shift(img)
    assert res["domain_shift_detected"] is True
    assert res["sensor_domain_confidence"] < 0.8
    assert "Camera sensor domain shift detected" in res["optical_profile_advisory"]


def test_adapt_fundus_domain():
    # Testing adaptation pipeline
    w, h = 100, 100
    r = np.full((h, w), 90, dtype=np.uint8)
    g = np.full((h, w), 110, dtype=np.uint8)
    b = np.full((h, w), 150, dtype=np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=-1))

    adapted_img, eval_data = adapt_fundus_domain(img, apply_color_constancy=True)
    assert isinstance(adapted_img, Image.Image)
    assert eval_data["domain_shift_detected"] is True
    assert eval_data["color_constancy_applied"] is True

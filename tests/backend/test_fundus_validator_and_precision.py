"""
Unit and integration tests for:
1. Retinal Fundus Domain Validator (OAC-DG) & Non-Fundus Rejection Guardrails
2. Calibration metadata files (FP16 vs. BF16)
3. Precision Benchmark consistency across 6 target retinal classes
"""
from __future__ import annotations

import json
import os
import unittest
from io import BytesIO
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi.testclient import TestClient

import backend.main as bm
from backend.fundus_validator import validate_fundus_image


class TestFundusDomainValidator(unittest.TestCase):

    def _create_synthetic_fundus(self, size: int = 256) -> Image.Image:
        """Generates a geometrically and chromatically compliant synthetic fundus image."""
        y, x = np.mgrid[:size, :size]
        center = size // 2
        radius = size // 2 - 12
        mask = (x - center)**2 + (y - center)**2 <= radius**2

        arr = np.zeros((size, size, 3), dtype=np.uint8)
        # Retinal choroid: strong red, moderate green, minimal blue
        arr[mask, 0] = np.clip(185 + 20 * np.sin(x[mask] / 25.0), 120, 245).astype(np.uint8)
        arr[mask, 1] = np.clip(85 + 15 * np.cos(y[mask] / 25.0), 40, 145).astype(np.uint8)
        arr[mask, 2] = np.clip(30 + 10 * np.sin((x[mask] + y[mask]) / 35.0), 10, 60).astype(np.uint8)
        return Image.fromarray(arr)

    def test_valid_fundus_accepted(self):
        img = self._create_synthetic_fundus(256)
        is_valid, conf, reason, metrics = validate_fundus_image(img)
        self.assertTrue(is_valid)
        self.assertGreaterEqual(conf, 0.50)
        self.assertIn("fundus", reason.lower())
        self.assertTrue(metrics["has_circular_mask"])
        self.assertGreaterEqual(metrics["rb_ratio"], 1.05)

    def test_low_resolution_rejected(self):
        img = Image.new("RGB", (64, 64), color=(180, 80, 30))
        is_valid, conf, reason, metrics = validate_fundus_image(img)
        self.assertFalse(is_valid)
        self.assertEqual(conf, 0.0)
        self.assertEqual(metrics.get("failure_stage"), "dimensions")

    def test_blank_solid_image_rejected(self):
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        is_valid, conf, reason, metrics = validate_fundus_image(img)
        self.assertFalse(is_valid)
        self.assertEqual(metrics.get("failure_stage"), "zero_variance")

    def test_white_document_rejected(self):
        img = Image.new("RGB", (300, 300), color=(250, 250, 250))
        is_valid, conf, reason, metrics = validate_fundus_image(img)
        self.assertFalse(is_valid)
        self.assertIn(metrics.get("failure_stage"), ("document_pattern", "zero_variance"))

    def test_pure_gaussian_noise_rejected(self):
        arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        is_valid, conf, reason, metrics = validate_fundus_image(img)
        self.assertFalse(is_valid)
        self.assertLess(conf, 0.50)

    def test_blue_sky_scenery_rejected(self):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[:, :, 0] = 50   # Low red
        arr[:, :, 1] = 120  # Medium green
        arr[:, :, 2] = 220  # High blue (sky / water)
        img = Image.fromarray(arr)
        is_valid, conf, reason, metrics = validate_fundus_image(img)
        self.assertFalse(is_valid)


class TestPrecisionAndCalibrationMetadata(unittest.TestCase):

    def test_fp16_calibration_file(self):
        path = os.path.join("models", "calibration.json")
        self.assertTrue(os.path.exists(path), f"Missing {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        expected_models = ["efficientnet_v2_m", "densenet201", "convnext_small", "efficientnet_b4", "resnet50"]
        for m in expected_models:
            self.assertIn(m, data)
            t = data[m]
            self.assertIsInstance(t, (int, float))
            self.assertGreater(t, 0.5)
            self.assertLess(t, 3.0)

    def test_bf16_calibration_file(self):
        path = os.path.join("models", "calibration_bf16.json")
        self.assertTrue(os.path.exists(path), f"Missing {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        expected_models = ["efficientnet_v2_m", "densenet201", "convnext_small", "efficientnet_b4", "resnet50"]
        for m in expected_models:
            self.assertIn(m, data)
            t = data[m]
            self.assertIsInstance(t, (int, float))
            self.assertGreater(t, 0.5)
            self.assertLess(t, 3.0)

    def test_evaluation_meta_ensemble_fp16_and_bf16_exist(self):
        fp16_path = os.path.join("models", "evaluation_meta_ensemble.json")
        bf16_path = os.path.join("models", "evaluation_meta_ensemble_bf16.json")
        self.assertTrue(os.path.exists(fp16_path))
        self.assertTrue(os.path.exists(bf16_path))

        with open(fp16_path, "r", encoding="utf-8") as f:
            fp16_data = json.load(f)
        with open(bf16_path, "r", encoding="utf-8") as f:
            bf16_data = json.load(f)

        fp16_acc = fp16_data.get("accuracy") or fp16_data.get("test_accuracy")
        bf16_acc = bf16_data.get("test_accuracy") or bf16_data.get("accuracy")
        self.assertIsNotNone(fp16_acc)
        self.assertIsNotNone(bf16_acc)
        # FP16 precision achieves higher empirical accuracy on retinal ensemble
        self.assertGreaterEqual(fp16_acc, bf16_acc)


class TestPredictEndpointGuardrailRejection(unittest.TestCase):

    def setUp(self):
        self._orig_monolith = getattr(bm, "MONOLITHIC_MODEL", None)
        # Supply a dummy model so /predict does not fail with 503 Model Not Loaded
        bm.MONOLITHIC_MODEL = nn.Linear(10, 6)
        self.client = TestClient(bm.app, raise_server_exceptions=False)

    def tearDown(self):
        bm.MONOLITHIC_MODEL = self._orig_monolith

    def test_predict_rejects_non_fundus_photo_with_422(self):
        # Solid white square
        img = Image.new("RGB", (256, 256), color=(255, 255, 255))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()

        res = self.client.post(
            "/predict",
            files={"file": ("photo.jpg", raw, "image/jpeg")},
            data={
                "pain": "None", "vision": "No", "itch": "No",
                "halos": "No", "discharge": "None",
                "light_sens": "No", "floaters": "No", "duration": "Not Sure",
            },
        )
        self.assertEqual(res.status_code, 422)
        detail = res.json().get("detail", "")
        self.assertIn("retinal fundus", detail.lower())


if __name__ == "__main__":
    unittest.main()

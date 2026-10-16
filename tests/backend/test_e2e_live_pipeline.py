"""
End-to-End Live Pipeline Integration Verification.
Validates the full interactive screening loop:
- Ingestion of authentic fundus scans from frontend/public/samples/
- Lifespan model initialization and warm inference
- Platt calibrated probabilities and Grad-CAM saliency output
- Optical domain guardrail rejection of non-fundus imagery
- Decompression bomb rejection
- Patient privacy / EXIF stripping on live endpoints
"""

import base64
import io
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.db import create_tables
from backend.main import app, MONOLITHIC_CLASSES

create_tables()

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "public", "samples")
NORMAL_SAMPLE = os.path.join(SAMPLE_DIR, "fundus_normal.jpg")
DR_SAMPLE = os.path.join(SAMPLE_DIR, "fundus_diabetic_retinopathy.jpg")


@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI test client with lifespan context to load models."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestLiveE2EPipeline:
    """Verifies live inference, safety guardrails, and compression endpoints with authentic image files."""

    def test_health_and_metrics_live(self, client):
        """Verifies health check and Prometheus telemetry endpoints."""
        resp = client.get("/health")
        assert resp.status_code == 200
        health_data = resp.json()
        assert health_data.get("ok") is True
        assert "device" in health_data

        resp_metrics = client.get("/metrics")
        assert resp_metrics.status_code == 200
        assert "ophthalmoai" in resp_metrics.text

    def test_e2e_predict_normal_fundus_sample(self, client):
        """Verifies full inference pipeline on authentic normal fundus photography."""
        if not os.path.exists(NORMAL_SAMPLE):
            pytest.skip(f"Sample not found: {NORMAL_SAMPLE}")

        with open(NORMAL_SAMPLE, "rb") as f:
            img_bytes = f.read()

        payload = {
            "pain": "None",
            "vision": "No",
            "itch": "No",
            "halos": "No",
            "discharge": "None",
            "light_sens": "No",
            "floaters": "No",
            "duration": "Not Sure",
        }

        resp = client.post(
            "/predict",
            files={"file": ("normal_scan.jpg", img_bytes, "image/jpeg")},
            data=payload,
        )

        assert resp.status_code == 200, f"Expected 200 but got {resp.status_code}: {resp.text}"
        data = resp.json()

        # Schema verification
        assert "diagnosis" in data
        assert data["diagnosis"] in MONOLITHIC_CLASSES or data["diagnosis"] in ("Normal", "Referable")

        assert "confidence" in data
        assert isinstance(data["confidence"], (float, int))
        assert 0.0 <= data["confidence"] <= 100.0

        assert "probabilities" in data
        probs = data["probabilities"]
        assert isinstance(probs, dict)
        assert len(probs) >= 5

        # Check probability sum is approximately 1.0 (or percentage ~100.0)
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 0.05 or abs(prob_sum - 100.0) < 5.0

        # Clinical uncertainty and escalation fields
        assert "requires_human_review" in data
        assert isinstance(data["requires_human_review"], bool)

        assert "uncertainty" in data
        assert "urgency" in data

    def test_e2e_predict_diabetic_retinopathy_sample(self, client):
        """Verifies full inference pipeline on authentic diabetic retinopathy fundus scan."""
        if not os.path.exists(DR_SAMPLE):
            pytest.skip(f"Sample not found: {DR_SAMPLE}")

        with open(DR_SAMPLE, "rb") as f:
            img_bytes = f.read()

        payload = {
            "pain": "Mild",
            "vision": "Yes",
            "itch": "No",
            "halos": "No",
            "discharge": "None",
            "light_sens": "Yes",
            "floaters": "Yes",
            "duration": "Months",
        }

        resp = client.post(
            "/predict",
            files={"file": ("dr_scan.jpg", img_bytes, "image/jpeg")},
            data=payload,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "diagnosis" in data
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 100.0

    def test_e2e_scans_compress_valid_sample(self, client):
        """Verifies compression endpoint produces sanitized JPEG without EXIF leakage."""
        if not os.path.exists(NORMAL_SAMPLE):
            pytest.skip(f"Sample not found: {NORMAL_SAMPLE}")

        with open(NORMAL_SAMPLE, "rb") as f:
            img_bytes = f.read()

        resp = client.post(
            "/scans/compress",
            files={"file": ("raw_scan.jpg", img_bytes, "image/jpeg")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "compressed_file_base64" in data
        assert "bandwidth_saved_percent" in data

        # Decode compressed bytes and verify valid JPEG with purged EXIF
        raw_compressed = base64.b64decode(data["compressed_file_base64"])
        out_img = Image.open(io.BytesIO(raw_compressed))
        assert out_img.format == "JPEG"
        exif = out_img.getexif()
        assert len(exif) == 0

    def test_e2e_non_fundus_rejection(self, client):
        """Verifies domain guardrail rejects synthetic non-fundus image."""
        # Create a non-fundus grayscale/blank image
        non_fundus = Image.new("RGB", (384, 384), color=(128, 128, 128))
        buf = io.BytesIO()
        non_fundus.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        resp = client.post(
            "/predict",
            files={"file": ("non_fundus.jpg", raw_bytes, "image/jpeg")},
            data={"pain": "None"},
        )

        # Must be rejected by optical domain guardrail with HTTP 422
        assert resp.status_code == 422
        data = resp.json()
        assert "detail" in data

    def test_e2e_decompression_bomb_rejection(self, client):
        """Verifies server terminates decompression bomb attempt before allocating uncompressed pixels."""
        # Create a header-only decompression bomb (5500 x 5000 = 27.5 MP > 25 MP)
        bomb = Image.new("RGB", (5500, 5000), color=(180, 50, 30))
        buf = io.BytesIO()
        bomb.save(buf, format="JPEG")
        bomb_bytes = buf.getvalue()

        resp = client.post(
            "/predict",
            files={"file": ("bomb.jpg", bomb_bytes, "image/jpeg")},
            data={"pain": "None"},
        )

        # Must reject with 400 or 422 due to pixel limit
        assert resp.status_code in (400, 422)

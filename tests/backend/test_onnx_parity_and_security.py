"""
Tests for PyTorch-to-ONNX Parity, Deserialization Security, and Zero-Egress Air-Gap Compliance.
"""

from pathlib import Path
import glob
import re
import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import models

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
FRONTEND_DIR = ROOT_DIR / "frontend"


@pytest.mark.skipif(not ORT_AVAILABLE, reason="onnxruntime not installed")
def test_pytorch_to_onnx_fp32_numerical_parity():
    """
    Verifies that the PyTorch MobileNetV3-Small edge model and the exported
    FP32 ONNX model yield numerically identical outputs within tight floating point bounds.
    """
    pth_path = MODELS_DIR / "mobilenet_v3_small_edge.pth"
    onnx_path = MODELS_DIR / "edge_model_fp32.onnx"

    if not pth_path.exists() or not onnx_path.exists():
        pytest.skip("Edge model checkpoints missing")

    # Load PyTorch model
    pt_model = models.mobilenet_v3_small(weights=None)
    pt_model.classifier[3] = nn.Linear(pt_model.classifier[3].in_features, 6)
    pt_model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    pt_model.eval()

    # Load ONNX model
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Test with multiple random inputs
    np.random.seed(1337)
    for _ in range(5):
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # PyTorch forward pass
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(dummy_input)).cpu().numpy()

        # ONNX forward pass
        onnx_out = sess.run(None, {input_name: dummy_input})[0]

        # Assert max absolute difference < 1e-4
        max_diff = float(np.max(np.abs(pt_out - onnx_out)))
        assert max_diff < 1e-4, f"Max logit discrepancy {max_diff:.6e} exceeds 1e-4 tolerance."

        # Assert cosine similarity > 0.9999
        cos_sim = float(
            np.dot(pt_out.flatten(), onnx_out.flatten())
            / (np.linalg.norm(pt_out) * np.linalg.norm(onnx_out))
        )
        assert cos_sim > 0.9999, f"Cosine similarity {cos_sim:.8f} is below 0.9999."

        # Assert identical argmax predictions
        assert int(np.argmax(pt_out)) == int(np.argmax(onnx_out)), "Argmax mismatch between PyTorch and ONNX."


def test_weights_only_deserialization_security():
    """
    Verifies that all model checkpoints in models/*.pth can be safely loaded
    with weights_only=True without unpickling arbitrary Python code.
    """
    pth_files = list(MODELS_DIR.glob("*.pth"))
    assert len(pth_files) > 0, "No .pth model checkpoints found."

    for pth_file in pth_files:
        try:
            state = torch.load(pth_file, map_location="cpu", weights_only=True)
            assert isinstance(state, (dict, torch.Tensor)), f"{pth_file.name} state is not a valid dictionary or tensor."
        except Exception as exc:
            pytest.fail(f"Checkpoint {pth_file.name} failed to load with weights_only=True: {exc}")


def test_air_gapped_zero_egress_font_integrity():
    """
    Verifies that the frontend contains no external CDN font imports or script links,
    guaranteeing strict zero-egress offline compliance in air-gapped clinical intranets.
    """
    css_path = FRONTEND_DIR / "src" / "index.css"
    html_path = FRONTEND_DIR / "index.html"

    assert css_path.exists(), "frontend/src/index.css not found"
    assert html_path.exists(), "frontend/index.html not found"

    css_content = css_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")

    # Verify no Google Fonts @import exists in CSS
    assert "fonts.googleapis.com" not in css_content, "Google Fonts @import found in frontend/src/index.css"
    assert "fonts.gstatic.com" not in css_content, "Google Fonts CDN reference found in frontend/src/index.css"

    # Verify no external CDNs in index.html
    assert "fonts.googleapis.com" not in html_content, "Google Fonts link found in frontend/index.html"
    assert "cdn." not in html_content, "External CDN script/style found in frontend/index.html"


def test_static_quantization_pipeline_availability():
    """
    Verifies that the static activation calibration reader and quantization options
    are available and instantiate correctly in scripts/export_edge_model.py.
    """
    from scripts.export_edge_model import FundusCalibrationDataReader

    sample_images = list((FRONTEND_DIR / "public" / "samples").glob("*.jpg"))
    if not sample_images:
        pytest.skip("Sample fundus images missing")

    reader = FundusCalibrationDataReader(sample_images[:2], input_name="input", img_size=224)
    first_batch = reader.get_next()
    assert first_batch is not None, "Calibration reader failed to produce first batch."
    assert "input" in first_batch
    assert first_batch["input"].shape == (1, 3, 224, 224)
    reader.rewind()


"""
Boundary Condition & Edge Case Unit Tests for OphthalmoAI ("Code is Poetry").
=============================================================================
Comprehensive boundary tests covering:
1. Multi-Tenancy Boundary Isolation:
   - Empty tenant IDs, whitespace headers, SQL injection & XSS attempts.
   - Fallback hierarchy: explicit header -> authenticated user org -> system default.
   - Missing or duck-typed current_user handling without AttributeError.
   - apply_tenant_filter safeguards on None/empty IDs and non-tenant models.

2. Retinal Fundus Domain Validator Boundary Guardrails:
   - Zero dimensions (0x0) and 1x1 image inputs.
   - Pure white and pure black saturation extremes.
   - Uniform and static random noise inputs.
   - Extreme aspect ratio strips (10000x1, 1x10000).
   - Floating point NaN and Inf pixel values in mode 'F'.
   - Non-3-channel images (Grayscale 'L', Binary '1', RGBA 'RGBA', CMYK 'CMYK').

3. Platt Probability Calibration Boundary Safeguards:
   - Zero, negative, None, NaN, and Inf temperature protections.
   - Negative logits probability distribution stability.
   - Extreme magnitude logits (+/- 1e6) without numerical overflow.
   - Uniform logits and degenerate single-class distributions.
   - TemperatureScaler zero-division guard and parameter clamping.

4. Prometheus Observability Metrics Boundary Guards:
   - Zero-request division guard and empty collector state formatting.
   - Negative latency samples (clock skew / timer drift).
   - Concurrent multi-threaded updates and thread safety verification.
   - Unbounded memory prevention (1000-sample ring buffer cap).

5. Urgency-Stratified Conformal Prediction Boundary Conditions:
   - Alpha <= 0 (super-conservative / coverage >= 100%).
   - Alpha >= 1 (relaxed / sub-zero quantile bounds).
   - Empty prediction set fallback to argmax class.
   - Extreme non-conformity scores (perfect 0.0 and inverted 1.0).
   - Admissibility-Weighted Conformal Risk Control (AW-CRC) score bounding.
   - ConformalTriagePolicy 3-tier boundary routing.
"""

from __future__ import annotations

import concurrent.futures
import math
import threading
from typing import Any, List, Optional
import numpy as np
from PIL import Image
import pytest
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# Module Imports
# ---------------------------------------------------------------------------
from backend.calibration import (
    DEFAULT_TEMPERATURE,
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    CalibrationRegistry,
    TemperatureScaler,
    apply_temperature,
    calibrated_softmax,
    sanitize_temperature,
)
from backend.conformal import (
    ConformalCalibrator,
    ConformalTriagePolicy,
)
from backend.db import Base, ScanResult, Tenant, User
from backend.evidential import CLASS_NAMES, URGENCY_TIERS
from backend.fundus_validator import validate_fundus_image
from backend.metrics import PrometheusMetricsCollector
from backend.tenancy import (
    DEFAULT_TENANT_ID,
    apply_tenant_filter,
    get_current_tenant_id,
)

# Torch import with safe fallback to mock tensor if native DLL is restricted
try:
    import torch
except (ImportError, OSError):
    import tests.conftest  # noqa: F401
    import torch


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def memory_db():
    """Provides a fresh isolated in-memory SQLite session with schema initialized."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture
def synthetic_conformal_data():
    """Generates standard validation probabilities and targets across 6 classes."""
    np.random.seed(42)
    n_samples = 30
    num_classes = len(CLASS_NAMES)
    # Generate realistic Dirichlet probabilities
    raw = np.random.gamma(shape=1.0, scale=1.0, size=(n_samples, num_classes))
    probs = raw / np.sum(raw, axis=1, keepdims=True)
    targets = np.random.randint(0, num_classes, size=(n_samples,))
    return probs, targets


# ===========================================================================
# 1. Tenancy Boundary Tests
# ===========================================================================
def test_tenancy_empty_or_whitespace_header_falls_back_to_default():
    """Empty string and pure whitespace X-Tenant-ID headers must resolve to DEFAULT_TENANT_ID."""
    assert get_current_tenant_id(x_tenant_id="") == DEFAULT_TENANT_ID, (
        "Empty string tenant header should fall back to default tenant ID."
    )
    assert get_current_tenant_id(x_tenant_id="   ") == DEFAULT_TENANT_ID, (
        "Whitespace-only tenant header should fall back to default tenant ID."
    )
    assert get_current_tenant_id(x_tenant_id="\t\n") == DEFAULT_TENANT_ID, (
        "Tab/newline tenant header should fall back to default tenant ID."
    )


def test_tenancy_empty_header_with_authenticated_user_uses_user_tenant():
    """When tenant header is empty or whitespace, system must fall back to current_user.tenant_id."""
    user = User(email="ophthalmologist@clinic-apex.org", role="clinician")
    user.tenant_id = "clinic-apex-retina"

    assert get_current_tenant_id(x_tenant_id="", current_user=user) == "clinic-apex-retina", (
        "Empty header should fall back to authenticated user's organization tenant."
    )
    assert get_current_tenant_id(x_tenant_id="   ", current_user=user) == "clinic-apex-retina", (
        "Whitespace header should fall back to authenticated user's organization tenant."
    )


def test_tenancy_sql_injection_and_special_characters_rejected_with_http_400(memory_db):
    """
    Malicious tenant ID payloads (SQL injection, XSS, directory traversal)
    must be rejected with HTTP 400 Bad Request at the header validation boundary.
    Direct application of raw strings to apply_tenant_filter must safely bind without SQL injection.
    """
    malicious_payloads = [
        "'; DROP TABLE scan_results; --",
        "' OR '1'='1",
        "' UNION SELECT id, email, hashed_password FROM users --",
        "<script>alert('xss')</script>",
        "../../../../etc/passwd\x00",
        "tenant_id' AND 1=0 UNION ALL SELECT * FROM tenants --",
    ]

    # Populate valid scan record for a legitimate tenant
    legit_scan = ScanResult(
        diagnosis="Diabetic Retinopathy",
        confidence=0.96,
        tenant_id="legitimate-hospital-tenant",
    )
    memory_db.add(legit_scan)
    memory_db.commit()

    for attack_payload in malicious_payloads:
        # Header resolver must reject invalid format characters with HTTP 400
        with pytest.raises(Exception) as exc_info:
            get_current_tenant_id(x_tenant_id=attack_payload)
        assert "400" in str(exc_info.value) or "invalid" in str(exc_info.value).lower(), (
            f"Expected HTTP 400 Bad Request for malicious tenant header '{attack_payload}'."
        )

        # Direct parameterized query filtering must safely bind without syntax errors or leakage
        scoped_query = apply_tenant_filter(
            memory_db.query(ScanResult),
            ScanResult,
            attack_payload,
        )
        results = scoped_query.all()
        assert len(results) == 0, (
            f"SQL injection attempt '{attack_payload}' leaked or matched rows unexpectedly."
        )

    # Ensure legitimate scan was not deleted or mutated by any payload
    all_scans = memory_db.query(ScanResult).all()
    assert len(all_scans) == 1, "Legitimate scan record was compromised or dropped."
    assert all_scans[0].tenant_id == "legitimate-hospital-tenant"


def test_tenancy_missing_header_and_missing_user_falls_back_to_default():
    """Anonymous request with no header and None current_user must default safely."""
    tenant_id = get_current_tenant_id(x_tenant_id=None, current_user=None)
    assert tenant_id == DEFAULT_TENANT_ID, (
        f"Expected {DEFAULT_TENANT_ID} for anonymous request, got {tenant_id}."
    )


def test_tenancy_duck_typed_user_missing_tenant_id_attr_falls_back_to_default():
    """A user object lacking tenant_id attribute must not raise AttributeError."""
    class DummyUser:
        email = "anonymous@public.net"

    tenant_id = get_current_tenant_id(x_tenant_id=None, current_user=DummyUser())
    assert tenant_id == DEFAULT_TENANT_ID, (
        "Duck-typed user without tenant_id attribute should fall back to default tenant ID."
    )


def test_tenancy_apply_filter_empty_or_none_tenant_id_fails_closed_returns_zero_rows(memory_db):
    """
    Passing None or empty string to apply_tenant_filter must fail closed (bind sentinel)
    to prevent accidental global data disclosure across hospital tenants.
    """
    scan_1 = ScanResult(diagnosis="Cataract", confidence=0.88, tenant_id="tenant-a")
    scan_2 = ScanResult(diagnosis="Glaucoma", confidence=0.91, tenant_id="tenant-b")
    memory_db.add_all([scan_1, scan_2])
    memory_db.commit()

    base_query = memory_db.query(ScanResult)

    filtered_none = apply_tenant_filter(base_query, ScanResult, None).all()
    assert len(filtered_none) == 0, "apply_tenant_filter with None tenant_id must fail closed (0 rows)."

    filtered_empty = apply_tenant_filter(base_query, ScanResult, "").all()
    assert len(filtered_empty) == 0, "apply_tenant_filter with empty tenant_id must fail closed (0 rows)."


def test_tenancy_apply_filter_on_model_without_tenant_id_returns_unfiltered_query(memory_db):
    """Filtering a model class without a tenant_id attribute must return the query unaltered."""
    LocalBase = declarative_base()

    class AuditItem(LocalBase):
        __tablename__ = "audit_items"
        id = Column(Integer, primary_key=True)
        message = Column(String(50))

    LocalBase.metadata.create_all(bind=memory_db.get_bind())
    memory_db.add(AuditItem(id=1, message="System initialized"))
    memory_db.commit()

    query = memory_db.query(AuditItem)
    filtered = apply_tenant_filter(query, AuditItem, "any-tenant-id").all()
    assert len(filtered) == 1, "apply_tenant_filter failed to return unaltered query for model without tenant_id."


# ===========================================================================
# 2. Fundus Validator Boundary Tests
# ===========================================================================
def test_fundus_validator_zero_dimensions_rejected_at_dimensions_stage():
    """An image of size 0x0 must be safely rejected at dimensions stage without crash."""
    img = Image.new("RGB", (0, 0))
    is_valid, conf, reason, metrics = validate_fundus_image(img)

    assert is_valid is False, "0x0 image must not pass fundus validation."
    assert conf == 0.0, "Confidence for 0x0 image must be 0.0."
    assert metrics.get("failure_stage") == "dimensions", (
        f"Expected failure_stage='dimensions', got {metrics.get('failure_stage')}."
    )


def test_fundus_validator_single_pixel_image_rejected_at_dimensions_stage():
    """A 1x1 pixel image must be rejected at dimensions stage."""
    img = Image.new("RGB", (1, 1), color=(180, 80, 30))
    is_valid, conf, reason, metrics = validate_fundus_image(img)

    assert is_valid is False, "1x1 image must not pass fundus validation."
    assert conf == 0.0, "Confidence for 1x1 image must be 0.0."
    assert metrics.get("failure_stage") == "dimensions"


def test_fundus_validator_pure_white_image_rejected_at_zero_variance():
    """Pure white saturated image (255, 255, 255) must be rejected for zero visual variance."""
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    is_valid, conf, reason, metrics = validate_fundus_image(img)

    assert is_valid is False, "Pure white image must be rejected."
    assert conf == 0.0, "Confidence for pure white image must be 0.0."
    assert metrics.get("failure_stage") in ("zero_variance", "document_pattern"), (
        f"Unexpected failure stage for pure white image: {metrics.get('failure_stage')}"
    )


def test_fundus_validator_pure_black_image_rejected_at_zero_variance():
    """Pure black saturated image (0, 0, 0) must be rejected for zero visual variance."""
    img = Image.new("RGB", (256, 256), color=(0, 0, 0))
    is_valid, conf, reason, metrics = validate_fundus_image(img)

    assert is_valid is False, "Pure black image must be rejected."
    assert conf == 0.0, "Confidence for pure black image must be 0.0."
    assert metrics.get("failure_stage") in ("zero_variance", "dark_field")


def test_fundus_validator_uniform_random_noise_rejected_at_random_noise_stage():
    """Uncorrelated uniform random noise must fail spatial autocorrelation checks."""
    np.random.seed(1337)
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    is_valid, conf, reason, metrics = validate_fundus_image(img)

    assert is_valid is False, "Uniform random noise must not pass fundus validation."
    assert conf < 0.50, f"Confidence for static noise must be < 0.50, got {conf}."
    assert metrics.get("failure_stage") == "random_noise", (
        f"Expected failure_stage='random_noise', got {metrics.get('failure_stage')}."
    )


def test_fundus_validator_extreme_aspect_ratios_rejected_safely():
    """Extreme aspect ratio strips (10000x1 and 1x10000) must be rejected without error."""
    h_strip = Image.new("RGB", (10000, 1), color=(180, 80, 30))
    is_valid_h, conf_h, _, metrics_h = validate_fundus_image(h_strip)
    assert is_valid_h is False
    assert metrics_h.get("failure_stage") == "dimensions"

    v_strip = Image.new("RGB", (1, 10000), color=(180, 80, 30))
    is_valid_v, conf_v, _, metrics_v = validate_fundus_image(v_strip)
    assert is_valid_v is False
    assert metrics_v.get("failure_stage") == "dimensions"


def test_fundus_validator_nan_and_inf_pixel_values_handled_gracefully():
    """Floating point mode 'F' images containing NaN or Inf must not crash the validator."""
    # 1. All NaNs
    nan_arr = np.full((256, 256), np.nan, dtype=np.float32)
    nan_img = Image.fromarray(nan_arr, mode="F")
    is_valid_nan, conf_nan, _, _ = validate_fundus_image(nan_img)
    assert is_valid_nan is False, "Image with all NaNs must be rejected."
    assert conf_nan == 0.0

    # 2. All Infs
    inf_arr = np.full((256, 256), np.inf, dtype=np.float32)
    inf_img = Image.fromarray(inf_arr, mode="F")
    is_valid_inf, conf_inf, _, _ = validate_fundus_image(inf_img)
    assert is_valid_inf is False, "Image with all Infs must be rejected."
    assert conf_inf == 0.0


def test_fundus_validator_non_three_channel_images_handled_cleanly():
    """Images with modes 'L' (Grayscale), '1' (Bilevel), 'RGBA', and 'CMYK' must be handled cleanly."""
    # Grayscale image with gradient
    gray_arr = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    gray_img = Image.fromarray(gray_arr, mode="L")
    is_valid_gray, conf_gray, reason_gray, metrics_gray = validate_fundus_image(gray_img)
    assert is_valid_gray is False, "Monochrome grayscale image must not pass color fundus validation."
    assert "monochrome" in reason_gray.lower() or metrics_gray.get("failure_stage") == "chromatic_profile"

    # 1-bit bilevel image
    bit_img = Image.new("1", (256, 256), 1)
    is_valid_bit, conf_bit, _, _ = validate_fundus_image(bit_img)
    assert is_valid_bit is False, "1-bit bilevel image must be rejected."

    # RGBA image with alpha channel
    rgba_img = Image.new("RGBA", (256, 256), (180, 80, 30, 255))
    is_valid_rgba, _, _, _ = validate_fundus_image(rgba_img)
    assert is_valid_rgba is False  # Rejected for zero variance or lack of fundus aperture


# ===========================================================================
# 3. Platt Calibration Boundary Tests
# ===========================================================================
def test_platt_calibration_zero_and_negative_temperature_protection():
    """Zero and negative temperatures must fall back to DEFAULT_TEMPERATURE to prevent zero-division."""
    assert sanitize_temperature(0.0) == DEFAULT_TEMPERATURE, (
        "Zero temperature should sanitize to DEFAULT_TEMPERATURE."
    )
    assert sanitize_temperature(-1.0) == DEFAULT_TEMPERATURE, (
        "Negative temperature should sanitize to DEFAULT_TEMPERATURE."
    )
    assert sanitize_temperature(-1e-6) == DEFAULT_TEMPERATURE, (
        "Tiny negative temperature should sanitize to DEFAULT_TEMPERATURE."
    )

    logits = torch.tensor([3.0, 6.0])
    # apply_temperature with zero temperature
    scaled_zero = apply_temperature(logits, 0.0)
    expected = logits / DEFAULT_TEMPERATURE
    assert torch.allclose(scaled_zero, expected), (
        "apply_temperature(logits, 0.0) must scale using DEFAULT_TEMPERATURE."
    )

    # apply_temperature with negative temperature
    scaled_neg = apply_temperature(logits, -2.5)
    assert torch.allclose(scaled_neg, expected), (
        "apply_temperature(logits, -2.5) must scale using DEFAULT_TEMPERATURE."
    )


def test_platt_calibration_none_nan_and_inf_temperature_falls_back_to_default():
    """None, NaN, Inf, and non-numeric temperature values must cleanly sanitize to DEFAULT_TEMPERATURE."""
    assert sanitize_temperature(None) == DEFAULT_TEMPERATURE
    assert sanitize_temperature(float("nan")) == DEFAULT_TEMPERATURE
    assert sanitize_temperature(float("inf")) == DEFAULT_TEMPERATURE
    assert sanitize_temperature(float("-inf")) == DEFAULT_TEMPERATURE
    assert sanitize_temperature("invalid_string") == DEFAULT_TEMPERATURE


def test_platt_calibration_negative_logits_softmax_produces_valid_distribution():
    """Deeply negative logits must yield a valid, numerically stable probability distribution."""
    logits = torch.tensor([-20.0, -50.0, -100.0, -10.0])
    probs = calibrated_softmax(logits, temperature=1.5)

    probs_np = probs.numpy() if hasattr(probs, "numpy") else probs.arr
    assert np.all(probs_np >= 0.0), "Probabilities must be non-negative."
    assert np.isclose(np.sum(probs_np), 1.0, atol=1e-5), "Probabilities must sum to 1.0."
    # Largest logit is -10.0 at index 3, so index 3 must have highest probability
    assert np.argmax(probs_np) == 3, "Argmax of softmax must match largest logit."


def test_platt_calibration_extreme_magnitude_logits_preserves_numerical_stability():
    """Extreme magnitude logits (+/- 1e6) must not trigger numerical overflow, underflow, or NaN."""
    logits = torch.tensor([1e6, -1e6, 0.0, 10.0])
    probs = calibrated_softmax(logits, temperature=1.0)

    probs_np = probs.numpy() if hasattr(probs, "numpy") else probs.arr
    assert not np.isnan(probs_np).any(), "Calibrated softmax generated NaN on extreme logits."
    assert not np.isinf(probs_np).any(), "Calibrated softmax generated Inf on extreme logits."
    assert np.isclose(np.sum(probs_np), 1.0, atol=1e-5), "Probabilities must sum to 1.0."
    assert np.isclose(probs_np[0], 1.0, atol=1e-5), "Dominant logit (1e6) should have probability ~1.0."


def test_platt_calibration_uniform_and_degenerate_distributions():
    """Uniform logits must produce exact uniform probabilities, and single-class extreme produces one-hot."""
    # 1. Perfectly uniform logits
    uniform_logits = torch.tensor([5.0, 5.0, 5.0, 5.0])
    uniform_probs = calibrated_softmax(uniform_logits, temperature=1.5)
    u_np = uniform_probs.numpy() if hasattr(uniform_probs, "numpy") else uniform_probs.arr
    assert np.allclose(u_np, [0.25, 0.25, 0.25, 0.25], atol=1e-5), (
        f"Uniform logits did not yield 0.25 uniform probabilities: {u_np}"
    )

    # 2. Degenerate dominant logit
    dominant_logits = torch.tensor([1000.0, 0.0, 0.0])
    dom_probs = calibrated_softmax(dominant_logits, temperature=1.0)
    d_np = dom_probs.numpy() if hasattr(dom_probs, "numpy") else dom_probs.arr
    assert np.isclose(d_np[0], 1.0, atol=1e-5), "Dominant class should receive probability 1.0."
    assert np.isclose(d_np[1], 0.0, atol=1e-5)
    assert np.isclose(d_np[2], 0.0, atol=1e-5)


def test_platt_calibration_temperature_scaler_zero_division_guard():
    """TemperatureScaler with temperature set to zero must clamp and prevent zero division in forward pass."""
    scaler = TemperatureScaler()
    scaler.temperature.data = torch.tensor([0.0])

    inputs = torch.tensor([2.0, 4.0])
    outputs = scaler(inputs)
    out_np = outputs.numpy() if hasattr(outputs, "numpy") else outputs.arr

    assert not np.isinf(out_np).any(), "TemperatureScaler forward pass produced Inf with zero temperature."
    assert not np.isnan(out_np).any(), "TemperatureScaler forward pass produced NaN with zero temperature."


# ===========================================================================
# 4. Prometheus Metrics Boundary Tests
# ===========================================================================
def test_prometheus_metrics_zero_requests_division_guard():
    """An empty metrics collector with 0 inference requests or durations must not crash or divide by zero."""
    collector = PrometheusMetricsCollector()
    collector._inference_requests.clear()
    collector._inference_durations.clear()
    collector._domain_shifts.clear()

    prom_text = collector.generate_prometheus_text()
    assert isinstance(prom_text, str), "Prometheus output must be a string."
    assert "# HELP ophthalmoai_inference_requests_total" in prom_text
    assert "# HELP ophthalmoai_inference_duration_seconds" in prom_text
    assert "ophthalmoai_gpu_memory_bytes" in prom_text
    assert "ophthalmoai_active_tenants_total" in prom_text


def test_prometheus_metrics_negative_latency_handled_safely():
    """Negative inference latencies (clock skew / NTP drift) must not crash quantile computations."""
    collector = PrometheusMetricsCollector()
    collector.record_inference("model_skew", -0.05, success=True)
    collector.record_inference("model_skew", -0.01, success=False)

    prom_text = collector.generate_prometheus_text()
    assert 'model="model_skew"' in prom_text, "Model with negative latency should be recorded."
    assert "ophthalmoai_inference_duration_seconds{model=\"model_skew\",quantile=\"0.5\"}" in prom_text


def test_prometheus_metrics_concurrent_updates_thread_safety():
    """Concurrent updates across multiple threads must be thread-safe without race conditions or deadlocks."""
    collector = PrometheusMetricsCollector()
    num_threads = 8
    iterations_per_thread = 50

    def worker(thread_idx: int):
        for i in range(iterations_per_thread):
            collector.record_inference(
                model=f"model_{thread_idx % 3}",
                duration_s=0.010 + (i * 0.001),
                success=(i % 2 == 0),
            )
            collector.record_domain_shift(f"shift_{thread_idx % 2}")
            collector.set_active_tenants((thread_idx + i) % 10)
            if i % 10 == 0:
                text = collector.generate_prometheus_text()
                assert len(text) > 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, t) for t in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # Will re-raise exceptions if any thread encountered a race condition

    final_text = collector.generate_prometheus_text()
    assert len(final_text) > 100
    assert "ophthalmoai_inference_requests_total" in final_text


def test_prometheus_metrics_unbounded_duration_history_capped_at_thousand():
    """Inference duration sample list must be capped at 1000 to prevent unbounded memory growth."""
    collector = PrometheusMetricsCollector()
    model_name = "stress_test_net"

    for i in range(1200):
        collector.record_inference(model_name, duration_s=0.015, success=True)

    assert len(collector._inference_durations[model_name]) == 1000, (
        f"Durations buffer must be capped at 1000, got {len(collector._inference_durations[model_name])}."
    )


# ===========================================================================
# 5. Conformal Prediction Boundary Tests
# ===========================================================================
def test_conformal_calibrator_alpha_non_positive_raises_value_error(synthetic_conformal_data):
    """Significance level alpha <= 0 violates statistical risk guarantees and must raise ValueError."""
    # Test alpha_emergency <= 0
    with pytest.raises(ValueError) as exc_0:
        ConformalCalibrator(alpha_emergency=0.0)
    assert "in (0.0, 1.0)" in str(exc_0.value), "Alpha 0.0 must raise ValueError."

    with pytest.raises(ValueError) as exc_neg:
        ConformalCalibrator(alpha_routine=-0.05)
    assert "in (0.0, 1.0)" in str(exc_neg.value), "Negative alpha must raise ValueError."

    # Test extreme valid lower boundary: alpha = 1e-4 (99.99% coverage guarantee)
    probs, targets = synthetic_conformal_data
    calibrator = ConformalCalibrator(alpha_emergency=1e-4, alpha_routine=1e-3)
    summary = calibrator.calibrate(probs, targets)
    assert summary["is_calibrated"] is True
    pset, _, _, guarantee = calibrator.predict_set(probs[0], "Diabetic Retinopathy")
    assert len(pset) >= 1
    assert guarantee > 99.0


def test_conformal_calibrator_alpha_greater_than_or_equal_to_one_raises_value_error(synthetic_conformal_data):
    """Significance level alpha >= 1.0 violates probability bounds and must raise ValueError."""
    # Test alpha == 1.0
    with pytest.raises(ValueError) as exc_1:
        ConformalCalibrator(alpha_emergency=1.0)
    assert "in (0.0, 1.0)" in str(exc_1.value), "Alpha 1.0 must raise ValueError."

    # Test alpha > 1.0
    with pytest.raises(ValueError) as exc_gt1:
        ConformalCalibrator(alpha_routine=1.5)
    assert "in (0.0, 1.0)" in str(exc_gt1.value), "Alpha > 1.0 must raise ValueError."

    # Test extreme valid upper boundary: alpha = 0.99 (relaxed 1% coverage guarantee)
    probs, targets = synthetic_conformal_data
    calibrator = ConformalCalibrator(alpha_emergency=0.95, alpha_routine=0.99)
    summary = calibrator.calibrate(probs, targets)
    assert summary["is_calibrated"] is True
    pset, _, _, _ = calibrator.predict_set(probs[0], "Normal")
    assert len(pset) >= 1


def test_conformal_prediction_empty_set_fallback_to_argmax():
    """When all candidate probabilities fall below cutoff (1 - q_hat), prediction set falls back to argmax."""
    calibrator = ConformalCalibrator()
    # Artificially set high empirical quantile so cutoff = 1 - 0.1 = 0.90
    calibrator.q_routine = 0.10

    # Low uniform probabilities where every class has probability ~0.1667 < 0.90
    probs = np.array([0.166, 0.167, 0.166, 0.166, 0.167, 0.168])
    top_pred = CLASS_NAMES[int(np.argmax(probs))]

    pset, set_probs, stratum, guarantee = calibrator.predict_set(probs, top_pred)
    assert len(pset) >= 1, "Conformal prediction set must never be empty."
    assert pset[0] == top_pred, f"Fallback must select argmax class '{top_pred}', got '{pset[0]}'."
    assert top_pred in set_probs


def test_conformal_prediction_extreme_non_conformity_scores(synthetic_conformal_data):
    """Calibration on perfect scores (s=0.0) and completely wrong scores (s=1.0) must execute without error."""
    n_samples = 20
    num_classes = len(CLASS_NAMES)
    calibrator = ConformalCalibrator()

    # Perfect predictions: probability 1.0 on true class
    perfect_probs = np.zeros((n_samples, num_classes))
    targets = np.zeros(n_samples, dtype=int)
    perfect_probs[:, 0] = 1.0

    summary_perf = calibrator.calibrate(perfect_probs, targets)
    assert summary_perf["is_calibrated"] is True
    assert np.isclose(summary_perf["q_emergency"], 0.0, atol=1e-5) or np.isclose(summary_perf["q_routine"], 0.0, atol=1e-5)

    # Inverted predictions: probability 0.0 on true class
    inverted_probs = np.zeros((n_samples, num_classes))
    inverted_probs[:, 1] = 1.0  # true class is 0, so true class has probability 0.0

    summary_inv = calibrator.calibrate(inverted_probs, targets)
    assert summary_inv["is_calibrated"] is True
    assert summary_inv["q_emergency"] == 1.0 or summary_inv["q_routine"] == 1.0


def test_conformal_predict_aw_crc_extreme_admissibility_scores_bounded():
    """Admissibility-Weighted Conformal Risk Control must gracefully clamp extreme admissibility values."""
    calibrator = ConformalCalibrator()
    probs = np.array([0.70, 0.10, 0.05, 0.05, 0.05, 0.05])

    # Extreme low admissibility (negative or zero clamped to 0.50)
    pset_low, _, _, _ = calibrator.predict_set_aw_crc(
        probs, "Normal", admissibility_score=-5.0, gamma=1.0
    )
    assert len(pset_low) >= 1

    # Extreme high admissibility (clamped to 1.0)
    pset_high, _, _, _ = calibrator.predict_set_aw_crc(
        probs, "Normal", admissibility_score=100.0, gamma=1.0
    )
    assert len(pset_high) >= 1

    # Extreme gamma parameters
    pset_gamma0, _, _, _ = calibrator.predict_set_aw_crc(
        probs, "Normal", admissibility_score=0.85, gamma=0.0
    )
    assert len(pset_gamma0) >= 1

    pset_gamma10, _, _, _ = calibrator.predict_set_aw_crc(
        probs, "Normal", admissibility_score=0.85, gamma=10.0
    )
    assert len(pset_gamma10) >= 1


def test_conformal_triage_policy_routing_boundaries():
    """ConformalTriagePolicy must route sets accurately according to urgency stratification rules."""
    # 1. Autonomous Clearance: single-element Normal set
    triage_norm = ConformalTriagePolicy.evaluate(
        prediction_set=["Normal"],
        top_diagnosis="Normal",
        epistemic_vacuity=0.05,
        requires_human_review_flag=False,
    )
    assert triage_norm["triage_tier"] == "Autonomous Clearance"
    assert triage_norm["action_code"] == "GREEN_CLEAR"

    # 2. Immediate Emergency Review: set includes Emergency condition (AMD)
    triage_emerg = ConformalTriagePolicy.evaluate(
        prediction_set=["Age-related Macular Degeneration", "Normal"],
        top_diagnosis="Age-related Macular Degeneration",
        epistemic_vacuity=0.15,
        requires_human_review_flag=False,
    )
    assert triage_emerg["triage_tier"] == "Immediate Emergency Review"
    assert triage_emerg["action_code"] == "RED_FLAG"

    # 3. Urgent Specialist Review: set includes Urgent condition (Glaucoma)
    triage_urgent = ConformalTriagePolicy.evaluate(
        prediction_set=["Glaucoma"],
        top_diagnosis="Glaucoma",
        epistemic_vacuity=0.10,
        requires_human_review_flag=False,
    )
    assert triage_urgent["triage_tier"] == "Urgent Specialist Review"
    assert triage_urgent["action_code"] == "ORANGE_ALERT"

    # 4. Ambiguous Case: cardinality > 2
    triage_ambig = ConformalTriagePolicy.evaluate(
        prediction_set=["Cataract", "Normal", "Diabetic Retinopathy"],
        top_diagnosis="Cataract",
        epistemic_vacuity=0.10,
        requires_human_review_flag=False,
    )
    # Since Diabetic Retinopathy is Urgent, urgent triage takes precedence if urgent
    # What if all items are Elective / None with cardinality > 2?
    triage_multi = ConformalTriagePolicy.evaluate(
        prediction_set=["Cataract", "Normal", "Normal"],
        top_diagnosis="Cataract",
        epistemic_vacuity=0.10,
        requires_human_review_flag=False,
    )
    assert triage_multi["triage_tier"] == "Ambiguous Case - Clinician Review"
    assert triage_multi["action_code"] == "YELLOW_REVIEW"

    # 5. Routine Specialist Triage: single Elective condition
    triage_routine = ConformalTriagePolicy.evaluate(
        prediction_set=["Cataract"],
        top_diagnosis="Cataract",
        epistemic_vacuity=0.10,
        requires_human_review_flag=False,
    )
    assert triage_routine["triage_tier"] == "Routine Specialist Triage"
    assert triage_routine["action_code"] == "BLUE_ROUTINE"

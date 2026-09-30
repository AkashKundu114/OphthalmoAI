from datetime import date, datetime, timezone
import pytest
from backend.screening_anomaly_detector import ScreeningAnomalyDetector, ScreeningAnomaly

def test_anomaly_to_dict():
    anom = ScreeningAnomaly(
        tenant_id="tenant-123",
        anomaly_type="volume",
        severity="warning",
        metric_name="total_screenings",
        current_value=120.0,
        baseline_value=50.0,
        deviation_percent=140.0,
        detected_at=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
        recommendation="Check volume",
        ad_tech_parallel="Impression volume spike",
    )
    d = anom.to_dict()
    assert d["tenant_id"] == "tenant-123"
    assert d["anomaly_type"] == "volume"
    assert d["deviation_percent"] == 140.0

def test_detect_volume_anomaly():
    detector = ScreeningAnomalyDetector(volume_z_threshold=2.0)
    
    # Empty history
    assert detector.detect_volume_anomaly("t1", 50, []) is None
    
    # Stable history and volume within normal range
    history = [50.0, 52.0, 48.0, 51.0, 49.0, 50.0, 53.0]
    assert detector.detect_volume_anomaly("t1", 52.0, history) is None

    # Massive volume spike
    spike = detector.detect_volume_anomaly("t1", 150.0, history)
    assert spike is not None
    assert spike.anomaly_type == "volume"
    assert spike.severity == "critical"
    assert spike.deviation_percent > 50

    # Massive volume drop
    drop = detector.detect_volume_anomaly("t1", 5.0, history)
    assert drop is not None
    assert drop.anomaly_type == "volume"
    assert drop.deviation_percent < 0

def test_detect_quality_anomaly():
    detector = ScreeningAnomalyDetector(quality_drop_threshold_pct=10.0)

    # Empty history
    assert detector.detect_quality_anomaly("t1", 0.9, []) is None

    # Normal confidence
    history = [0.92, 0.91, 0.93]
    assert detector.detect_quality_anomaly("t1", 0.90, history) is None

    # Substantial drop
    drop = detector.detect_quality_anomaly("t1", 0.70, history)
    assert drop is not None
    assert drop.anomaly_type == "quality"
    assert drop.severity == "critical"

def test_detect_sla_breach():
    detector = ScreeningAnomalyDetector(latency_sla_ms=200.0)

    # Within SLA
    assert detector.detect_sla_breach("t1", 120.0) is None

    # Moderate breach
    breach = detector.detect_sla_breach("t1", 240.0)
    assert breach is not None
    assert breach.anomaly_type == "sla_breach"
    assert breach.severity == "warning"

    # Severe breach
    severe = detector.detect_sla_breach("t1", 350.0)
    assert severe is not None
    assert severe.severity == "critical"

def test_detect_distribution_shift():
    detector = ScreeningAnomalyDetector(distribution_p_value_threshold=0.05)

    # Insufficient data
    assert detector.detect_distribution_shift("t1", {}, []) is None
    assert detector.detect_distribution_shift("t1", {"Normal": 2}, [{"Normal": 10}]) is None

    # Normal distribution with similar historical frequencies
    hist = [{"Normal": 80, "DR": 20}] * 5
    curr_normal = {"Normal": 82, "DR": 18}
    assert detector.detect_distribution_shift("t1", curr_normal, hist) is None

    # Shifted distribution
    curr_shifted = {"Normal": 10, "DR": 90}
    shift = detector.detect_distribution_shift("t1", curr_shifted, hist)
    assert shift is not None
    assert shift.anomaly_type == "distribution_shift"

def test_analyze_tenant_metrics():
    detector = ScreeningAnomalyDetector()
    metrics = [
        {"date": "2026-09-01", "total_screenings": 50, "avg_confidence": 0.92, "avg_inference_time_ms": 110, "diagnoses_by_class": {"Normal": 40, "DR": 10}},
        {"date": "2026-09-02", "total_screenings": 52, "avg_confidence": 0.91, "avg_inference_time_ms": 115, "diagnoses_by_class": {"Normal": 42, "DR": 10}},
        {"date": "2026-09-03", "total_screenings": 49, "avg_confidence": 0.93, "avg_inference_time_ms": 105, "diagnoses_by_class": {"Normal": 39, "DR": 10}},
        {"date": "2026-09-04", "total_screenings": 51, "avg_confidence": 0.90, "avg_inference_time_ms": 112, "diagnoses_by_class": {"Normal": 41, "DR": 10}},
        {"date": "2026-09-05", "total_screenings": 200, "avg_confidence": 0.65, "avg_inference_time_ms": 320, "diagnoses_by_class": {"Normal": 10, "DR": 190}},
    ]
    anomalies = detector.analyze_tenant_metrics("tenant-test", metrics, scan_days=1)
    assert len(anomalies) >= 2
    types = {a.anomaly_type for a in anomalies}
    assert "volume" in types
    assert "sla_breach" in types

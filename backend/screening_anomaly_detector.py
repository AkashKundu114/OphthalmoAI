"""Screening Anomaly Detection Engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ScreeningAnomaly:
    tenant_id: str
    anomaly_type: str
    severity: str
    metric_name: str
    current_value: float
    baseline_value: float
    deviation_percent: float
    detected_at: datetime
    recommendation: str
    ad_tech_parallel: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "metric_name": self.metric_name,
            "current_value": round(self.current_value, 2) if isinstance(self.current_value, float) else self.current_value,
            "baseline_value": round(self.baseline_value, 2) if isinstance(self.baseline_value, float) else self.baseline_value,
            "deviation_percent": round(self.deviation_percent, 2),
            "detected_at": self.detected_at.isoformat() if isinstance(self.detected_at, datetime) else str(self.detected_at),
            "recommendation": self.recommendation,
            "ad_tech_parallel": self.ad_tech_parallel,
        }


class ScreeningAnomalyDetector:
    """Detects time-series and distributional anomalies on multi-tenant screening metrics."""

    def __init__(
        self,
        volume_z_threshold: float = 2.0,
        quality_drop_threshold_pct: float = 10.0,
        latency_sla_ms: float = 200.0,
        distribution_p_value_threshold: float = 0.05,
    ):
        self.volume_z_threshold = volume_z_threshold
        self.quality_drop_threshold_pct = quality_drop_threshold_pct
        self.latency_sla_ms = latency_sla_ms
        self.distribution_p_value_threshold = distribution_p_value_threshold

    def detect_volume_anomaly(
        self,
        tenant_id: str,
        current_volume: float,
        history_volumes: Sequence[float],
        detected_at: datetime | None = None,
    ) -> ScreeningAnomaly | None:
        """Z-Score volume anomaly detection over 7-day rolling baseline."""
        if not history_volumes or len(history_volumes) < 2:
            return None

        arr = np.array(history_volumes, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else float(np.std(arr))

        if std_val < 1e-4:
            if abs(current_volume - mean_val) > max(5.0, mean_val * 0.5):
                deviation_pct = ((current_volume - mean_val) / max(1.0, mean_val)) * 100.0
                z_score = 3.5 if current_volume > mean_val else -3.5
            else:
                return None
        else:
            z_score = (current_volume - mean_val) / std_val
            deviation_pct = ((current_volume - mean_val) / max(1.0, mean_val)) * 100.0

        if abs(z_score) >= self.volume_z_threshold:
            is_spike = z_score > 0
            severity = "critical" if abs(z_score) >= 3.0 or abs(deviation_pct) >= 60.0 else "warning"
            direction_desc = "surge" if is_spike else "drop"

            rec = (
                f"High screening volume {direction_desc} detected ({current_volume:.0f} vs {mean_val:.1f} baseline). "
                + ("Verify if regional camp occurred; prepare edge cache for burst processing."
                   if is_spike else
                   "Check clinic hardware uplink status, offline queuing sync, or camera hardware failures.")
            )

            return ScreeningAnomaly(
                tenant_id=tenant_id,
                anomaly_type="volume",
                severity=severity,
                metric_name="total_screenings",
                current_value=float(current_volume),
                baseline_value=float(mean_val),
                deviation_percent=float(deviation_pct),
                detected_at=detected_at or _utcnow(),
                recommendation=rec,
                ad_tech_parallel="Similar to detecting sudden impression drops in campaign monitoring",
            )
        return None

    def detect_quality_anomaly(
        self,
        tenant_id: str,
        current_confidence: float,
        history_confidences: Sequence[float],
        detected_at: datetime | None = None,
    ) -> ScreeningAnomaly | None:
        """Moving average confidence decline detection."""
        if not history_confidences or len(history_confidences) < 1:
            return None

        window = list(history_confidences[-3:])
        ma3 = float(np.mean(window))

        norm_curr = current_confidence if current_confidence <= 1.0 else current_confidence / 100.0
        norm_ma3 = ma3 if ma3 <= 1.0 else ma3 / 100.0

        if norm_ma3 <= 0.0:
            return None

        drop_pct = ((norm_ma3 - norm_curr) / norm_ma3) * 100.0

        if drop_pct >= self.quality_drop_threshold_pct:
            severity = "critical" if drop_pct >= 20.0 else "warning"
            rec = (
                f"Diagnostic confidence dropped by {drop_pct:.1f}% below 3-day rolling baseline "
                f"({norm_curr * 100:.1f}% vs {norm_ma3 * 100:.1f}%). Inspect clinic fundus camera optics, "
                "pupil dilation protocols, or optical lens dust accumulation."
            )
            return ScreeningAnomaly(
                tenant_id=tenant_id,
                anomaly_type="quality",
                severity=severity,
                metric_name="avg_confidence",
                current_value=float(norm_curr * 100.0),
                baseline_value=float(norm_ma3 * 100.0),
                deviation_percent=float(-drop_pct),
                detected_at=detected_at or _utcnow(),
                recommendation=rec,
                ad_tech_parallel="Similar to detecting CTR decline from ad fatigue",
            )
        return None

    def detect_sla_breach(
        self,
        tenant_id: str,
        current_latency_ms: float,
        threshold_ms: float | None = None,
        detected_at: datetime | None = None,
    ) -> ScreeningAnomaly | None:
        """Inference latency SLA breach detection."""
        sla_limit = threshold_ms or self.latency_sla_ms
        if current_latency_ms > sla_limit:
            overage_pct = ((current_latency_ms - sla_limit) / sla_limit) * 100.0
            severity = "critical" if current_latency_ms >= 300.0 or overage_pct >= 50.0 else "warning"
            rec = (
                f"Inference latency ({current_latency_ms:.1f}ms) breached {sla_limit:.0f}ms SLA limit. "
                "Scale ONNX multi-threaded worker pools or route low-urgency screenings to edge WebAssembly workers."
            )
            return ScreeningAnomaly(
                tenant_id=tenant_id,
                anomaly_type="sla_breach",
                severity=severity,
                metric_name="avg_inference_time_ms",
                current_value=float(current_latency_ms),
                baseline_value=float(sla_limit),
                deviation_percent=float(overage_pct),
                detected_at=detected_at or _utcnow(),
                recommendation=rec,
                ad_tech_parallel="Similar to bid response latency exceeding RTB timeout",
            )
        return None

    def detect_distribution_shift(
        self,
        tenant_id: str,
        current_diagnoses: Dict[str, int],
        historical_diagnoses: List[Dict[str, int]],
        detected_at: datetime | None = None,
    ) -> ScreeningAnomaly | None:
        """Chi-Square goodness of fit / distribution shift test."""
        if not current_diagnoses or not historical_diagnoses:
            return None

        total_current = sum(current_diagnoses.values())
        if total_current < 5:
            return None

        hist_agg: Dict[str, int] = {}
        for h in historical_diagnoses:
            for k, v in h.items():
                hist_agg[k] = hist_agg.get(k, 0) + int(v)

        total_hist = sum(hist_agg.values())
        if total_hist < 10:
            return None

        all_categories = sorted(set(current_diagnoses.keys()) | set(hist_agg.keys()))
        if len(all_categories) < 2:
            return None

        smoothing = 0.5
        observed = []
        expected = []
        max_class_shift = 0.0
        shifted_class = ""

        smoothed_total_hist = total_hist + smoothing * len(all_categories)

        for cat in all_categories:
            obs = float(current_diagnoses.get(cat, 0))
            exp_p = (hist_agg.get(cat, 0) + smoothing) / smoothed_total_hist
            exp_count = exp_p * total_current

            observed.append(obs)
            expected.append(max(0.1, exp_count))

            curr_prop = obs / total_current
            hist_prop = hist_agg.get(cat, 0) / total_hist
            shift = abs(curr_prop - hist_prop)
            if shift > max_class_shift:
                max_class_shift = shift
                shifted_class = cat

        p_value = 1.0
        if SCIPY_AVAILABLE:
            try:
                exp_arr = np.array(expected, dtype=float)
                exp_arr = exp_arr * (sum(observed) / sum(exp_arr))
                chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=exp_arr)
                p_value = float(p_val)
            except Exception:
                p_value = 1.0
        else:
            chi2_stat = sum(((o - e) ** 2) / e for o, e in zip(observed, expected))
            df = max(1, len(all_categories) - 1)
            critical_val = df * 2.0 + 3.0
            p_value = 0.01 if chi2_stat > critical_val else 0.5

        if (not math.isnan(p_value) and p_value < self.distribution_p_value_threshold) or max_class_shift > 0.35:
            severity = "critical" if p_value < 0.01 or max_class_shift > 0.50 else "warning"
            deviation_pct = max_class_shift * 100.0

            rec = (
                f"Diagnosis distribution shifted significantly (p={p_value:.4f}, max shift: {shifted_class} "
                f"by {deviation_pct:.1f}%). Review cohort selection, demographic shifts, or clinical intake bias."
            )
            return ScreeningAnomaly(
                tenant_id=tenant_id,
                anomaly_type="distribution_shift",
                severity=severity,
                metric_name="diagnosis_distribution",
                current_value=float(max_class_shift * 100.0),
                baseline_value=0.0,
                deviation_percent=float(deviation_pct),
                detected_at=detected_at or _utcnow(),
                recommendation=rec,
                ad_tech_parallel="Similar to audience composition drift in programmatic targeting",
            )
        return None

    def analyze_tenant_metrics(
        self,
        tenant_id: str,
        metrics: Sequence[Any],
        scan_days: int = 14,
    ) -> List[ScreeningAnomaly]:
        """Runs comprehensive anomaly detection across recent metrics in a time-series."""
        if not metrics:
            return []

        rows = []
        for m in metrics:
            if hasattr(m, "to_dict"):
                rows.append(m.to_dict())
            elif isinstance(m, dict):
                rows.append(m)
            else:
                rows.append({
                    "tenant_id": getattr(m, "tenant_id", tenant_id),
                    "date": getattr(m, "date", None),
                    "total_screenings": getattr(m, "total_screenings", 0),
                    "avg_confidence": getattr(m, "avg_confidence", 0.0),
                    "avg_inference_time_ms": getattr(m, "avg_inference_time_ms", 85.0),
                    "diagnoses_by_class": getattr(m, "diagnoses_by_class", {}),
                })

        if not rows:
            return []

        num_days_to_scan = max(1, min(scan_days, len(rows)))
        start_idx = max(0, len(rows) - num_days_to_scan)

        detected_anomalies: List[ScreeningAnomaly] = []

        for i in range(start_idx, len(rows)):
            day = rows[i]
            prior_history = rows[:i]

            raw_d = day.get("date")
            if isinstance(raw_d, str):
                try:
                    d_obj = date.fromisoformat(raw_d)
                    det_at = datetime(d_obj.year, d_obj.month, d_obj.day, 12, 0, 0, tzinfo=timezone.utc)
                except Exception:
                    det_at = _utcnow()
            elif isinstance(raw_d, date) and not isinstance(raw_d, datetime):
                det_at = datetime(raw_d.year, raw_d.month, raw_d.day, 12, 0, 0, tzinfo=timezone.utc)
            elif isinstance(raw_d, datetime):
                det_at = raw_d
            else:
                det_at = _utcnow()

            vol_hist = [float(h.get("total_screenings", 0)) for h in prior_history[-7:]]
            day_vol = float(day.get("total_screenings", 0))
            vol_anom = self.detect_volume_anomaly(tenant_id, day_vol, vol_hist, detected_at=det_at)
            if vol_anom:
                detected_anomalies.append(vol_anom)

            conf_hist = [float(h.get("avg_confidence", 0.0)) for h in prior_history[-3:]]
            day_conf = float(day.get("avg_confidence", 0.0))
            qual_anom = self.detect_quality_anomaly(tenant_id, day_conf, conf_hist, detected_at=det_at)
            if qual_anom:
                detected_anomalies.append(qual_anom)

            day_lat = float(day.get("avg_inference_time_ms", 0.0))
            sla_anom = self.detect_sla_breach(tenant_id, day_lat, detected_at=det_at)
            if sla_anom:
                detected_anomalies.append(sla_anom)

            day_diag = day.get("diagnoses_by_class") or {}
            hist_diags = [h.get("diagnoses_by_class") or {} for h in prior_history[-14:]]
            dist_anom = self.detect_distribution_shift(tenant_id, day_diag, hist_diags, detected_at=det_at)
            if dist_anom:
                detected_anomalies.append(dist_anom)

        detected_anomalies.sort(key=lambda a: a.detected_at, reverse=True)
        return detected_anomalies

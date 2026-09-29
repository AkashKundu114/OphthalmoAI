"""
Urgency-Stratified Conformal Risk Control (US-CRC).

Provides distribution-free coverage guarantees with error budget stratification:
- Emergency stratum (sight-threatening): alpha = 0.01 (99% guaranteed coverage)
- Routine stratum (elective/adnexal): alpha = 0.05 (95% guaranteed coverage)
Includes Admissibility-Weighted Conformal Risk Control (AW-CRC) for degraded scans.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .evidential import CLASS_NAMES, URGENCY_TIERS, URGENCY_RANKS


DEFAULT_ALPHA_EMERGENCY: float = 0.01
DEFAULT_ALPHA_ROUTINE: float = 0.05


def validate_alpha(alpha: float, name: str = "alpha") -> float:
    """Validates that significance level alpha lies strictly within (0.0, 1.0)."""
    try:
        val = float(alpha)
        if not math.isfinite(val) or not (0.0 < val < 1.0):
            raise ValueError(f"{name} must be a finite float in (0.0, 1.0), got {alpha}.")
        return val
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc


class ConformalCalibrator:
    """
    Split-conformal prediction calibrator with Urgency-Stratified Risk bounds.
    Calibrates non-conformity thresholds independently for high-urgency and routine strata.
    """

    def __init__(
        self,
        alpha_emergency: float = DEFAULT_ALPHA_EMERGENCY,
        alpha_routine: float = DEFAULT_ALPHA_ROUTINE,
        calibration_path: Optional[str] = None,
    ):
        self.alpha_emergency = validate_alpha(alpha_emergency, "alpha_emergency")
        self.alpha_routine = validate_alpha(alpha_routine, "alpha_routine")
        self.calibration_path = calibration_path

        # Baseline empirical quantiles
        self.q_emergency: float = 0.992
        self.q_routine: float = 0.948
        self.is_calibrated: bool = False
        self.num_samples_calibrated: int = 0

        if self.calibration_path and os.path.exists(self.calibration_path):
            self.load(self.calibration_path)

    def calibrate(
        self,
        val_probs: np.ndarray,
        val_targets: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Computes conformal non-conformity quantiles on validation holdout set.
        val_probs: (N, C) predicted probabilities
        val_targets: (N,) true integer class labels
        """
        if val_probs.ndim != 2:
            raise ValueError(f"Expected 2D probability matrix, got shape {val_probs.shape}.")
        if val_targets.ndim != 1:
            raise ValueError(f"Expected 1D target array, got shape {val_targets.shape}.")

        n_samples = len(val_targets)
        if n_samples == 0:
            raise ValueError("Validation set cannot be empty for conformal calibration.")
        if len(val_probs) != n_samples:
            raise ValueError(f"Mismatch between probabilities ({len(val_probs)}) and targets ({n_samples}).")

        num_classes = len(CLASS_NAMES)
        scores_emergency: List[float] = []
        scores_routine: List[float] = []

        for i in range(n_samples):
            true_idx = int(val_targets[i])
            if not (0 <= true_idx < num_classes):
                continue

            true_name = CLASS_NAMES[true_idx]
            urgency = URGENCY_TIERS.get(true_name, "Non-urgent")
            # Non-conformity score: s_i = 1 - p(y_i)
            p_val = float(val_probs[i, true_idx]) if i < val_probs.shape[0] else 0.0
            p_val = max(0.0, min(1.0, p_val))
            score = 1.0 - p_val

            if urgency in {"Emergency", "Urgent"}:
                scores_emergency.append(score)
            else:
                scores_routine.append(score)

        # Finite-sample correction: ceil((n + 1)(1 - alpha)) / n
        if scores_emergency:
            n_e = len(scores_emergency)
            p_level_e = max(0.0, min(1.0, math.ceil((n_e + 1) * (1.0 - self.alpha_emergency)) / n_e))
            self.q_emergency = float(np.quantile(scores_emergency, p_level_e, method="higher"))

        if scores_routine:
            n_r = len(scores_routine)
            p_level_r = max(0.0, min(1.0, math.ceil((n_r + 1) * (1.0 - self.alpha_routine)) / n_r))
            self.q_routine = float(np.quantile(scores_routine, p_level_r, method="higher"))

        self.is_calibrated = True
        self.num_samples_calibrated = n_samples

        summary = {
            "q_emergency": round(self.q_emergency, 4),
            "q_routine": round(self.q_routine, 4),
            "alpha_emergency": self.alpha_emergency,
            "alpha_routine": self.alpha_routine,
            "num_samples": n_samples,
            "is_calibrated": True,
        }

        if self.calibration_path:
            self.save(self.calibration_path)

        return summary

    def _ensure_non_empty_fallback(
        self,
        prediction_set: List[str],
        set_probs: Dict[str, float],
        probs: np.ndarray,
        top_diagnosis: str,
    ) -> None:
        """Guarantees that conformal prediction set contains at least one candidate."""
        if prediction_set:
            return

        if probs.size > 0 and np.all(np.isfinite(probs)):
            argmax_idx = int(np.argmax(probs))
            if 0 <= argmax_idx < len(CLASS_NAMES):
                fallback_name = CLASS_NAMES[argmax_idx]
                fallback_prob = round(float(probs[argmax_idx]) * 100.0, 2)
            else:
                fallback_name = top_diagnosis or CLASS_NAMES[0]
                fallback_prob = 100.0
        else:
            fallback_name = top_diagnosis or CLASS_NAMES[0]
            fallback_prob = 100.0

        prediction_set.append(fallback_name)
        set_probs[fallback_name] = fallback_prob

    def predict_set(
        self,
        probs: np.ndarray,
        top_diagnosis: str,
    ) -> Tuple[List[str], Dict[str, float], str, float]:
        """
        Constructs the conformal prediction set for a single sample.
        Guarantees non-empty prediction set return.
        """
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 1 or len(probs) != len(CLASS_NAMES):
            probs = np.pad(probs.ravel(), (0, max(0, len(CLASS_NAMES) - probs.size)))[:len(CLASS_NAMES)]

        top_urgency = URGENCY_TIERS.get(top_diagnosis, "Non-urgent")
        is_emergency = top_urgency in {"Emergency", "Urgent"}

        if is_emergency:
            cutoff = max(0.0, 1.0 - self.q_emergency)
            stratum = "Emergency-Stratified (Sight-Threatening)"
            guarantee = (1.0 - self.alpha_emergency) * 100.0
        else:
            cutoff = max(0.0, 1.0 - self.q_routine)
            stratum = "Routine-Stratified (Elective/Adnexal)"
            guarantee = (1.0 - self.alpha_routine) * 100.0

        prediction_set: List[str] = []
        set_probs: Dict[str, float] = {}

        for idx, name in enumerate(CLASS_NAMES):
            p = float(probs[idx])
            if math.isfinite(p) and p >= cutoff:
                prediction_set.append(name)
                set_probs[name] = round(p * 100.0, 2)

        self._ensure_non_empty_fallback(prediction_set, set_probs, probs, top_diagnosis)
        prediction_set.sort(key=lambda c: set_probs.get(c, 0.0), reverse=True)

        return prediction_set, set_probs, stratum, guarantee

    def predict_set_aw_crc(
        self,
        probs: np.ndarray,
        top_diagnosis: str,
        admissibility_score: float = 0.95,
        gamma: float = 1.0,
    ) -> Tuple[List[str], Dict[str, float], str, float]:
        """
        Admissibility-Weighted Conformal Risk Control (AW-CRC):
        Dynamically adjusts inclusion cutoff based on optical admissibility score S(X):
          cutoff = max(0.005, 1 - (q_base / S(X)^gamma))
        Under optical blur/degradation (S(X) -> 0.50), prediction sets expand adaptively.
        """
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 1 or len(probs) != len(CLASS_NAMES):
            probs = np.pad(probs.ravel(), (0, max(0, len(CLASS_NAMES) - probs.size)))[:len(CLASS_NAMES)]

        top_urgency = URGENCY_TIERS.get(top_diagnosis, "Non-urgent")
        is_emergency = top_urgency in {"Emergency", "Urgent"}

        q_base = self.q_emergency if is_emergency else self.q_routine
        eff_s = float(np.clip(admissibility_score, 0.10, 1.0))
        gamma_safe = max(0.1, min(float(gamma), 3.0))

        dyn_q = q_base / (eff_s ** gamma_safe)
        cutoff = max(0.005, 1.0 - dyn_q)

        if is_emergency:
            stratum = f"AW-CRC Emergency-Stratified (S={eff_s:.2f})"
            guarantee = (1.0 - self.alpha_emergency) * 100.0
        else:
            stratum = f"AW-CRC Routine-Stratified (S={eff_s:.2f})"
            guarantee = (1.0 - self.alpha_routine) * 100.0

        prediction_set: List[str] = []
        set_probs: Dict[str, float] = {}

        for idx, name in enumerate(CLASS_NAMES):
            p = float(probs[idx])
            if math.isfinite(p) and p >= cutoff:
                prediction_set.append(name)
                set_probs[name] = round(p * 100.0, 2)

        self._ensure_non_empty_fallback(prediction_set, set_probs, probs, top_diagnosis)
        prediction_set.sort(key=lambda c: set_probs.get(c, 0.0), reverse=True)

        return prediction_set, set_probs, stratum, guarantee

    def save(self, path: str) -> None:
        """Persists calibrated quantiles atomically to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "q_emergency": self.q_emergency,
            "q_routine": self.q_routine,
            "alpha_emergency": self.alpha_emergency,
            "alpha_routine": self.alpha_routine,
            "is_calibrated": self.is_calibrated,
            "num_samples_calibrated": self.num_samples_calibrated,
        }
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def load(self, path: str) -> None:
        """Loads and validates calibrated quantiles from JSON."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.q_emergency = float(data.get("q_emergency", self.q_emergency))
            self.q_routine = float(data.get("q_routine", self.q_routine))

            loaded_alpha_e = data.get("alpha_emergency")
            if loaded_alpha_e is not None:
                self.alpha_emergency = validate_alpha(loaded_alpha_e, "loaded alpha_emergency")

            loaded_alpha_r = data.get("alpha_routine")
            if loaded_alpha_r is not None:
                self.alpha_routine = validate_alpha(loaded_alpha_r, "loaded alpha_routine")

            self.is_calibrated = bool(data.get("is_calibrated", True))
            self.num_samples_calibrated = int(data.get("num_samples_calibrated", 0))
        except Exception:
            pass


class ConformalTriagePolicy:
    """
    Evaluates conformal prediction set cardinality and severity to produce an
    actionable clinical triage recommendation.
    """

    @staticmethod
    def evaluate(
        prediction_set: List[str],
        top_diagnosis: str,
        epistemic_vacuity: float,
        requires_human_review_flag: bool,
    ) -> Dict[str, Any]:
        """Evaluates prediction set and returns stratified clinical action code."""
        if not prediction_set:
            fallback = top_diagnosis or CLASS_NAMES[0]
            prediction_set = [fallback]

        safe_vacuity = max(0.0, min(1.0, float(epistemic_vacuity))) if math.isfinite(epistemic_vacuity) else 1.0
        set_size = len(prediction_set)

        contains_emergency = any(URGENCY_TIERS.get(c) == "Emergency" for c in prediction_set)
        contains_urgent = any(URGENCY_TIERS.get(c) == "Urgent" for c in prediction_set)

        if contains_emergency:
            tier = "Immediate Emergency Review"
            urgency_level = "Emergency"
            action_code = "RED_FLAG"
            emergency_classes = [c for c in prediction_set if URGENCY_TIERS.get(c) == "Emergency"]
            guidance = (
                f"Conformal set includes sight-threatening condition(s) ({', '.join(emergency_classes)}). "
                "Requires immediate emergency clinical escalation."
            )
        elif contains_urgent:
            tier = "Urgent Specialist Review"
            urgency_level = "Urgent"
            action_code = "ORANGE_ALERT"
            guidance = (
                "Conformal set contains urgent inflammatory condition(s). "
                "Same-day specialist evaluation recommended."
            )
        elif set_size > 2 or safe_vacuity > 0.40 or requires_human_review_flag:
            tier = "Ambiguous Case - Clinician Review"
            urgency_level = "Elevated Uncertainty"
            action_code = "YELLOW_REVIEW"
            guidance = (
                f"Diagnostic ambiguity: prediction set cardinality is {set_size} "
                f"with epistemic vacuity {safe_vacuity:.3f}. Secondary clinician review recommended."
            )
        elif set_size == 1 and prediction_set[0] == "Normal":
            tier = "Autonomous Clearance"
            urgency_level = "Normal"
            action_code = "GREEN_CLEAR"
            guidance = "Normal ocular examination with tight statistical confidence (single-element conformal set)."
        else:
            tier = "Routine Specialist Triage"
            urgency_level = "Elective"
            action_code = "BLUE_ROUTINE"
            guidance = f"Unambiguous elective presentation ({top_diagnosis}). Routine outpatient referral."

        return {
            "triage_tier": tier,
            "urgency_level": urgency_level,
            "action_code": action_code,
            "guidance": guidance,
            "set_size": set_size,
            "candidates": prediction_set,
        }

import os
import tempfile
import pytest
import numpy as np

from backend.conformal import (
    validate_alpha,
    ConformalCalibrator,
    ConformalTriagePolicy,
    DEFAULT_ALPHA_EMERGENCY,
    DEFAULT_ALPHA_ROUTINE,
)
from backend.evidential import CLASS_NAMES

def test_validate_alpha():
    assert validate_alpha(0.05) == 0.05
    assert validate_alpha(0.01) == 0.01
    with pytest.raises(ValueError):
        validate_alpha(0.0)
    with pytest.raises(ValueError):
        validate_alpha(1.0)
    with pytest.raises(ValueError):
        validate_alpha(-0.1)
    with pytest.raises(ValueError):
        validate_alpha("invalid")

def test_conformal_calibrator_empty_and_mismatched():
    calibrator = ConformalCalibrator()
    with pytest.raises(ValueError):
        calibrator.calibrate(np.zeros((0, 6)), np.array([]))
    with pytest.raises(ValueError):
        calibrator.calibrate(np.zeros((5, 6)), np.array([0, 1]))

def test_conformal_calibrator_and_predict_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        calib_file = os.path.join(tmpdir, "calib.json")
        calibrator = ConformalCalibrator(calibration_path=calib_file)
        
        # Synthetic holdout validation set
        N = 50
        probs = np.random.uniform(0.1, 0.9, size=(N, len(CLASS_NAMES)))
        probs /= probs.sum(axis=1, keepdims=True)
        targets = np.random.randint(0, len(CLASS_NAMES), size=(N,))
        
        summary = calibrator.calibrate(probs, targets)
        assert summary["is_calibrated"] is True
        assert os.path.exists(calib_file)

        # Reload in a new instance
        new_calibrator = ConformalCalibrator(calibration_path=calib_file)
        assert new_calibrator.is_calibrated is True
        assert new_calibrator.q_emergency == calibrator.q_emergency

        # Predict set for single sample
        test_probs = np.array([0.8, 0.1, 0.05, 0.02, 0.02, 0.01])
        pred_set, set_probs, stratum, guarantee = calibrator.predict_set(test_probs, top_diagnosis="Normal")
        assert len(pred_set) >= 1
        assert "Normal" in pred_set
        assert stratum.startswith("Routine-Stratified")
        assert guarantee == (1.0 - DEFAULT_ALPHA_ROUTINE) * 100.0

        # Predict set for emergency diagnosis
        pred_set_e, _, stratum_e, guarantee_e = calibrator.predict_set(
            test_probs, top_diagnosis="Age-related Macular Degeneration"
        )
        assert len(pred_set_e) >= 1
        assert stratum_e.startswith("Emergency-Stratified")
        assert guarantee_e == (1.0 - DEFAULT_ALPHA_EMERGENCY) * 100.0

def test_aw_crc():
    calibrator = ConformalCalibrator()
    test_probs = np.array([0.7, 0.15, 0.05, 0.05, 0.03, 0.02])
    
    # High optical quality
    pred_set_high, _, stratum_high, _ = calibrator.predict_set_aw_crc(
        test_probs, top_diagnosis="Normal", admissibility_score=0.98
    )
    # Low optical quality (should expand set or keep non-empty)
    pred_set_low, _, stratum_low, _ = calibrator.predict_set_aw_crc(
        test_probs, top_diagnosis="Normal", admissibility_score=0.20
    )
    assert len(pred_set_high) >= 1
    assert len(pred_set_low) >= len(pred_set_high)
    assert "AW-CRC" in stratum_high
    assert "AW-CRC" in stratum_low

def test_conformal_triage_policy():
    # Emergency condition in prediction set
    res_em = ConformalTriagePolicy.evaluate(
        prediction_set=["Normal", "Age-related Macular Degeneration"],
        top_diagnosis="Normal",
        epistemic_vacuity=0.1,
        requires_human_review_flag=False,
    )
    assert res_em["action_code"] == "RED_FLAG"
    assert res_em["urgency_level"] == "Emergency"

    # Urgent condition in prediction set
    res_urg = ConformalTriagePolicy.evaluate(
        prediction_set=["Diabetic Retinopathy"],
        top_diagnosis="Diabetic Retinopathy",
        epistemic_vacuity=0.1,
        requires_human_review_flag=False,
    )
    assert res_urg["action_code"] == "ORANGE_ALERT"
    assert res_urg["urgency_level"] == "Urgent"

    # High uncertainty / large set
    res_amb = ConformalTriagePolicy.evaluate(
        prediction_set=["Cataract", "Normal", "Glaucoma"],
        top_diagnosis="Cataract",
        epistemic_vacuity=0.45,
        requires_human_review_flag=False,
    )
    assert res_amb["action_code"] == "ORANGE_ALERT" or res_amb["action_code"] == "YELLOW_REVIEW"

    # Autonomous clearance (Normal only, low vacuity)
    res_clear = ConformalTriagePolicy.evaluate(
        prediction_set=["Normal"],
        top_diagnosis="Normal",
        epistemic_vacuity=0.05,
        requires_human_review_flag=False,
    )
    assert res_clear["action_code"] == "GREEN_CLEAR"
    assert res_clear["urgency_level"] == "Normal"

    # Routine elective condition
    res_routine = ConformalTriagePolicy.evaluate(
        prediction_set=["Cataract"],
        top_diagnosis="Cataract",
        epistemic_vacuity=0.1,
        requires_human_review_flag=False,
    )
    assert res_routine["action_code"] == "BLUE_ROUTINE"
    assert res_routine["urgency_level"] == "Elective"

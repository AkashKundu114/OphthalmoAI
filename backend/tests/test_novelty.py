"""
OphthalmoAI - Test Suite for Research Novelty Modules
=====================================================
Tests:
1. DirichletMetaClassifier evidence, Dirichlet properties, and vacuity bounds.
2. AsymmetricClinicalLoss cost matrix properties, gradient propagation, and asymmetric penalties.
3. ConformalCalibrator split-conformal quantile calibration and prediction set generation.
4. ConformalTriagePolicy automated 3-tier clinical decision mapping.
5. extract_visual_biomarkers spatial, morphological, and colorimetric biomarker extraction.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from backend.evidential import (
    DirichletMetaClassifier,
    AsymmetricClinicalLoss,
    EvidentialOODDetector,
    build_clinical_cost_matrix,
    CLASS_NAMES,
)
from backend.conformal import ConformalCalibrator, ConformalTriagePolicy
from backend.biomarker_extractor import extract_visual_biomarkers, format_biomarkers_for_llm
from backend.uncertainty import needs_human_review, build_review_payload


def test_dirichlet_meta_classifier():
    batch_size = 4
    in_features = 36
    num_classes = 12

    model = DirichletMetaClassifier(in_features=in_features, num_classes=num_classes)
    dummy_input = torch.randn(batch_size, in_features)

    probs, vacuity, alpha = model(dummy_input)

    assert probs.shape == (batch_size, num_classes)
    assert vacuity.shape == (batch_size,)
    assert alpha.shape == (batch_size, num_classes)

    # Check probabilities sum to 1
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5)

    # Check non-negative evidence: alpha >= 1.0
    assert torch.all(alpha >= 1.0)

    # Check vacuity bounds
    assert torch.all(vacuity >= 0.0) and torch.all(vacuity <= 1.0)


def test_asymmetric_clinical_cost_matrix():
    C = build_clinical_cost_matrix(12)
    assert C.shape == (12, 12)

    # Diagonal must be 0
    assert torch.all(torch.diag(C) == 0.0)

    # Keratitis (index 5) is Emergency, Conjunctivitis (index 3) is Non-urgent
    keratitis_idx = CLASS_NAMES.index("Keratitis")
    conjunctivitis_idx = CLASS_NAMES.index("Conjunctivitis")

    # Missing Keratitis as Conjunctivitis must have high penalty (5.0)
    cost_miss_emergency = C[keratitis_idx, conjunctivitis_idx].item()
    assert cost_miss_emergency >= 4.0

    # Mistaking Conjunctivitis for Keratitis (safe over-triage) must have low penalty (0.2)
    cost_overtriage = C[conjunctivitis_idx, keratitis_idx].item()
    assert cost_overtriage < 1.0
    assert cost_miss_emergency > cost_overtriage


def test_asymmetric_clinical_loss_gradient_flow():
    model = DirichletMetaClassifier(in_features=36, num_classes=12)
    loss_fn = AsymmetricClinicalLoss(num_classes=12)

    dummy_input = torch.randn(4, 36)
    targets = torch.tensor([0, 5, 2, 11], dtype=torch.long)

    probs, vacuity, alpha = model(dummy_input)
    loss = loss_fn(alpha, targets, epoch=1)

    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_evidential_ood_detector():
    detector = EvidentialOODDetector(vacuity_threshold=0.60)

    # Low vacuity, high evidence -> In-distribution
    alpha_in = torch.ones(12) * 5.0
    is_ood, msg = detector.evaluate(vacuity=0.20, alpha=alpha_in)
    assert not is_ood

    # High vacuity, no evidence -> OOD
    alpha_out = torch.ones(12) * 1.05
    is_ood, msg = detector.evaluate(vacuity=0.85, alpha=alpha_out)
    assert is_ood
    assert "Out-Of-Distribution" in msg


def test_conformal_calibrator_and_prediction_sets():
    calibrator = ConformalCalibrator(alpha_emergency=0.01, alpha_routine=0.05)

    n_val = 100
    np.random.seed(123)
    val_targets = np.random.randint(0, 12, size=(n_val,))
    val_probs = np.random.dirichlet(np.ones(12) * 0.5, size=(n_val,))
    for i in range(n_val):
        val_probs[i, val_targets[i]] += 3.0
        val_probs[i] /= np.sum(val_probs[i])

    summary = calibrator.calibrate(val_probs, val_targets)
    assert summary["is_calibrated"] is True
    assert 0.0 < summary["q_emergency"] <= 1.0
    assert 0.0 < summary["q_routine"] <= 1.0

    # Test prediction set on a sample
    sample_probs = np.zeros(12)
    sample_probs[5] = 0.85 # Keratitis (Emergency)
    sample_probs[3] = 0.15 # Conjunctivitis
    pset, pset_probs, stratum, guarantee = calibrator.predict_set(sample_probs, "Keratitis")

    assert "Keratitis" in pset
    assert guarantee >= 95.0
    assert len(pset) >= 1


def test_conformal_triage_policy():
    # 1. Emergency condition in set -> Red flag Immediate Review
    triage_emerg = ConformalTriagePolicy.evaluate(
        prediction_set=["Keratitis", "Conjunctivitis"],
        top_diagnosis="Keratitis",
        epistemic_vacuity=0.10,
        requires_human_review_flag=False
    )
    assert triage_emerg["triage_tier"] == "Immediate Emergency Review"
    assert triage_emerg["action_code"] == "RED_FLAG"

    # 2. Single Normal diagnosis -> Autonomous Clearance
    triage_normal = ConformalTriagePolicy.evaluate(
        prediction_set=["Normal"],
        top_diagnosis="Normal",
        epistemic_vacuity=0.05,
        requires_human_review_flag=False
    )
    assert triage_normal["triage_tier"] == "Autonomous Clearance"
    assert triage_normal["action_code"] == "GREEN_CLEAR"

    # 3. High cardinality -> Clinician Review
    triage_ambig = ConformalTriagePolicy.evaluate(
        prediction_set=["Blepharitis", "Chalazion", "Stye"],
        top_diagnosis="Blepharitis",
        epistemic_vacuity=0.45,
        requires_human_review_flag=True
    )
    assert "Review" in triage_ambig["triage_tier"]


def test_visual_biomarker_extraction():
    img = Image.new("RGB", (380, 380), color=(180, 100, 100))
    cam = np.zeros((380, 380), dtype=np.float32)
    # Put high activation in the center
    cam[140:240, 140:240] = 1.0

    biomarkers = extract_visual_biomarkers(img, cam, "Keratitis")
    assert "corneal_involvement_pct" in biomarkers
    assert "vascular_erythema_index" in biomarkers
    assert "scleral_icterus_index" in biomarkers
    assert "saliency_focus_profile" in biomarkers
    assert biomarkers["active_lesion_area_pct"] > 0

    llm_context = format_biomarkers_for_llm(
        biomarkers, "Keratitis", ["Keratitis"], 99.0, 0.04
    )
    assert "VERIFIED VISION PIPELINE" in llm_context
    assert "Statistical coverage guarantee: 99.0%" in llm_context


def test_uncertainty_module_extensions():
    # Test critical emergency triggering human review even at 85% confidence
    flagged, reasons = needs_human_review(
        diagnosis="Keratitis",
        confidence_fraction=0.85,
        uncertainty=0.08,
        vacuity=0.10,
        conformal_set=["Keratitis", "Conjunctivitis"]
    )
    assert flagged is True
    assert any("sight-threatening" in r for r in reasons)

    payload = build_review_payload(
        diagnosis="Normal",
        confidence_fraction=0.98,
        uncertainty=0.02,
        vacuity=0.05,
        conformal_set=["Normal"]
    )
    assert payload["requires_human_review"] is False
    assert payload["epistemic_vacuity"] == 0.05

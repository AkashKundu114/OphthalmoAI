import pytest
import torch
import torch.nn as nn
from backend.evidential import (
    CLASS_NAMES,
    URGENCY_TIERS,
    URGENCY_RANKS,
    build_clinical_cost_matrix,
    DirichletMetaClassifier,
    AsymmetricClinicalLoss,
    EvidentialOODDetector,
)

def test_urgency_tiers_and_ranks():
    assert len(CLASS_NAMES) == 6
    for name in CLASS_NAMES:
        assert name in URGENCY_TIERS
        tier = URGENCY_TIERS[name]
        assert tier in URGENCY_RANKS

def test_clinical_cost_matrix():
    C = build_clinical_cost_matrix(6)
    assert C.shape == (6, 6)
    # Diagonal should be 0
    for i in range(6):
        assert C[i, i].item() == 0.0
    
    # Under-triage of high rank to Normal should be heavily penalized (5.0)
    amd_idx = CLASS_NAMES.index("Age-related Macular Degeneration")
    normal_idx = CLASS_NAMES.index("Normal")
    assert C[amd_idx, normal_idx].item() == 5.0

    # Over-triage should have low penalty (0.25)
    assert C[normal_idx, amd_idx].item() == 0.25

def test_dirichlet_meta_classifier():
    model = DirichletMetaClassifier(in_features=18, num_classes=6, hidden_dim=32)
    x = torch.randn(4, 18)
    probs, vacuity, alpha = model(x)

    assert probs.shape == (4, 6)
    assert vacuity.shape == (4,)
    assert alpha.shape == (4, 6)

    # Probabilities should sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    # Alpha should be >= 1.0 (since evidence >= 0)
    assert (alpha >= 1.0).all()
    # Vacuity u = K / S in (0, 1]
    assert (vacuity > 0).all()
    assert (vacuity <= 1.0).all()

def test_asymmetric_clinical_loss():
    loss_fn = AsymmetricClinicalLoss(num_classes=6, lambda_cost=1.0, lambda_kl=0.01)
    alpha = torch.tensor([[2.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 3.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    loss = loss_fn(alpha, targets, epoch=1)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0.0

def test_evidential_ood_detector():
    detector = EvidentialOODDetector(vacuity_threshold=0.50)
    
    # Normal case: low vacuity, high max prob
    res_normal = detector.evaluate(vacuity=0.2, max_prob=0.85)
    assert res_normal["is_ood"] is False
    assert res_normal["rejection_reason"] is None
    assert res_normal["epistemic_vacuity"] == 0.2

    # High vacuity OOD
    res_high_vacuity = detector.evaluate(vacuity=0.65, max_prob=0.80)
    assert res_high_vacuity["is_ood"] is True
    assert "High epistemic uncertainty" in res_high_vacuity["rejection_reason"]

    # Low confidence OOD
    res_low_conf = detector.evaluate(vacuity=0.30, max_prob=0.35)
    assert res_low_conf["is_ood"] is True
    assert res_low_conf["rejection_reason"] is not None

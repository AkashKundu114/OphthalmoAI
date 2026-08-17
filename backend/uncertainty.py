from __future__ import annotations
from typing import Dict, List, Tuple, Any
import torch

CRITICAL_DIAGNOSES = {"Uveitis", "Jaundice"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_UNCERTAINTY_THRESHOLD = 0.15
CRITICAL_CONFIDENCE_THRESHOLD = 0.90


@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    n_passes: int = 8,
) -> Tuple[torch.Tensor, float]:
    
    was_training = model.training
    model.train()
    try:
        probs_mc = []
        for _ in range(n_passes):
            logits = model(input_tensor)
            probs_mc.append(torch.nn.functional.softmax(logits[0], dim=0))
        probs_stack = torch.stack(probs_mc)
        mean_probs = probs_stack.mean(dim=0)
        epistemic_uncertainty = float(probs_stack.var(dim=0).sum().item())
        return mean_probs, epistemic_uncertainty
    finally:
        model.train(was_training)


def needs_human_review(
    diagnosis: str,
    confidence_fraction: float,
    uncertainty: float,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    critical_confidence_threshold: float = CRITICAL_CONFIDENCE_THRESHOLD,
) -> Tuple[bool, List[str]]:
    
    reasons: List[str] = []
    if confidence_fraction < confidence_threshold:
        reasons.append(f"Confidence ({confidence_fraction*100:.1f}%) is below the {confidence_threshold*100:.0f}% review threshold.")
    if uncertainty > uncertainty_threshold:
        reasons.append(f"Model uncertainty ({uncertainty:.3f}) exceeds the {uncertainty_threshold:.2f} threshold across repeated passes.")
    if diagnosis in CRITICAL_DIAGNOSES and confidence_fraction < critical_confidence_threshold:
        reasons.append(f"'{diagnosis}' is a sight-threatening or systemic-emergency diagnosis; confidence must exceed {critical_confidence_threshold*100:.0f}% to skip review, but was {confidence_fraction*100:.1f}%.")
    return (len(reasons) > 0, reasons)


def build_review_payload(
    diagnosis: str,
    confidence_fraction: float,
    uncertainty: float,
) -> Dict[str, Any]:
    
    flagged, reasons = needs_human_review(diagnosis, confidence_fraction, uncertainty)
    return {"requires_human_review": flagged, "review_reasons": reasons, "uncertainty": round(uncertainty, 4)}

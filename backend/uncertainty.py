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
    """
    Perform Monte Carlo Dropout predictions to estimate model epistemic uncertainty.

    Args:
        model: The trained PyTorch neural network model.
        input_tensor: The preprocessed input image tensor.
        n_passes: Number of forward passes to perform with dropout active.

    Returns:
        A tuple containing:
            - The mean probability distribution tensor over the classes.
            - The epistemic uncertainty (variance sum across classes) as a float.
    """
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
    """
    Evaluate if an AI classification requires human review based on confidence, uncertainty,
    and clinical criticality guidelines.

    Args:
        diagnosis: The predicted classification label.
        confidence_fraction: The calibrated prediction probability (0.0 to 1.0).
        uncertainty: Estimated epistemic uncertainty.
        confidence_threshold: Review threshold for standard diagnoses.
        uncertainty_threshold: Max uncertainty allowed before requiring review.
        critical_confidence_threshold: Review threshold for sight-threatening diagnoses.

    Returns:
        A tuple containing:
            - A boolean indicating if review is required.
            - A list of string reasons explaining why review is required (if any).
    """
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
    """
    Construct the database clinician review payload for an AI screening task.

    Args:
        diagnosis: The predicted eye condition.
        confidence_fraction: Calibrated probability.
        uncertainty: Epistemic uncertainty value.

    Returns:
        A dictionary containing keys: 'requires_human_review', 'review_reasons', and 'uncertainty'.
    """
    flagged, reasons = needs_human_review(diagnosis, confidence_fraction, uncertainty)
    return {"requires_human_review": flagged, "review_reasons": reasons, "uncertainty": round(uncertainty, 4)}

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List, Tuple
from weakref import WeakKeyDictionary

import torch

CRITICAL_DIAGNOSES = {
    "Uveitis",
    "Jaundice",
    "Keratitis",
    "Age-related Macular Degeneration",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Hypertensive Retinopathy / Pathological Myopia",
}
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_UNCERTAINTY_THRESHOLD = 0.15
DEFAULT_VACUITY_THRESHOLD = 0.40
CRITICAL_CONFIDENCE_THRESHOLD = 0.90

_DROPOUT_MODULES = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
    torch.nn.FeatureAlphaDropout,
)
_MODEL_LOCKS: WeakKeyDictionary = WeakKeyDictionary()
_MODEL_LOCKS_GUARD = Lock()


def _model_lock(model: torch.nn.Module) -> Lock:
    with _MODEL_LOCKS_GUARD:
        lock = _MODEL_LOCKS.get(model)
        if lock is None:
            lock = Lock()
            _MODEL_LOCKS[model] = lock
        return lock


@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    n_passes: int = 8,
) -> Tuple[torch.Tensor, float]:
    """Return mean probabilities and summed class-wise MC variance.

    Only registered ``nn.Dropout`` modules are activated. Functional or
    recurrent-layer internal dropout cannot be enabled independently and
    is intentionally not claimed as supported by this helper.
    """
    if isinstance(n_passes, bool) or not isinstance(n_passes, int) or n_passes < 1:
        raise ValueError("n_passes must be a positive integer")

    with _model_lock(model):
        modules = list(model.modules())
        module_states = [(module, module.training) for module in modules]
        try:
            # Set flags directly to avoid recursive train()/eval() calls,
            # which can overwrite mixed child states or invoke custom hooks.
            for module in modules:
                module.training = False
            for module in modules:
                if isinstance(module, _DROPOUT_MODULES):
                    module.training = True

            probs_mc = []
            for _ in range(n_passes):
                logits = model(input_tensor)
                if logits.ndim < 2 or logits.shape[0] < 1:
                    raise ValueError("model must return a non-empty batch of logits")
                probs_mc.append(torch.nn.functional.softmax(logits[0], dim=0))
            probs_stack = torch.stack(probs_mc)
            mean_probs = probs_stack.mean(dim=0)
            epistemic_uncertainty = float(probs_stack.var(dim=0, unbiased=False).sum().item())
            return mean_probs, epistemic_uncertainty
        finally:
            for module, was_training in module_states:
                module.training = was_training


def needs_human_review(
    diagnosis: str,
    confidence_fraction: float,
    uncertainty: float,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    critical_confidence_threshold: float = CRITICAL_CONFIDENCE_THRESHOLD,
    conformal_set: List[str] | None = None,
    vacuity: float | None = None,
    vacuity_threshold: float = DEFAULT_VACUITY_THRESHOLD,
) -> Tuple[bool, List[str]]:

    reasons: List[str] = []
    if confidence_fraction < confidence_threshold:
        reasons.append(f"Confidence ({confidence_fraction*100:.1f}%) is below the {confidence_threshold*100:.0f}% review threshold.")
    if uncertainty > uncertainty_threshold:
        reasons.append(f"Model uncertainty ({uncertainty:.3f}) exceeds the {uncertainty_threshold:.2f} threshold.")
    if vacuity is not None and vacuity > vacuity_threshold:
        reasons.append(f"Evidential vacuity ({vacuity:.3f}) indicates elevated model ignorance/OOD risk (threshold {vacuity_threshold:.2f}).")
    if conformal_set is not None:
        if len(conformal_set) > 2:
            reasons.append(f"Conformal prediction set is multi-modal with {len(conformal_set)} candidates ({', '.join(conformal_set)}).")
        emergency_candidates = [c for c in conformal_set if c in CRITICAL_DIAGNOSES]
        if emergency_candidates and diagnosis not in CRITICAL_DIAGNOSES:
            reasons.append(f"Conformal set includes undetected emergency condition(s): {', '.join(emergency_candidates)}.")
    if diagnosis in CRITICAL_DIAGNOSES and confidence_fraction < critical_confidence_threshold:
        reasons.append(f"'{diagnosis}' is a sight-threatening or systemic-emergency diagnosis; confidence must exceed {critical_confidence_threshold*100:.0f}% to skip review, but was {confidence_fraction*100:.1f}%.")
    return (len(reasons) > 0, reasons)


def build_review_payload(
    diagnosis: str,
    confidence_fraction: float,
    uncertainty: float,
    conformal_set: List[str] | None = None,
    vacuity: float | None = None,
) -> Dict[str, Any]:

    flagged, reasons = needs_human_review(
        diagnosis, confidence_fraction, uncertainty,
        conformal_set=conformal_set, vacuity=vacuity
    )
    payload = {
        "requires_human_review": flagged,
        "review_reasons": reasons,
        "uncertainty": round(uncertainty, 4)
    }
    if vacuity is not None:
        payload["epistemic_vacuity"] = round(vacuity, 4)
    if conformal_set is not None:
        payload["conformal_candidates_count"] = len(conformal_set)
    return payload

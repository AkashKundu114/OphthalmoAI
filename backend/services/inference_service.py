"""
Clinical Inference & Triage Service.

Coordinates:
- Inference capacity throttling with asynchronous semaphores
- Clinical symptom cross-checks & heuristic discordance analysis
- Ensemble forward pass with temperature scaling and Platt calibration
- Epistemic vacuity and Dirichlet evidential uncertainty estimation
- Urgency-stratified Conformal Prediction (AW-CRC)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from backend.calibration import CalibrationRegistry, apply_temperature
from backend.config import (
    CALIBRATION_PATH,
    CONFORMAL_CALIBRATION_PATH,
    DEVICE,
    MAX_CONCURRENT_INFERENCES,
    MONOLITHIC_CLASSES,
)
from backend.conformal import ConformalCalibrator
from backend.logging_config import get_logger

logger = get_logger("ophthalmoai.inference_service")

# Global registries initialized once
CALIBRATION_REGISTRY = CalibrationRegistry(CALIBRATION_PATH)
CONFORMAL_CALIBRATOR = ConformalCalibrator(calibration_path=CONFORMAL_CALIBRATION_PATH)

# Semaphore for GPU/CPU inference concurrency protection
_INFERENCE_SEMAPHORE: Optional[asyncio.Semaphore] = None


def get_inference_semaphore() -> asyncio.Semaphore:
    """Returns the shared asyncio semaphore limiting simultaneous model executions."""
    global _INFERENCE_SEMAPHORE
    if _INFERENCE_SEMAPHORE is None:
        _INFERENCE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
    return _INFERENCE_SEMAPHORE


_SEVERITY_ICON = {"info": "✅", "warning": "⚠️", "urgent": "🚨"}


def _build_symptom_alerts(
    diagnosis: str,
    pain_level: str,
    vision_loss: str,
    itchiness: str,
    halos: str = "No",
    discharge: str = "None",
    light_sensitivity: str = "No",
    floaters: str = "No",
    duration: str = "Not Sure",
    **kwargs: Any,
) -> List[Tuple[str, str]]:
    alerts: List[Tuple[str, str]] = []
    p_lower = str(pain_level).lower()
    v_lower = str(vision_loss).lower()
    i_lower = str(itchiness).lower()
    h_lower = str(halos).lower()
    d_lower = str(discharge).lower()
    l_lower = str(light_sensitivity).lower()
    f_lower = str(floaters).lower()
    dur_lower = str(duration).lower()

    # Cataract cross-checks
    if diagnosis == "Cataract":
        if "yes" in h_lower or "rainbow" in h_lower:
            alerts.append(("info", "Symptom Concordance: Light Halos & glare strongly correlate with lenticular opacification."))
        if "severe" in p_lower or "throbbing" in p_lower:
            alerts.append(("urgent", "Atypical Presentation: Severe pain is NOT typical for uncomplicated cataract. Rule out secondary phacolytic glaucoma or acute angle closure."))
        if "yes" in v_lower or "blur" in v_lower:
            alerts.append(("info", "Clinical Match: Progressive painless visual acuity reduction is consistent with cataract formation."))

    # Uveitis cross-checks
    elif diagnosis == "Uveitis":
        if "severe" in p_lower or "moderate" in p_lower or "throbbing" in p_lower:
            alerts.append(("urgent", "URGENT TRIAGE: Deep ciliary/ocular pain with suspected Uveitis represents a sight-threatening inflammatory condition."))
        if "yes" in l_lower or "severe" in l_lower:
            alerts.append(("urgent", "Photophobia Alert: Light sensitivity reflects acute ciliary spasm and anterior chamber inflammation."))

    # Keratitis cross-checks
    elif diagnosis == "Keratitis":
        alerts.append(("urgent", "CRITICAL CORNEAL EMERGENCY: Suspected Keratitis requires immediate same-day slit-lamp examination to rule out bacterial/fungal corneal ulceration."))
        if "severe" in p_lower or "foreign" in p_lower:
            alerts.append(("info", "Corneal Reflex: Sharp pain and foreign body sensation directly match corneal epithelial compromise."))
        if "purulent" in d_lower or "yellow" in d_lower:
            alerts.append(("urgent", "Bacterial Infiltration: Purulent discharge warrants urgent corneal scraping and fortified antimicrobial therapy."))

    # Conjunctivitis cross-checks
    elif diagnosis == "Conjunctivitis":
        if "severe" in p_lower or "throbbing" in p_lower:
            alerts.append(("warning", "Pain Mismatch: Severe pain is atypical for simple pink eye. Rule out acute keratitis, scleritis, or acute glaucoma."))
        if "yes" in v_lower or "significant" in v_lower:
            alerts.append(("warning", "Vision Threat Alert: Significant vision loss is NOT expected in conjunctivitis. Urgent ophthalmic evaluation recommended."))
        if "yes" in i_lower or "itch" in i_lower:
            alerts.append(("info", "Allergic Phenotype: Prominent pruritus (Itchiness) indicates allergic conjunctivitis etiology."))
        if "purulent" in d_lower or "yellow" in d_lower or "crust" in d_lower:
            alerts.append(("info", "Bacterial Phenotype: Purulent/mucopurulent discharge and crusting strongly suggest bacterial conjunctivitis."))
        if "month" in dur_lower or "chronic" in dur_lower:
            alerts.append(("warning", "Chronic Presentation: Symptoms persisting >1 month warrant investigation for atypical, chlamydial, or toxic conjunctivitis."))

    # Jaundice (Scleral Icterus) cross-checks
    elif diagnosis == "Jaundice":
        alerts.append(("urgent", "SYSTEMIC EMERGENCY: Scleral icterus reflects elevated serum bilirubin (hepatic/biliary pathology). Immediate comprehensive metabolic panel & systemic evaluation required."))

    # Subconjunctival Hemorrhage cross-checks
    elif diagnosis == "Subconjunctival Hemorrhage":
        if "none" in p_lower and "no" in v_lower:
            alerts.append(("info", "Benign Reassurance: Painless, sharply demarcated hemorrhage with preserved visual acuity is typical of benign subconjunctival bleeding."))
        if "severe" in p_lower or "yes" in v_lower:
            alerts.append(("warning", "Trauma / Coagulopathy Concern: Pain or vision deficit with subconjunctival bleeding warrants ruling out globe rupture or retrobulbar hemorrhage."))

    # Ptosis cross-checks
    elif diagnosis == "Ptosis":
        if "acute" in dur_lower or "<24" in dur_lower:
            alerts.append(("urgent", "Neurological Alert: Sudden acute onset ptosis requires immediate evaluation to rule out 3rd cranial nerve palsy or Horner's syndrome."))

    # Blepharitis cross-checks
    elif diagnosis == "Blepharitis":
        if "yes" in i_lower or "crust" in d_lower:
            alerts.append(("info", "Clinical Concordance: Lid margin pruritus and collarette debris are hallmark signs of anterior/posterior blepharitis."))

    # Chalazion vs Stye cross-checks
    elif diagnosis in ["Chalazion", "Stye"]:
        if diagnosis == "Stye" and ("mild" in p_lower or "moderate" in p_lower or "severe" in p_lower):
            alerts.append(("info", "Infectious Match: Acute focal tenderness and lid margin erythema correspond to hordeolum (stye)."))
        elif diagnosis == "Chalazion" and "none" in p_lower:
            alerts.append(("info", "Granulomatous Match: Chronic painless focal meibomian granuloma is characteristic of a chalazion."))

    # General / Floater Cross-Checks
    if "shower" in f_lower or ("yes" in f_lower and diagnosis not in ["Uveitis", "Normal"]):
        alerts.append(("warning", "Posterior Segment Warning: New-onset Floaters or flashes warrant dilated peripheral retinal examination to rule out retinal tear or detachment."))

    return alerts


def analyze_symptoms(
    diagnosis: str,
    pain_level: str,
    vision_loss: str,
    itchiness: str,
    **kwargs: Any,
) -> List[str]:
    """Formats clinical symptom discordance alerts with severity icons."""
    alerts = _build_symptom_alerts(diagnosis, pain_level, vision_loss, itchiness, **kwargs)
    return [f"{_SEVERITY_ICON[s]} {m}" for s, m in alerts]


def analyze_symptoms_structured(
    diagnosis: str,
    pain_level: str,
    vision_loss: str,
    itchiness: str,
    **kwargs: Any,
) -> List[Dict[str, str]]:
    """Returns structured severity-message dictionaries for frontend triage alerts."""
    alerts = _build_symptom_alerts(diagnosis, pain_level, vision_loss, itchiness, **kwargs)
    return [{"severity": s, "message": m} for s, m in alerts]

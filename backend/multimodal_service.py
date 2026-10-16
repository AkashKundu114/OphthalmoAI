"""OphthalmoAI - Multimodal Clinical Fusion Service"""

from typing import Any, Dict

CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy / Pathological Myopia"
]

def compute_multimodal_clinical_prior(
    age: int | None = None,
    sex: str | None = None,
    hba1c: float | None = None,
    sbp: int | None = None,
    dbp: int | None = None,
    iop: float | None = None,
    va_logmar: float | None = None,
    dm_duration: float | None = None,
) -> Dict[str, float]:
    """Computes epidemiological likelihood multipliers based on systemic clinical biomarkers."""
    weights = {c: 1.0 for c in CLASSES}

    if hba1c is not None:
        if hba1c >= 9.0:
            weights["Diabetic Retinopathy"] *= 3.8
            weights["Normal"] *= 0.3
        elif hba1c >= 7.5:
            weights["Diabetic Retinopathy"] *= 2.2
            weights["Normal"] *= 0.6
        elif hba1c < 5.7:
            weights["Diabetic Retinopathy"] *= 0.2
            weights["Normal"] *= 1.4

    if dm_duration is not None and dm_duration > 10.0:
        weights["Diabetic Retinopathy"] *= 2.0

    if iop is not None:
        if iop >= 25.0:
            weights["Glaucoma"] *= 4.5
            weights["Normal"] *= 0.2
        elif iop >= 22.0:
            weights["Glaucoma"] *= 2.6
            weights["Normal"] *= 0.6
        elif iop <= 16.0:
            weights["Glaucoma"] *= 0.5

    if age is not None and age >= 65:
        weights["Glaucoma"] *= 1.4
        weights["Cataract"] *= 2.5
        weights["Age-related Macular Degeneration"] *= 2.8

    if sbp is not None and dbp is not None:
        if sbp >= 160 or dbp >= 100:
            weights["Hypertensive Retinopathy / Pathological Myopia"] *= 4.0
            weights["Normal"] *= 0.3
        elif sbp >= 140 or dbp >= 90:
            weights["Hypertensive Retinopathy / Pathological Myopia"] *= 2.2
            weights["Normal"] *= 0.7

    if va_logmar is not None and va_logmar >= 0.5:
        weights["Cataract"] *= 1.8

    return weights

def fuse_image_and_biodata(
    image_probs: Dict[str, float],
    patient_biodata: Dict[str, Any]
) -> Dict[str, Any]:
    """Bayesian integration of visual posterior with systemic biomarkers:"""
    age = patient_biodata.get("patient_age") or patient_biodata.get("age")
    sex = patient_biodata.get("sex")
    hba1c = patient_biodata.get("hba1c") or patient_biodata.get("hba1c_pct")
    sbp = patient_biodata.get("systolic_bp")
    dbp = patient_biodata.get("diastolic_bp")
    iop = patient_biodata.get("intraocular_pressure") or patient_biodata.get("intraocular_pressure_mmhg")
    va = patient_biodata.get("visual_acuity_logmar")
    dm_dur = patient_biodata.get("diabetes_duration_years")

    bio_weights = compute_multimodal_clinical_prior(
        age=age, sex=sex, hba1c=hba1c, sbp=sbp, dbp=dbp, iop=iop, va_logmar=va, dm_duration=dm_dur
    )

    unnorm_probs = {}
    for c in CLASSES:
        img_p = image_probs.get(c, 0.0) / 100.0 if image_probs.get(c, 0.0) > 1.0 else image_probs.get(c, 0.0)
        unnorm_probs[c] = img_p * bio_weights.get(c, 1.0)

    total = sum(unnorm_probs.values()) if sum(unnorm_probs.values()) > 0 else 1.0
    fused_probs = {c: round((val / total) * 100.0, 2) for c, val in unnorm_probs.items()}
    top_diagnosis = max(fused_probs, key=fused_probs.get)

    clinical_flags = []
    if (iop and iop >= 24.0) and top_diagnosis == "Normal":
        clinical_flags.append("Ocular Hypertension Flag: Elevated IOP (>= 24 mmHg) detected despite normal posterior pole view. Slit-lamp gonioscopy recommended.")
    if (hba1c and hba1c >= 8.5) and top_diagnosis == "Normal":
        clinical_flags.append("High-Risk Metabolic Flag: Severe hyperglycemia (HbA1c >= 8.5%) presents imminent risk for diabetic microangiopathy. 6-month screening recall advised.")
    if (sbp and sbp >= 160) and top_diagnosis == "Normal":
        clinical_flags.append("Stage 2 Systemic Hypertension Flag: Patient exhibits severe hypertension without focal retinal microvascular signs.")

    return {
        "fused_probabilities": fused_probs,
        "fused_diagnosis": top_diagnosis,
        "fused_confidence": fused_probs[top_diagnosis],
        "systemic_prior_weights": bio_weights,
        "clinical_flags": clinical_flags
    }

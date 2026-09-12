from __future__ import annotations
from typing import Dict, List, Optional, TypedDict

class ClinicalCodeEntry(TypedDict):
    icd10: str
    snomed_ct: str
    urgency: str
    urgency_rank: int
    referral: str
    escalation_message: Optional[str]

CLINICAL_CODES: Dict[str, ClinicalCodeEntry] = {
    "Diabetic Retinopathy": {
        "icd10": "E11.319 / H36.0",
        "snomed_ct": "4855003",
        "urgency": "urgent",
        "urgency_rank": 3,
        "referral": "Retina Specialist / Vitreoretinal Surgeon",
        "escalation_message": (
            "Diabetic Retinopathy is a microvascular complication of diabetes causing retinal ischemia, "
            "macular edema, or neovascularization. Urgent evaluation with optical coherence tomography (OCT) "
            "and dilated funduscopy is indicated. If visual distortion or sudden floaters are present, seek same-day care."
        )
    },
    "Glaucoma": {
        "icd10": "H40.9",
        "snomed_ct": "23986001",
        "urgency": "urgent",
        "urgency_rank": 3,
        "referral": "Glaucoma Specialist / Comprehensive Ophthalmologist",
        "escalation_message": (
            "Glaucomatous optic neuropathy causes irreversible retinal ganglion cell axon loss and progressive visual field constriction. "
            "Comprehensive tonometry, gonioscopy, and retinal nerve fiber layer (RNFL) imaging are required promptly."
        )
    },
    "Age-related Macular Degeneration": {
        "icd10": "H35.30",
        "snomed_ct": "267718000",
        "urgency": "urgent",
        "urgency_rank": 3,
        "referral": "Retinal Specialist - Urgent (within 24-48 hours if wet AMD suspected)",
        "escalation_message": (
            "Macular degeneration damages central photopic vision. Sudden metamorphopsia (distorted straight lines) "
            "or central scotoma indicates possible choroidal neovascularization ('wet' AMD) requiring emergency anti-VEGF intervention."
        )
    },
    "Cataract": {
        "icd10": "H26.9",
        "snomed_ct": "193570009",
        "urgency": "elective",
        "urgency_rank": 1,
        "referral": "Cataract Surgeon / General Ophthalmologist",
        "escalation_message": None
    },
    "Hypertensive Retinopathy / Pathological Myopia": {
        "icd10": "H35.0 / H44.20",
        "snomed_ct": "38341003",
        "urgency": "urgent",
        "urgency_rank": 2,
        "referral": "Ophthalmologist & Primary Care / Cardiologist",
        "escalation_message": (
            "Retinal vascular changes reflect end-organ microvascular damage or progressive axial elongation. "
            "Immediate systemic blood pressure evaluation and dilated peripheral retinal examination are indicated."
        )
    },
    "Normal": {
        "icd10": "Z01.00",
        "snomed_ct": "165070006",
        "urgency": "none",
        "urgency_rank": 0,
        "referral": "Routine annual dilated retinal screening",
        "escalation_message": None
    },
    # Secondary and historical ocular conditions for comprehensive clinical triage
    "Uveitis": {
        "icd10": "H20.9",
        "snomed_ct": "128473001",
        "urgency": "urgent",
        "urgency_rank": 3,
        "referral": "Uveitis specialist or ophthalmologist - same-week, sooner if pain/photophobia present",
        "escalation_message": (
            "Uveitis is a sight-threatening emergency. Untreated, it can progress to "
            "glaucoma, cataracts, or permanent vision loss. Seek an ophthalmologist "
            "or uveitis specialist as soon as possible - same-day if pain, photophobia, "
            "or vision changes are present."
        )
    },
    "Jaundice": {
        "icd10": "R17",
        "snomed_ct": "65142007",
        "urgency": "emergency",
        "urgency_rank": 4,
        "referral": "Internal medicine / Gastroenterology - same-day evaluation",
        "escalation_message": (
            "Scleral icterus (yellowing of the eye) is a systemic warning sign, not "
            "an eye disease - it indicates elevated bilirubin and possible liver, "
            "gallbladder, or blood disorder. This requires same-day evaluation by a "
            "physician with liver function tests (LFTs), not routine eye care."
        )
    },
    "Conjunctivitis": {
        "icd10": "H10.9",
        "snomed_ct": "9826008",
        "urgency": "elective",
        "urgency_rank": 1,
        "referral": "Primary Eye Care / General Practitioner",
        "escalation_message": None
    },
}

def get_clinical_code(diagnosis: str) -> ClinicalCodeEntry:
    # Normalize diagnosis strings
    diag_clean = diagnosis.strip()
    for key in CLINICAL_CODES:
        if key.lower() == diag_clean.lower():
            return CLINICAL_CODES[key]
    for key in CLINICAL_CODES:
        if key.lower() in diag_clean.lower() or diag_clean.lower() in key.lower():
            return CLINICAL_CODES[key]
    return {
        "icd10": "H35.9",
        "snomed_ct": "371087003",
        "urgency": "urgent",
        "urgency_rank": 2,
        "referral": "Ophthalmologist (diagnosis not recognised by clinical code table)",
        "escalation_message": "This diagnosis label isn't recognised by the clinical coding table. Treat this result with caution and consult an ophthalmologist.",
    }

def is_critical(diagnosis: str) -> bool:
    return get_clinical_code(diagnosis)["urgency_rank"] >= 3

def sort_by_urgency(diagnoses: List[str], reverse: bool = True) -> List[str]:
    return sorted(diagnoses, key=lambda d: get_clinical_code(d)["urgency_rank"], reverse=reverse)

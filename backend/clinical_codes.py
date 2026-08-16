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
    "Cataract": {"icd10": "H26.9", "snomed_ct": "193570009", "urgency": "elective", "urgency_rank": 1,
                 "referral": "Ophthalmologist (cataract surgery evaluation)", "escalation_message": None},
    "Conjunctivitis": {"icd10": "H10.9", "snomed_ct": "9826008", "urgency": "non-urgent", "urgency_rank": 2,
                        "referral": "GP or optometrist", "escalation_message": None},
    "Ptosis": {"icd10": "H02.409", "snomed_ct": "111002", "urgency": "non-urgent", "urgency_rank": 2,
               "referral": "Ophthalmologist or Oculoplastic Surgeon", "escalation_message": None},
    "Blepharitis": {"icd10": "H01.009", "snomed_ct": "41503000", "urgency": "non-urgent", "urgency_rank": 1,
                    "referral": "Optometrist or GP", "escalation_message": None},
    "Chalazion": {"icd10": "H00.19", "snomed_ct": "373686008", "urgency": "elective", "urgency_rank": 1,
                  "referral": "Optometrist or Ophthalmologist", "escalation_message": None},
    "Stye": {"icd10": "H00.019", "snomed_ct": "128566001", "urgency": "non-urgent", "urgency_rank": 2,
             "referral": "GP or Optometrist", "escalation_message": None},
    "Keratitis": {"icd10": "H16.9", "snomed_ct": "51877002", "urgency": "urgent", "urgency_rank": 3,
                  "referral": "Ophthalmologist - Same Day", "escalation_message": "Keratitis (corneal inflammation/ulcer) is a medical emergency that can lead to rapid vision loss. Seek immediate ophthalmic care."},
    "Subconjunctival Hemorrhage": {"icd10": "H11.30", "snomed_ct": "95724003", "urgency": "none", "urgency_rank": 0,
                                   "referral": "Routine monitor (GP)", "escalation_message": None},
    "Pterygium": {"icd10": "H11.009", "snomed_ct": "65876009", "urgency": "elective", "urgency_rank": 1,
                  "referral": "Ophthalmologist (monitor; surgical referral if visually significant)", "escalation_message": None},
    "Uveitis": {"icd10": "H20.9", "snomed_ct": "128473001", "urgency": "urgent", "urgency_rank": 3,
                "referral": "Uveitis specialist or ophthalmologist - same-week, sooner if pain/photophobia present",
                "escalation_message": ("Uveitis is a sight-threatening emergency. Untreated, it can progress to "
                                        "glaucoma, cataracts, or permanent vision loss. Seek an ophthalmologist "
                                        "or uveitis specialist as soon as possible - same-day if pain, photophobia, "
                                        "or vision changes are present.")},
    "Jaundice": {"icd10": "R17", "snomed_ct": "65142007", "urgency": "emergency", "urgency_rank": 4,
                 "referral": "Internal medicine / Gastroenterology - same-day evaluation",
                 "escalation_message": ("Scleral icterus (yellowing of the eye) is a systemic warning sign, not "
                                         "an eye disease - it indicates elevated bilirubin and possible liver, "
                                         "gallbladder, or blood disorder. This requires same-day evaluation by a "
                                         "physician with liver function tests (LFTs), not routine eye care.")},
    "Normal": {"icd10": "Z01.00", "snomed_ct": "165070006", "urgency": "none", "urgency_rank": 0,
               "referral": "Routine screening per standard schedule", "escalation_message": None},
}


def get_clinical_code(diagnosis: str) -> ClinicalCodeEntry:
    return CLINICAL_CODES.get(diagnosis, {
        "icd10": "Z01.00", "snomed_ct": "165070006", "urgency": "non-urgent", "urgency_rank": 2,
        "referral": "Ophthalmologist (diagnosis not recognised by clinical code table)",
        "escalation_message": ("This diagnosis label isn't recognised by the clinical coding table. "
                                "Treat this result with caution and consult an ophthalmologist."),
    })


def is_critical(diagnosis: str) -> bool:
    return get_clinical_code(diagnosis)["urgency_rank"] >= 3


def sort_by_urgency(diagnoses: List[str]) -> List[str]:
    return sorted(diagnoses, key=lambda d: get_clinical_code(d)["urgency_rank"], reverse=True)

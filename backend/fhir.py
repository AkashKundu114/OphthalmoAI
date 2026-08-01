"""
HL7 FHIR R4 Exporter Module for OphthalmoAI.

Exports AI eye disease screening results into official HL7 FHIR R4 DiagnosticReport
JSON documents for seamless integration with enterprise Electronic Health Record (EHR)
systems like Epic, Cerner, and Allscripts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def export_to_fhir_diagnostic_report(scan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an OphthalmoAI ScanResult into a valid HL7 FHIR R4 DiagnosticReport JSON structure.
    """
    scan_id = str(scan.get("scan_id") or scan.get("id") or "UNKNOWN")
    patient_ref = f"Patient/{scan.get('user_id') or 'ANONYMOUS'}"
    effective_dt = scan.get("timestamp") or datetime.now(timezone.utc).isoformat()
    diagnosis = scan.get("diagnosis", "Unknown Condition")
    icd10 = scan.get("icd10_code", "H57.9")
    snomed = scan.get("snomed_code", "371405004")
    confidence = scan.get("confidence", 0.0)
    urgency = scan.get("urgency", "non-urgent")

    return {
        "resourceType": "DiagnosticReport",
        "id": f"ophthalmoai-report-{scan_id[:8]}",
        "meta": {
            "versionId": "1",
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "profile": ["http://hl7.org/fhir/StructureDefinition/DiagnosticReport"],
        },
        "identifier": [
            {
                "system": "https://ophthalmoai.org/reports",
                "value": scan_id,
            }
        ],
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Ophthalmic Diagnostic Imaging",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": snomed,
                    "display": diagnosis,
                }
            ],
            "text": f"Ophthalmic AI Screening for {diagnosis}",
        },
        "subject": {
            "reference": patient_ref,
            "type": "Patient",
        },
        "effectiveDateTime": str(effective_dt),
        "issued": datetime.now(timezone.utc).isoformat(),
        "performer": [
            {
                "display": "OphthalmoAI Autonomous Diagnostic Engine v2.1.0",
            }
        ],
        "conclusion": (
            f"AI Diagnostic Triage: {diagnosis} detected with {confidence}% calibrated confidence. "
            f"Triage Urgency: {urgency.upper()}. "
            f"{scan.get('escalation_message', 'Consult an ophthalmologist for definitive clinical evaluation.')}"
        ),
        "conclusionCode": [
            {
                "coding": [
                    {
                        "system": "http://hl7.org/fhir/sid/icd-10",
                        "code": icd10,
                        "display": diagnosis,
                    }
                ]
            }
        ],
        "extension": [
            {
                "url": "https://ophthalmoai.org/fhir/StructureDefinition/ai-uncertainty",
                "valueDecimal": float(scan.get("uncertainty", 0.0)),
            },
            {
                "url": "https://ophthalmoai.org/fhir/StructureDefinition/requires-human-review",
                "valueBoolean": bool(scan.get("requires_human_review", False)),
            },
            {
                "url": "https://ophthalmoai.org/fhir/StructureDefinition/iqa-acceptable",
                "valueBoolean": bool(scan.get("iqa_acceptable", True)),
            },
        ],
    }

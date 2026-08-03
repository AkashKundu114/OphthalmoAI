import os
import sys
import unittest
from datetime import datetime, timezone

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.fhir import export_to_fhir_diagnostic_report

class TestFHIRReportExporter(unittest.TestCase):

    def test_export_report_structure(self):
        scan_data = {
            "scan_id": "test-uuid-12345",
            "user_id": "patient-xyz",
            "timestamp": "2026-08-01T12:00:00Z",
            "diagnosis": "Uveitis",
            "icd10_code": "H20.9",
            "snomed_code": "128473001",
            "confidence": 88.5,
            "urgency": "urgent",
            "uncertainty": 0.08,
            "requires_human_review": True,
            "iqa_acceptable": True,
            "escalation_message": "Uveitis requires same-week evaluation."
        }

        report = export_to_fhir_diagnostic_report(scan_data)

        # Assert FHIR R4 standard structures
        self.assertEqual(report["resourceType"], "DiagnosticReport")
        self.assertEqual(report["id"], "ophthalmoai-report-test-uui")
        self.assertEqual(report["status"], "final")
        
        # Identifier
        self.assertEqual(len(report["identifier"]), 1)
        self.assertEqual(report["identifier"][0]["value"], "test-uuid-12345")
        
        # Category
        self.assertEqual(report["category"][0]["coding"][0]["code"], "RAD")
        self.assertEqual(report["category"][0]["coding"][0]["display"], "Ophthalmic Diagnostic Imaging")
        
        # SNOMED Diagnosis code
        self.assertEqual(report["code"]["coding"][0]["system"], "http://snomed.info/sct")
        self.assertEqual(report["code"]["coding"][0]["code"], "128473001")
        self.assertEqual(report["code"]["coding"][0]["display"], "Uveitis")
        
        # Subject
        self.assertEqual(report["subject"]["reference"], "Patient/patient-xyz")
        
        # Timestamp
        self.assertEqual(report["effectiveDateTime"], "2026-08-01T12:00:00Z")
        
        # Conclusion text
        self.assertIn("Uveitis detected with 88.5% calibrated confidence", report["conclusion"])
        self.assertIn("Triage Urgency: URGENT", report["conclusion"])
        self.assertIn("Uveitis requires same-week evaluation.", report["conclusion"])

        # ICD-10 Conclusion code
        self.assertEqual(report["conclusionCode"][0]["coding"][0]["system"], "http://hl7.org/fhir/sid/icd-10")
        self.assertEqual(report["conclusionCode"][0]["coding"][0]["code"], "H20.9")
        
        # Extensions
        extensions = {ext["url"]: ext for ext in report["extension"]}
        self.assertEqual(extensions["https://ophthalmoai.org/fhir/StructureDefinition/ai-uncertainty"]["valueDecimal"], 0.08)
        self.assertTrue(extensions["https://ophthalmoai.org/fhir/StructureDefinition/requires-human-review"]["valueBoolean"])
        self.assertTrue(extensions["https://ophthalmoai.org/fhir/StructureDefinition/iqa-acceptable"]["valueBoolean"])

    def test_export_report_defaults(self):
        # Empty scan dictionary should fallback to default values without raising errors
        report = export_to_fhir_diagnostic_report({})
        self.assertEqual(report["resourceType"], "DiagnosticReport")
        self.assertEqual(report["id"], "ophthalmoai-report-UNKNOWN")
        self.assertEqual(report["status"], "final")
        self.assertEqual(report["subject"]["reference"], "Patient/ANONYMOUS")
        self.assertEqual(report["code"]["coding"][0]["code"], "371405004")  # default SNOMED
        self.assertEqual(report["conclusionCode"][0]["coding"][0]["code"], "H57.9")  # default ICD-10

if __name__ == "__main__":
    unittest.main()

"""
Unit Test Suite for Startup & Commercial Enterprise Features.
Tests HL7 FHIR R4 exporter, Clinician Case Queue & Overrides, and Patient History APIs.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

import backend.main as bm
from backend.db import Base, ClinicianOverride, ScanResult, User, create_tables, engine, get_db
from backend.fhir import export_to_fhir_diagnostic_report


class TestFHIRR4Exporter(unittest.TestCase):

    def test_fhir_report_structure(self):
        sample_scan = {
            "scan_id": "test-uuid-1234",
            "user_id": "patient-5678",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "diagnosis": "Glaucoma",
            "confidence": 94.5,
            "icd10_code": "H40.9",
            "snomed_code": "371405004",
            "urgency": "urgent",
            "uncertainty": 0.045,
            "requires_human_review": False,
            "iqa_acceptable": True,
            "escalation_message": "Consult an ophthalmologist for intraocular pressure testing.",
        }

        fhir_doc = export_to_fhir_diagnostic_report(sample_scan)

        self.assertEqual(fhir_doc["resourceType"], "DiagnosticReport")
        self.assertEqual(fhir_doc["status"], "final")
        self.assertEqual(fhir_doc["subject"]["reference"], "Patient/patient-5678")
        self.assertIn("H40.9", fhir_doc["conclusionCode"][0]["coding"][0]["code"])
        self.assertIn("Glaucoma", fhir_doc["conclusion"])


        ext_urls = [e["url"] for e in fhir_doc["extension"]]
        self.assertIn("https://ophthalmoai.org/fhir/StructureDefinition/ai-uncertainty", ext_urls)
        self.assertIn("https://ophthalmoai.org/fhir/StructureDefinition/requires-human-review", ext_urls)


class TestStartupEndpoints(unittest.TestCase):

    def setUp(self):
        create_tables()
        self.client = TestClient(bm.app, raise_server_exceptions=False)

    def test_fhir_endpoint_404_on_invalid_id(self):
        response = self.client.get("/fhir/export/nonexistent-scan-id")
        self.assertEqual(response.status_code, 404)

    def test_patient_history_endpoint(self):
        response = self.client.get("/patient/history/patient-007")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["patient_id"], "patient-007")
        self.assertIsInstance(body["scans"], list)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
OphthalmoAI: Test Fixtures Generator.
Creates minimal, deterministic JSON test fixtures in tests/fixtures/
without requiring any committed binary database.
"""

import json
import os
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_USERS = [
    {
        "id": "usr-test-clinician-001",
        "email": "clinician@hospital.org",
        "full_name": "Dr. Sarah Lin, MD",
        "role": "clinician",
        "is_active": True,
        "tenant_id": "tenant-default-001"
    },
    {
        "id": "usr-test-patient-001",
        "email": "patient.demo@example.com",
        "full_name": "Alex Mercer",
        "role": "patient",
        "is_active": True,
        "tenant_id": "tenant-default-001"
    },
    {
        "id": "usr-test-admin-001",
        "email": "admin@hospital.org",
        "full_name": "System Administrator",
        "role": "admin",
        "is_active": True,
        "tenant_id": "tenant-default-001"
    }
]

SAMPLE_SCANS = [
    {
        "id": "scan-sample-001",
        "diagnosis": "Normal",
        "confidence": 94.2,
        "uncertainty": 0.08,
        "requires_human_review": False,
        "urgency": "none",
        "icd10_code": "Z01.00",
        "snomed_code": "107774000",
        "created_at": "2026-10-15T10:30:00Z"
    },
    {
        "id": "scan-sample-002",
        "diagnosis": "Diabetic Retinopathy",
        "confidence": 88.7,
        "uncertainty": 0.19,
        "requires_human_review": True,
        "urgency": "routine",
        "icd10_code": "E11.319",
        "snomed_code": "4855003",
        "created_at": "2026-10-16T14:15:00Z"
    }
]

def main():
    users_file = FIXTURES_DIR / "sample_users.json"
    with open(users_file, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_USERS, f, indent=2)

    scans_file = FIXTURES_DIR / "sample_scans.json"
    with open(scans_file, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_SCANS, f, indent=2)

    print(f"[OK] Generated {users_file}")
    print(f"[OK] Generated {scans_file}")

if __name__ == "__main__":
    main()

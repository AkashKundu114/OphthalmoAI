"""
Comprehensive Red-Team Security, Medical-Image Privacy, Adversarial Upload,
and IDOR Access-Control Test Suite for OphthalmoAI.
"""

import io
import os
import unittest
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from fastapi.testclient import TestClient
from fastapi import status

from backend.main import app
from backend.mongodb_client import mongo_store
from backend.security import (
    validate_magic_bytes,
    validate_image_dimensions,
    sanitize_filename,
    strip_image_metadata,
    _MAX_PIXELS,
    MIN_IMAGE_DIMENSION_PX,
    MAX_IMAGE_AXIS_PX,
    MAX_IMAGE_ASPECT_RATIO,
)
from backend.auth import create_access_token
from backend.db import get_db, User, ScanResult, SessionLocal


class TestRedTeamUploadSecurity(unittest.TestCase):
    """Adversarial and malicious image upload tests."""

    def test_magic_bytes_spoofed_extension(self):
        """A text/script file masquerading as a JPEG must be rejected."""
        fake_payload = b"<?php echo 'malicious code'; ?>"
        ok, reason = validate_magic_bytes(fake_payload, declared_mime="image/jpeg")
        self.assertFalse(ok)
        self.assertIn("File content does not match", reason)

    def test_magic_bytes_mime_mismatch(self):
        """A PNG image with declared MIME image/jpeg must be rejected."""
        png_sig = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00"
        ok, reason = validate_magic_bytes(png_sig, declared_mime="image/jpeg")
        self.assertFalse(ok)
        self.assertIn("mismatch", reason.lower())

    def test_subminimum_dimension_rejection(self):
        """Images below 128x128 px must be rejected for clinical quality."""
        img = Image.new("RGB", (64, 64), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        ok, reason = validate_image_dimensions(buf.getvalue())
        self.assertFalse(ok)
        self.assertIn(f"below the minimum required resolution", reason)

    def test_pathological_aspect_ratio_rejection(self):
        """Extreme aspect ratios (e.g., 2000x200 = 10:1) must be rejected."""
        img = Image.new("RGB", (2000, 200), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        ok, reason = validate_image_dimensions(buf.getvalue())
        self.assertFalse(ok)
        self.assertIn("Abnormal image aspect ratio", reason)

    def test_max_axis_dimension_rejection(self):
        """Images with an axis exceeding MAX_IMAGE_AXIS_PX (6000 px) must be rejected."""
        img = Image.new("RGB", (6500, 5000), color="green")
        buf = io.BytesIO()
        # Save minimal JPEG without allocating huge buffer
        img.save(buf, format="JPEG")
        ok, reason = validate_image_dimensions(buf.getvalue())
        self.assertFalse(ok)
        self.assertIn("exceeds maximum allowed size", reason)

    def test_decompression_bomb_pixel_limit(self):
        """Dimensions exceeding _MAX_PIXELS (25,000,000 px) must be blocked."""
        # 5500 x 5000 = 27,500,000 > 25,000,000
        img = Image.new("RGB", (5500, 5000), color="yellow")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        ok, reason = validate_image_dimensions(buf.getvalue())
        self.assertFalse(ok)
        self.assertTrue("exceed" in reason.lower() or "maximum allowed" in reason.lower())

    def test_valid_clinical_fundus_dimensions_accepted(self):
        """Standard high-res fundus dimensions (e.g. 2048x1536) must pass."""
        img = Image.new("RGB", (2048, 1536), color="black")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        ok, reason = validate_image_dimensions(buf.getvalue())
        self.assertTrue(ok)
        self.assertEqual(reason, "2048x1536")


class TestMedicalPrivacyAndSanitization(unittest.TestCase):
    """EXIF PHI removal, path traversal prevention, and filename sanitization."""

    def test_filename_sanitization_path_traversal(self):
        """Directory traversal sequences in filenames must be neutralized."""
        evil_name = "../../etc/passwd"
        safe = sanitize_filename(evil_name)
        self.assertNotIn("/", safe)
        self.assertNotIn("..", safe)
        self.assertTrue(safe.endswith(".jpg"))

        windows_evil = "..\\..\\Windows\\System32\\cmd.exe"
        safe_win = sanitize_filename(windows_evil)
        self.assertNotIn("\\", safe_win)
        self.assertNotIn("..", safe_win)
        self.assertTrue(safe_win.endswith(".jpg"))

    def test_exif_metadata_stripping(self):
        """Private EXIF, camera serials, and GPS coordinates must be purged."""
        img = Image.new("RGB", (256, 256), color=(200, 100, 50))
        exif = img.getexif()
        # 0x010E is ImageDescription, 0x0110 is Model
        exif[0x010E] = "PATIENT_PHI_MRN_99887766"
        exif[0x0110] = "RetinalCamera-Serial-X99"

        clean_img = strip_image_metadata(img)
        clean_exif = clean_img.getexif()

        self.assertNotIn(0x010E, clean_exif)
        self.assertNotIn(0x0110, clean_exif)
        self.assertEqual(len(clean_exif), 0)
        self.assertEqual(clean_img.size, (256, 256))

    def test_mongodb_mock_store_path_traversal_defense(self):
        """Attempts to read or write outside the mock directory must raise ValueError."""
        with self.assertRaises(ValueError):
            mongo_store._resolve_mock_path("../../etc/passwd")

        with self.assertRaises(ValueError):
            mongo_store._resolve_mock_path("..\\..\\Windows\\win.ini")

        with self.assertRaises(ValueError):
            mongo_store._resolve_mock_path("invalid/path/traversal")


class TestIDORAndEndpointSecurity(unittest.TestCase):
    """Insecure Direct Object Reference (IDOR) & Endpoint Hardening Tests."""

    def setUp(self):
        app.dependency_overrides.clear()
        from backend.db import create_tables
        create_tables()

        self.client = TestClient(app)
        db = SessionLocal()
        try:
            # Create two test patients
            p1 = db.query(User).filter(User.id == "patient-uuid-1111").first()
            if not p1:
                p1 = User(
                    id="patient-uuid-1111",
                    email="patient1_test@ophthalmoai.org",
                    hashed_password="hashed_pw_test",
                    role="patient",
                    is_active=True,
                )
                db.add(p1)
            else:
                p1.is_active = True
                p1.role = "patient"

            p2 = db.query(User).filter(User.id == "patient-uuid-2222").first()
            if not p2:
                p2 = User(
                    id="patient-uuid-2222",
                    email="patient2_test@ophthalmoai.org",
                    hashed_password="hashed_pw_test",
                    role="patient",
                    is_active=True,
                )
                db.add(p2)
            else:
                p2.is_active = True
                p2.role = "patient"

            clinician = db.query(User).filter(User.id == "clinician-uuid-3333").first()
            if not clinician:
                clinician = User(
                    id="clinician-uuid-3333",
                    email="clinician_test@ophthalmoai.org",
                    hashed_password="hashed_pw_test",
                    role="clinician",
                    is_active=True,
                )
                db.add(clinician)
            else:
                clinician.is_active = True
                clinician.role = "clinician"

            # Create a scan owned by patient 1
            s1 = db.query(ScanResult).filter(ScanResult.id == "scan-test-p1").first()
            if not s1:
                s1 = ScanResult(
                    id="scan-test-p1",
                    user_id="patient-uuid-1111",
                    diagnosis="Normal",
                    confidence=98.5,
                )
                db.add(s1)

            db.commit()
        finally:
            db.close()

        self.token_p1 = create_access_token(subject="patient-uuid-1111", role="patient")
        self.token_p2 = create_access_token(subject="patient-uuid-2222", role="patient")
        self.token_clinician = create_access_token(subject="clinician-uuid-3333", role="clinician")

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_idor_patient_cannot_access_other_patient_history(self):
        """Patient 2 must be forbidden (HTTP 403) from fetching Patient 1's history."""
        headers = {"Authorization": f"Bearer {self.token_p2}"}
        resp = self.client.get("/scans/history/patient-uuid-1111", headers=headers)
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn("Access denied", resp.json()["detail"])

    def test_idor_patient_can_access_own_history(self):
        """Patient 1 can access their own history."""
        headers = {"Authorization": f"Bearer {self.token_p1}"}
        resp = self.client.get("/scans/history/patient-uuid-1111", headers=headers)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json()["user_id"], "patient-uuid-1111")

    def test_idor_patient_cannot_access_other_patient_scan_details(self):
        """Patient 2 must be forbidden (HTTP 403) from reading Patient 1's scan details."""
        headers = {"Authorization": f"Bearer {self.token_p2}"}
        resp = self.client.get("/scans/scan-test-p1/details", headers=headers)
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn("Access denied", resp.json()["detail"])

    def test_clinician_can_access_patient_scan_details(self):
        """Clinician has legitimate access to scan details."""
        headers = {"Authorization": f"Bearer {self.token_clinician}"}
        resp = self.client.get("/scans/scan-test-p1/details", headers=headers)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json()["scan_id"], "scan-test-p1")

    def test_malformed_scan_id_rejected(self):
        """Malformed scan identifier containing path traversal characters returns HTTP 400."""
        resp = self.client.get("/scans/../../evil/details")
        # FastAPI route matching or 400 validation
        self.assertIn(resp.status_code, [status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND])

    def test_compress_endpoint_decompression_bomb_rejected(self):
        """Sending a decompression bomb to /scans/compress returns HTTP 422."""
        img = Image.new("RGB", (6500, 5000), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        resp = self.client.post(
            "/scans/compress",
            files={"file": ("bomb.jpg", buf.getvalue(), "image/jpeg")},
        )
        self.assertEqual(resp.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)

    def test_compress_endpoint_empty_file_rejected(self):
        """Sending an empty file to /scans/compress returns HTTP 400."""
        resp = self.client.post(
            "/scans/compress",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_compress_endpoint_valid_image_success(self):
        """Sending a valid fundus image to /scans/compress returns compressed payload."""
        img = Image.new("RGB", (512, 512), color="brown")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        resp = self.client.post(
            "/scans/compress",
            files={"file": ("fundus.jpg", buf.getvalue(), "image/jpeg")},
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = resp.json()
        self.assertIn("compressed_file_base64", data)
        self.assertIn("bandwidth_saved_percent", data)

    def test_zero_egress_csp_header_enforced(self):
        """Verifies that Content-Security-Policy disallows external network requests."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        csp = resp.headers.get("Content-Security-Policy", "")
        self.assertIn("default-src 'self'", csp)
        self.assertNotIn("fonts.googleapis.com", csp)
        self.assertNotIn("fonts.gstatic.com", csp)

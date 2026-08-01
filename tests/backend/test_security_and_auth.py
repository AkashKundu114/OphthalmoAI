"""
Unit tests for security, authentication, password hashing, JWT operations,
IP anonymization, magic bytes verification, and security headers middleware.
"""
from __future__ import annotations

import os
import unittest
from datetime import timedelta
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import (
    authenticate_user,
    create_access_token,
    decode_token,
    hash_password,
    require_role,
    revoke_token,
    verify_password,
)
from backend.security import (
    SecurityHeadersMiddleware,
    anonymise_ip,
    validate_image_dimensions,
    validate_magic_bytes,
)
from backend.validators import (
    sanitise_chat_message,
    validate_email,
    validate_password_strength,
)


class TestSecurityAndAuth(unittest.TestCase):

    def test_password_hash_and_verify(self):
        pwd = "SecureP@ssw0rd2026!"
        hashed = hash_password(pwd)
        self.assertNotEqual(pwd, hashed)
        self.assertTrue(verify_password(pwd, hashed))
        self.assertFalse(verify_password("WrongPassword123!", hashed))

    def test_password_strength_validation(self):
        valid, err = validate_password_strength("Short1!")
        self.assertFalse(valid)
        self.assertIn("at least 12", err.lower())

        valid_strong, err_strong = validate_password_strength("Complex!P@ss2026")
        self.assertTrue(valid_strong, msg=f"Password failed with: {err_strong}")
        self.assertEqual(err_strong, "OK")

    def test_email_validation_and_normalization(self):
        valid, email = validate_email("  Test.User@Example.COM  ")
        self.assertTrue(valid)
        self.assertEqual(email, "test.user@example.com")

        invalid, err = validate_email("invalid-email-string")
        self.assertFalse(invalid)

    def test_jwt_create_and_decode(self):
        token = create_access_token(subject="user_123", role="clinician")
        payload = decode_token(token)
        self.assertEqual(payload["sub"], "user_123")
        self.assertEqual(payload["role"], "clinician")

    def test_jwt_revocation(self):
        token = create_access_token(subject="user_revoke", role="patient")
        revoke_token(token)
        with self.assertRaises(Exception):
            decode_token(token)

    def test_ip_anonymization(self):
        ipv4_anon = anonymise_ip("192.168.1.100")
        self.assertEqual(ipv4_anon, "192.168.1.0")

        ipv6_anon = anonymise_ip("2001:db8:85a3:8d3d:370:7334:712e:7a36")
        self.assertTrue(ipv6_anon.endswith("::"))

        self.assertIsNone(anonymise_ip(None))

    def test_magic_bytes_jpeg_validation(self):
        jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
        ok, mime = validate_magic_bytes(jpeg_header, declared_mime="image/jpeg")
        self.assertTrue(ok)
        self.assertEqual(mime, "image/jpeg")

    def test_magic_bytes_rejection(self):
        fake_file = b"MALICIOUS_EXEC_PAYLOAD"
        ok, err = validate_magic_bytes(fake_file, declared_mime="image/jpeg")
        self.assertFalse(ok)

    def test_security_headers_middleware(self):
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware, is_production=True)

        @app.get("/test")
        def ping():
            return {"status": "ok"}

        client = TestClient(app)
        res = client.get("/test")
        self.assertEqual(res.headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertIn("Strict-Transport-Security", res.headers)
        self.assertEqual(res.headers.get("Server"), "OphthalmoAI")

    def test_chat_message_sanitization(self):
        ok, msg = sanitise_chat_message("   Hello, AI doctor!   ")
        self.assertTrue(ok)
        self.assertEqual(msg, "Hello, AI doctor!")

        invalid, err = sanitise_chat_message("<script>alert('xss')</script>")
        self.assertFalse(invalid)


if __name__ == "__main__":
    unittest.main()

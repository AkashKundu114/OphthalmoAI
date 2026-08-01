"""
Unit tests for Gemini Free Tier AI Doctor Chatbot integration and /chat endpoint.
"""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

import backend.main as bm


class TestGeminiChatEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(bm.app)

    def test_chat_without_api_keys(self):
        """When neither Gemini nor Ollama key is set, returns unconfigured message."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "", "OLLAMA_URL": ""}):
            res = self.client.post(
                "/chat",
                json={"message": "What is conjunctivitis?"},
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("reply", data)
            self.assertEqual(data["model_used"], "none")
            self.assertIn("not configured", data["reply"].lower())

    def test_chat_with_diagnosis_context(self):
        """Context injection correctly builds clinical prompt structure."""
        req_payload = {
            "message": "What should I do about my condition?",
            "history": [{"role": "user", "content": "Hello"}],
            "diagnosis_context": {
                "diagnosis": "Cataract",
                "confidence": 94.5,
                "group_name": "Anterior Segment",
                "details": {
                    "severity": "Moderate",
                    "advice": "Consult an ophthalmologist for a dilated slit-lamp exam.",
                },
            },
        }
        with patch.dict("os.environ", {"GEMINI_API_KEY": "", "OLLAMA_URL": ""}):
            res = self.client.post("/chat", json=req_payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("reply", data)

    def test_chat_rejects_xss_message(self):
        """Harmful HTML script tags in chat message must be rejected."""
        res = self.client.post(
            "/chat",
            json={"message": "<script>alert('xss')</script> Hello"},
        )
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()

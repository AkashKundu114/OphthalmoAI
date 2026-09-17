"""
Unit tests for Sarvam AI integration and Hybrid Clinical Voice/Language Triage in OphthalmoAI.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

# Mock torch & torchvision if not installed (e.g. Python 3.14 / lightweight test runners)
for mod in [
    "torch", "torch.nn", "torch.nn.functional",
    "torchvision", "torchvision.models", "torchvision.transforms",
    "pytorch_grad_cam", "pytorch_grad_cam.utils.image", "pytorch_grad_cam.utils.model_targets"
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Ensure test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-unit-testing-32-chars-long"

from fastapi.testclient import TestClient
from backend.main import app
from backend.sarvam_service import (
    transcribe_speech,
    translate_text,
    text_to_speech,
    is_sarvam_available,
    SUPPORTED_INDIC_LANGUAGES,
)
from backend.validators import detect_medical_emergency


class TestSarvamService(unittest.IsolatedAsyncioTestCase):

    def test_supported_languages(self):
        """Ensure all primary Indian languages are supported."""
        self.assertIn("hi-IN", SUPPORTED_INDIC_LANGUAGES)
        self.assertIn("bn-IN", SUPPORTED_INDIC_LANGUAGES)
        self.assertIn("ta-IN", SUPPORTED_INDIC_LANGUAGES)
        self.assertIn("te-IN", SUPPORTED_INDIC_LANGUAGES)
        self.assertIn("mr-IN", SUPPORTED_INDIC_LANGUAGES)

    @patch.dict(os.environ, {"SARVAM_API_KEY": ""})
    def test_is_sarvam_available_false(self):
        self.assertFalse(is_sarvam_available())

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key-123"})
    def test_is_sarvam_available_true(self):
        self.assertTrue(is_sarvam_available())

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key-123"})
    @patch("httpx.AsyncClient.post")
    async def test_transcribe_speech_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "transcript": "मेरी आँख में जाला जैसा दिखता है",
            "language_code": "hi-IN",
        }
        mock_post.return_value = mock_resp

        ok, transcript, lang = await transcribe_speech(
            audio_bytes=b"fake-audio-bytes-123",
            language_code="hi-IN",
        )
        self.assertTrue(ok)
        self.assertEqual(transcript, "मेरी आँख में जाला जैसा दिखता है")
        self.assertEqual(lang, "hi-IN")

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key-123"})
    @patch("httpx.AsyncClient.post")
    async def test_translate_text_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "translated_text": "I see cobwebs in front of my eyes",
        }
        mock_post.return_value = mock_resp

        ok, translated = await translate_text(
            text="मेरी आँख में जाला जैसा दिखता है",
            source_lang="hi-IN",
            target_lang="en-IN",
        )
        self.assertTrue(ok)
        self.assertEqual(translated, "I see cobwebs in front of my eyes")

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key-123"})
    @patch("httpx.AsyncClient.post")
    async def test_text_to_speech_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "audios": ["UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="],
        }
        mock_post.return_value = mock_resp

        ok, audio_b64, err = await text_to_speech(
            text="कृपया तुरंत अपने नजदीकी नेत्र विशेषज्ञ से संपर्क करें।",
            target_lang="hi-IN",
        )
        self.assertTrue(ok)
        self.assertIsNotNone(audio_b64)
        self.assertIsNone(err)


class TestHybridClinicalEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_emergency_interceptor_indian_hotlines(self):
        """Verify Indian emergency escalation triggers with 112 / 108."""
        is_emerg, msg = detect_medical_emergency("I have sudden loss of vision in my left eye")
        self.assertTrue(is_emerg)
        self.assertIn("112", msg)
        self.assertIn("108", msg)

        # Test Indic transliterated emergency
        is_emerg_indic, msg_indic = detect_medical_emergency("achanak dikhna band ho gaya")
        self.assertTrue(is_emerg_indic)
        self.assertIn("112", msg_indic)

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-sarvam-key", "GEMINI_API_KEY": "test-gemini-key"})
    @patch("backend.main.translate_text")
    @patch("backend.main.text_to_speech")
    def test_chat_endpoint_indic_translation_hybrid(
        self, mock_tts, mock_trans
    ):
        """Verify Indic translation and TTS generation in /chat."""
        mock_trans.side_effect = [
            (True, "What are the early signs of cataracts?"),  # query translation
            (True, "मोतियाबिंद के शुरुआती लक्षणों में दृष्टि का धुंधलापन शामिल है।"),  # reply translation
        ]
        mock_tts.return_value = (True, "mock-audio-base64", None)

        mock_genai = MagicMock()
        mock_session = AsyncMock()
        mock_chat_res = MagicMock()
        mock_chat_res.text = "Early signs of cataracts include blurred vision."
        mock_session.send_message_async.return_value = mock_chat_res

        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_session
        mock_genai.GenerativeModel.return_value = mock_model

        with patch.dict(sys.modules, {"google.generativeai": mock_genai}):
            resp = self.client.post(
                "/chat",
                json={
                    "message": "मोतियाबिंद के शुरुआती लक्षण क्या हैं?",
                    "history": [],
                    "language": "hi-IN",
                    "audio_output_requested": True,
                },
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["language"], "hi-IN")
            self.assertIn("मोतियाबिंद", data["reply"])
            self.assertEqual(data["audio_base64"], "mock-audio-base64")

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-sarvam-key", "GEMINI_API_KEY": "test-gemini-key"})
    @patch("backend.main.transcribe_speech")
    @patch("backend.main.translate_text")
    @patch("backend.main.text_to_speech")
    def test_chat_voice_endpoint(self, mock_tts, mock_trans, mock_stt):
        mock_stt.return_value = (True, "आंखों में धुंधलापन है", "hi-IN")
        mock_trans.side_effect = [
            (True, "There is blurriness in the eyes"),
            (True, "कृपया निकटतम नेत्र केंद्र में जांच कराएं।"),
        ]
        mock_tts.return_value = (True, "voice-reply-audio-b64", None)

        mock_genai = MagicMock()
        mock_session = AsyncMock()
        mock_chat_res = MagicMock()
        mock_chat_res.text = "Please get an eye examination."
        mock_session.send_message_async.return_value = mock_chat_res
        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_session
        mock_genai.GenerativeModel.return_value = mock_model

        with patch.dict(sys.modules, {"google.generativeai": mock_genai}):
            files = {
                "file": (
                    "test_voice.webm",
                    b"x" * 250,
                    "audio/webm",
                )
            }
            resp = self.client.post(
                "/chat/voice",
                data={"language": "hi-IN", "audio_output_requested": "true"},
                files=files,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["transcript"], "आंखों में धुंधलापन है")
            self.assertIn("audio_base64", data)


    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-sarvam-key"})
    @patch("backend.main.text_to_speech")
    def test_tts_endpoint(self, mock_tts):
        mock_tts.return_value = (True, "wav-base64-audio-payload", None)

        resp = self.client.post(
            "/chat/tts",
            json={
                "text": "नमस्ते, आपकी रेटिना सामान्य प्रतीत होती है।",
                "language": "hi-IN",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["audio_base64"], "wav-base64-audio-payload")
        self.assertEqual(data["language"], "hi-IN")

    @patch.dict(os.environ, {"SARVAM_API_KEY": ""})
    def test_tts_endpoint_unavailable_without_key(self):
        resp = self.client.post(
            "/chat/tts",
            json={"text": "Hello", "language": "en-IN"},
        )
        self.assertEqual(resp.status_code, 503)


if __name__ == "__main__":
    unittest.main()

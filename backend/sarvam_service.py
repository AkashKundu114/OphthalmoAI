"""
Sarvam AI Service Module for OphthalmoAI.

Provides Indic speech-to-text (Saaras:v1), translation (Mayura:v1),
and text-to-speech (Bulbul:v1) capabilities tailored for Indian clinics,
Primary Health Centres (PHCs), and visually impaired patient accessibility.
"""

import os
import logging
from typing import Optional, Tuple
import httpx

logger = logging.getLogger("ophthalmoai.sarvam")

SARVAM_API_BASE = "https://api.sarvam.ai"

# Supported Indic languages mapping
SUPPORTED_INDIC_LANGUAGES = {
    "en-IN": {"name": "English (India)", "native": "English"},
    "hi-IN": {"name": "Hindi", "native": "हिन्दी"},
    "bn-IN": {"name": "Bengali", "native": "বাংলা"},
    "ta-IN": {"name": "Tamil", "native": "தமிழ்"},
    "te-IN": {"name": "Telugu", "native": "తెలుగు"},
    "mr-IN": {"name": "Marathi", "native": "मराठी"},
    "gu-IN": {"name": "Gujarati", "native": "ગુજરાતી"},
    "kn-IN": {"name": "Kannada", "native": "ಕನ್ನಡ"},
    "ml-IN": {"name": "Malayalam", "native": "മലയാളം"},
    "pa-IN": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ"},
    "od-IN": {"name": "Odia", "native": "ଓଡ଼ିଆ"},
}

DEFAULT_SPEAKER = "meera"


def get_sarvam_api_key() -> str:
    """Retrieve the Sarvam API subscription key from environment."""
    return os.getenv("SARVAM_API_KEY", "").strip()


def is_sarvam_available() -> bool:
    """Return True if Sarvam API key is configured."""
    return bool(get_sarvam_api_key())


async def transcribe_speech(
    audio_bytes: bytes,
    filename: str = "recording.webm",
    content_type: str = "audio/webm",
    language_code: str = "hi-IN",
    timeout: float = 30.0,
) -> Tuple[bool, str, str]:
    """
    Transcribe audio bytes to text using Sarvam Saaras STT API.

    Returns:
        (success: bool, transcript_or_error: str, detected_language: str)
    """
    api_key = get_sarvam_api_key()
    if not api_key:
        return False, "Sarvam API key not configured.", language_code

    headers = {
        "api-subscription-key": api_key,
    }

    files = {
        "file": (filename, audio_bytes, content_type),
    }
    data = {
        "model": "saaras:v1",
        "language_code": language_code if language_code in SUPPORTED_INDIC_LANGUAGES else "unknown",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{SARVAM_API_BASE}/speech-to-text",
                headers=headers,
                data=data,
                files=files,
            )
            if resp.status_code == 200:
                result = resp.json()
                transcript = result.get("transcript", "").strip()
                detected_lang = result.get("language_code", language_code)
                return True, transcript, detected_lang
            else:
                err_msg = f"Sarvam STT failed ({resp.status_code}): {resp.text}"
                logger.error("sarvam.stt_error", extra={"error": err_msg})
                return False, err_msg, language_code
    except Exception as exc:
        logger.error("sarvam.stt_exception", extra={"error": str(exc)})
        return False, f"Speech transcription error: {str(exc)}", language_code


async def translate_text(
    text: str,
    source_lang: str = "hi-IN",
    target_lang: str = "en-IN",
    mode: str = "formal",
    timeout: float = 20.0,
) -> Tuple[bool, str]:
    """
    Translate text between Indic languages and English using Sarvam Mayura Translate API.

    Returns:
        (success: bool, translated_text: str)
    """
    if not text or not text.strip():
        return True, ""

    # No translation needed if source and target match or both are English
    if source_lang == target_lang or (source_lang.startswith("en") and target_lang.startswith("en")):
        return True, text

    api_key = get_sarvam_api_key()
    if not api_key:
        return False, "Sarvam API key not configured."

    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": api_key,
    }

    payload = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "speaker_gender": "Female",
        "mode": mode,
        "model": "mayura:v1",
        "enable_preprocessing": True,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{SARVAM_API_BASE}/translate",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 200:
                result = resp.json()
                translated = result.get("translated_text", text)
                return True, translated
            else:
                err_msg = f"Sarvam Translate failed ({resp.status_code}): {resp.text}"
                logger.error("sarvam.translate_error", extra={"error": err_msg})
                return False, err_msg
    except Exception as exc:
        logger.error("sarvam.translate_exception", extra={"error": str(exc)})
        return False, f"Translation error: {str(exc)}"


async def text_to_speech(
    text: str,
    target_lang: str = "hi-IN",
    speaker: str = DEFAULT_SPEAKER,
    timeout: float = 30.0,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Convert text to natural Indic speech using Sarvam Bulbul TTS API.

    Returns:
        (success: bool, base64_audio_wav: Optional[str], error_message: Optional[str])
    """
    if not text or not text.strip():
        return False, None, "Empty text for TTS."

    api_key = get_sarvam_api_key()
    if not api_key:
        return False, None, "Sarvam API key not configured."

    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": api_key,
    }

    # Bulbul supports inputs up to ~500 chars per entry. Clean and trim safely.
    clean_text = text.replace("*", "").replace("#", "").replace("`", "").strip()
    if len(clean_text) > 480:
        cut = clean_text[:480].rfind(".")
        if cut == -1 or cut < 200:
            cut = clean_text[:480].rfind(" ")
        clean_text = (clean_text[:cut] + "...") if cut > 0 else clean_text[:480]

    payload = {
        "inputs": [clean_text],
        "target_language_code": target_lang if target_lang in SUPPORTED_INDIC_LANGUAGES else "hi-IN",
        "speaker": speaker,
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.5,
        "speech_sample_rate": 22050,
        "enable_preprocessing": True,
        "model": "bulbul:v1",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{SARVAM_API_BASE}/text-to-speech",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 200:
                result = resp.json()
                audios = result.get("audios", [])
                if audios:
                    return True, audios[0], None
                return False, None, "No audio generated in Sarvam response."
            else:
                err_msg = f"Sarvam TTS failed ({resp.status_code}): {resp.text}"
                logger.error("sarvam.tts_error", extra={"error": err_msg})
                return False, None, err_msg
    except Exception as exc:
        logger.error("sarvam.tts_exception", extra={"error": str(exc)})
        return False, None, f"TTS error: {str(exc)}"

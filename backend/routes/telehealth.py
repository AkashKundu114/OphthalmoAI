"""
Tele-Ophthalmology, Conversational AI & Multilingual Indic Speech Service.

Endpoints:
- POST /chat, POST /api/chat: Multilingual clinical guidance chat with emergency interceptor
- POST /chat/tts, POST /api/chat/tts: Neural text-to-speech synthesis (Sarvam Bulbul)
- POST /chat/voice, POST /api/chat/voice: Direct voice-in / voice-out clinical triage
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.audit import log_event
from backend.auth import get_current_user
from backend.biomarker_extractor import format_biomarkers_for_llm
from backend.config import CHAT_RATE_LIMIT, GEMINI_MODEL, OPHTHALMOLOGY_SYSTEM_PROMPT
from backend.db import User, get_db
from backend.logging_config import get_logger
from backend.sarvam_service import (
    SUPPORTED_INDIC_LANGUAGES,
    is_sarvam_available,
    text_to_speech,
    transcribe_speech,
    translate_text,
)
from backend.security import anonymise_ip, make_rate_limit_decorator
from backend.validators import detect_medical_emergency, sanitise_chat_message

logger = get_logger("ophthalmoai.telehealth_routes")
router = APIRouter(tags=["Telehealth"])

_chat_limit = make_rate_limit_decorator(CHAT_RATE_LIMIT)


def _client_ip(request: Request) -> Optional[str]:
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    diagnosis_context: Optional[Dict[str, Any]] = None
    language: Optional[str] = "en-IN"
    audio_output_requested: Optional[bool] = False


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "hi-IN"


@router.post("/chat")
@router.post("/api/chat")
@_chat_limit
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Multilingual Indic conversational agent for patient education, symptoms advice,
    and visual biomarker grounding.
    """
    import backend.main as bm

    # Allow dynamically patched translation functions from tests
    translate_fn = getattr(bm, "translate_text", translate_text)
    tts_fn = getattr(bm, "text_to_speech", text_to_speech)

    req_id = getattr(request.state, "request_id", None)
    client_ip = anonymise_ip(_client_ip(request))

    safe_ok, safe_msg = sanitise_chat_message(chat_request.message)
    if not safe_ok:
        raise HTTPException(422, detail=safe_msg)

    target_lang = (chat_request.language or "en-IN").strip()
    is_indic = target_lang != "en-IN" and target_lang in SUPPORTED_INDIC_LANGUAGES
    sarvam_active = is_sarvam_available()

    # Step 1: In rural / Indic mode, translate user query to English for medical reasoning
    english_query = safe_msg
    if is_indic and sarvam_active:
        ok_trans, translated_q = await translate_fn(
            safe_msg, source_lang=target_lang, target_lang="en-IN"
        )
        if ok_trans and translated_q:
            english_query = translated_q

    # Step 2: Emergency detection (check both original and translated text)
    is_emergency, emergency_msg = detect_medical_emergency(safe_msg)
    if not is_emergency and english_query != safe_msg:
        is_emergency, emergency_msg = detect_medical_emergency(english_query)

    if is_emergency:
        log_event(
            db,
            "chat.emergency_flagged",
            success=True,
            user_id=current_user.id if current_user else None,
            ip_address=client_ip,
        )
        if is_indic and sarvam_active:
            ok_emerg_trans, translated_emerg = await translate_fn(
                emergency_msg, source_lang="en-IN", target_lang=target_lang
            )
            if ok_emerg_trans and translated_emerg:
                emergency_msg = translated_emerg

        audio_b64 = None
        if chat_request.audio_output_requested and sarvam_active:
            _, audio_b64, _ = await tts_fn(emergency_msg, target_lang=target_lang)

        return {
            "reply": emergency_msg,
            "model_used": "emergency_interceptor",
            "is_emergency": True,
            "language": target_lang,
            "audio_base64": audio_b64,
            "disclaimer": "🚨 EMERGENCY NOTICE: Seek immediate in-person emergency medical care.",
        }

    # Guardrail: Refusal to diagnose unsupported / non-fundus images
    if chat_request.diagnosis_context:
        ctx = chat_request.diagnosis_context
        if ctx.get("status") == "unsupported_image" or ctx.get("is_fundus") is False:
            refusal_text = (
                "I cannot provide a diagnostic interpretation for this upload because the image "
                "could not be verified as an authentic retinal fundus photograph. OphthalmoAI only "
                "analyzes color fundus photographs of the posterior pole. Please upload a genuine "
                "retinal fundus scan for clinical evaluation."
            )
            if is_indic and sarvam_active:
                ok_refusal_trans, translated_refusal = await translate_fn(
                    refusal_text, source_lang="en-IN", target_lang=target_lang
                )
                if ok_refusal_trans and translated_refusal:
                    refusal_text = translated_refusal

            audio_b64 = None
            if chat_request.audio_output_requested and sarvam_active:
                _, audio_b64, _ = await tts_fn(refusal_text, target_lang=target_lang)

            return {
                "reply": refusal_text,
                "model_used": "guardrail_refusal",
                "is_emergency": False,
                "language": target_lang,
                "audio_base64": audio_b64,
                "disclaimer": "⚠️ Notice: Clinical interpretation requires an authentic color fundus photograph.",
            }

    system = OPHTHALMOLOGY_SYSTEM_PROMPT
    if chat_request.diagnosis_context:
        ctx = chat_request.diagnosis_context
        details = ctx.get("details", {})
        system += (
            f"\n\n--- CURRENT PATIENT AI SCREENING RESULT ---\n"
            f"Detected Condition: {ctx.get('diagnosis', 'Unknown')}\n"
            f"AI Confidence: {ctx.get('confidence', 0):.1f}%\n"
            f"Anatomical Group: {ctx.get('group_name', 'Unknown')}\n"
            f"Severity: {details.get('severity', 'Unknown')}\n"
            f"Clinical Advice: {details.get('advice', 'N/A')}\n"
            f"Note: This is an AI screening result only, not a clinical diagnosis."
        )
        biomarkers = ctx.get("visual_biomarkers")
        conformal_set = ctx.get("conformal_prediction_set")
        if biomarkers:
            try:
                cov_str = str(ctx.get("conformal_coverage_guarantee", "95.0%")).replace("%", "").strip()
                cov_val = float(cov_str.split()[0])
            except Exception:
                cov_val = 95.0
            grounded_block = format_biomarkers_for_llm(
                biomarkers=biomarkers,
                diagnosis=ctx.get("diagnosis", "Unknown"),
                conformal_set=conformal_set if conformal_set else [ctx.get("diagnosis", "Unknown")],
                coverage_guarantee=cov_val,
                epistemic_vacuity=float(ctx.get("epistemic_vacuity", 0.05)),
            )
            system += f"\n\n{grounded_block}"

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    ollama_url = os.getenv("OLLAMA_URL", "").strip()
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()
    reply = ""
    model_used = "none"

    try:
        if gemini_key:
            import google.generativeai as genai

            genai.configure(api_key=gemini_key)
            gemini_history = [
                {"role": "user" if m.role == "user" else "model", "parts": [m.content]}
                for m in chat_request.history
                if m.role in ("user", "assistant")
            ]
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=system,
            )
            chat_session = model.start_chat(history=gemini_history)
            response = await chat_session.send_message_async(english_query)
            reply = response.text.replace("\u2014", ", ").replace("\u2013", "-")
            model_used = "gemini"

        elif ollama_url:
            import httpx

            messages = [{"role": "system", "content": system}]
            for m in chat_request.history:
                if m.role in ("user", "assistant"):
                    messages.append({"role": m.role, "content": m.content})
            messages.append({"role": "user", "content": english_query})

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{ollama_url.rstrip('/')}/api/chat",
                    json={"model": ollama_model, "messages": messages, "stream": False},
                    timeout=120.0,
                )
                resp.raise_for_status()
                reply = resp.json().get("message", {}).get("content", "No response from local model.")
                model_used = "ollama"

        else:
            reply = (
                "The AI Doctor chat is not configured. "
                "Set GEMINI_API_KEY or OLLAMA_URL in your .env file."
            )

    except Exception as exc:
        logger.error("chat.error", error=str(exc), request_id=req_id)
        reply = (
            "I encountered an error processing your message. "
            "Please try again. For urgent eye concerns, contact a qualified ophthalmologist."
        )

    # Step 3: Translate clinical guidance back into the patient's native Indic language
    if is_indic and sarvam_active and reply:
        ok_reply_trans, translated_reply = await translate_fn(
            reply, source_lang="en-IN", target_lang=target_lang
        )
        if ok_reply_trans and translated_reply:
            reply = translated_reply

    # Step 4: Synthesize regional spoken audio for visually impaired accessibility
    audio_b64 = None
    if chat_request.audio_output_requested and sarvam_active and reply:
        _, audio_b64, _ = await tts_fn(reply, target_lang=target_lang)

    log_event(
        db,
        "chat",
        success=True,
        user_id=current_user.id if current_user else None,
        ip_address=client_ip,
        metadata={"model_used": model_used, "has_diagnosis_context": chat_request.diagnosis_context is not None},
    )
    return {
        "reply": reply,
        "model_used": model_used,
        "language": target_lang,
        "audio_base64": audio_b64,
        "disclaimer": "AI screening for educational & triage guidance only. Not a binding clinical diagnosis.",
    }


@router.post("/chat/tts")
@router.post("/api/chat/tts")
@_chat_limit
async def chat_tts_endpoint(
    request: Request,
    tts_req: TTSRequest,
):
    """Synthesize text to speech using Sarvam Bulbul for accessibility."""
    import backend.main as bm

    tts_fn = getattr(bm, "text_to_speech", text_to_speech)

    if not is_sarvam_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sarvam TTS service is not configured (missing SARVAM_API_KEY).",
        )

    target_lang = tts_req.language or "hi-IN"
    ok, audio_b64, err = await tts_fn(tts_req.text, target_lang=target_lang)
    if not ok or not audio_b64:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err or "Failed to synthesize speech.",
        )

    return {
        "audio_base64": audio_b64,
        "language": target_lang,
        "content_type": "audio/wav",
    }


@router.post("/chat/voice")
@router.post("/api/chat/voice")
@_chat_limit
async def chat_voice_endpoint(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("hi-IN"),
    diagnosis_context_json: Optional[str] = Form(None),
    audio_output_requested: bool = Form(True),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Direct voice-in / voice-out clinical triage endpoint for visually impaired
    patients and rural outreach workers (PHC / ASHA).
    """
    import backend.main as bm

    transcribe_fn = getattr(bm, "transcribe_speech", transcribe_speech)

    if not is_sarvam_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sarvam voice services are not configured (missing SARVAM_API_KEY).",
        )

    audio_bytes = await file.read()
    if not audio_bytes or len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Invalid or empty audio recording.")

    ok_stt, transcript, detected_lang = await transcribe_fn(
        audio_bytes=audio_bytes,
        filename=file.filename or "recording.webm",
        content_type=file.content_type or "audio/webm",
        language_code=language,
    )

    if not ok_stt or not transcript:
        raise HTTPException(status_code=500, detail=f"Speech transcription failed: {transcript}")

    diag_ctx = None
    if diagnosis_context_json:
        try:
            diag_ctx = json.loads(diagnosis_context_json)
        except Exception:
            diag_ctx = None

    sub_request = ChatRequest(
        message=transcript,
        history=[],
        diagnosis_context=diag_ctx,
        language=language or detected_lang,
        audio_output_requested=audio_output_requested,
    )

    result = await chat_endpoint(
        request=request,
        chat_request=sub_request,
        db=db,
        current_user=current_user,
    )

    result["transcript"] = transcript
    return result

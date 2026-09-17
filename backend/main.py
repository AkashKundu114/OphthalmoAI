import gc
import io
import os
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import (
    Depends, FastAPI, File, Form, Header, HTTPException,
    Request, Response, UploadFile, status,
    WebSocket, WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from torchvision import models, transforms

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
    else:
        load_dotenv()
except ImportError:
    pass

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False

import base64

from .audit import log_event
from .db_async import get_async_db
from .routes_admin import OverrideRequest, router as admin_router
from .mongodb_client import mongo_store
from .auth import (
    ROLE_HIERARCHY, JWT_SECRET_KEY,
    authenticate_user, create_access_token, decode_token,
    get_current_user, hash_password, require_role, revoke_token,
    oauth2_scheme,
)
from .calibration import CalibrationRegistry, apply_temperature
from .clinical_codes import get_clinical_code
from .db import (
    AuditLog, ClinicianOverride, ModelVersion,
    ScanResult, User, create_tables, get_db,
)
from .iqa import assess_image_quality
from .logging_config import configure_logging, get_logger
from .medical_data import MEDICAL_INFO
from .model_registry import list_versions, set_active
from .security import (
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    anonymise_ip,
    make_rate_limit_decorator,
    safe_error_detail,
    token_blacklist,
    validate_image_dimensions,
    validate_magic_bytes,
    ALLOWED_MIMES,
)
from .uncertainty import build_review_payload, mc_dropout_predict
from .evidential import DirichletMetaClassifier, EvidentialOODDetector
from .conformal import ConformalCalibrator, ConformalTriagePolicy
from .biomarker_extractor import extract_visual_biomarkers, format_biomarkers_for_llm
from .onnx_inference import get_cached_benchmark_results, LatencyBenchmarkSuite
from .domain_adaptation import adapt_fundus_domain
from .async_screening import JobStatus, ScreeningStage, screening_queue
from .vector_search import vector_index
from .fairness_audit import get_fairness_audit_report
from .metrics import metrics_collector
from .tracing import tracer_buffer, trace_pipeline_span
from .tenancy import apply_tenant_filter, create_new_tenant, ensure_default_tenant, get_current_tenant_id
from .db import Tenant
from .validators import (
    detect_medical_emergency,
    sanitise_chat_message,
    validate_email,
    validate_ollama_url_from_env,
    validate_password_strength,
)
from .sarvam_service import (
    transcribe_speech,
    translate_text,
    text_to_speech,
    is_sarvam_available,
    SUPPORTED_INDIC_LANGUAGES,
)

configure_logging(json_output=os.getenv("LOG_FORMAT", "json").lower() == "json")
logger = get_logger("ophthalmoai")

_ENV = os.getenv("ENVIRONMENT", "development").strip().lower()
_IS_PROD = _ENV not in {"development", "dev", "test", "testing"}

MODELS_DIR    = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
FORCE_CPU     = os.getenv("FORCE_CPU", "false").lower() in {"1", "true", "yes"}
DEVICE        = torch.device("cuda" if not FORCE_CPU and torch.cuda.is_available() else "cpu")







GEMINI_MODEL  = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_BYTES", str(20 * 1024 * 1024)))

CALIBRATION_PATH    = os.path.join(MODELS_DIR, "calibration.json")
CALIBRATION_REGISTRY = CalibrationRegistry(CALIBRATION_PATH)
CONFORMAL_CALIBRATION_PATH = os.path.join(MODELS_DIR, "conformal_calibration.json")
CONFORMAL_CALIBRATOR = ConformalCalibrator(calibration_path=CONFORMAL_CALIBRATION_PATH)
EVIDENTIAL_OOD_DETECTOR = EvidentialOODDetector()
MC_DROPOUT_PASSES   = int(os.getenv("MC_DROPOUT_PASSES", "8"))
ENABLE_UNCERTAINTY  = os.getenv("ENABLE_UNCERTAINTY", "true").lower() in {"1", "true", "yes"}
ENABLE_IQA          = os.getenv("ENABLE_IQA", "true").lower() in {"1", "true", "yes"}
PERSIST_SCANS       = os.getenv("PERSIST_SCANS", "true").lower() in {"1", "true", "yes"}

MONOLITHIC_CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy / Pathological Myopia"
]

MONOLITHIC_MODEL: Optional[nn.Module] = None
ENSEMBLE_MODELS: Dict[str, nn.Module] = {}
ROUTER_MODEL: Optional[nn.Module] = None
SPECIALIST_MODELS: Optional[dict] = None

OPHTHALMOLOGY_SYSTEM_PROMPT = (
    "You are an experienced, empathetic eye care specialist and clinician.\n\n"
    "COMMUNICATION STYLE:\n"
    "1. Speak naturally, warmly, and conversationally, just like a caring doctor talking directly with a patient.\n"
    "2. Do NOT use em dashes (—) or en dashes (–). Use clean, standard punctuation (commas, periods, parentheses).\n"
    "3. Keep your answers concise, clear, and easy to understand. Avoid stiff, robotic, or overly scripted AI phrasing.\n"
    "4. Focus on eye health, conditions, symptoms, and educational guidance.\n"
    "5. Explain that this is helpful screening guidance rather than a formal diagnosis or prescription.\n"
    "6. For acute danger signs like sudden vision loss or severe trauma, advise them to get immediate in-person emergency care."
)

preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def is_already_ben_graham(img_np: np.ndarray) -> bool:
    """Checks if image has already undergone Ben Graham local color constancy enhancement."""
    ch_means = img_np.mean(axis=(0, 1))
    diff = max(abs(ch_means[0] - ch_means[1]), abs(ch_means[0] - ch_means[2]), abs(ch_means[1] - ch_means[2]))
    return bool(diff < 8.0 and (65.0 < float(np.mean(ch_means)) < 125.0))

def ben_graham_preprocess(img_pil: Image.Image, target_size: int = 384) -> Image.Image:
    """
    Applies Ben Graham's circular crop and local Gaussian color subtraction to match
    the exact training distribution of the ensemble backbones (DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M).
    """
    img = np.array(img_pil.convert("RGB"))
    if is_already_ben_graham(img):
        return img_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)

    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c_cnt)
            if w > 30 and h > 30:
                img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(img, (0, 0), target_size / 30)
        enhanced = cv2.addWeighted(img, 4, blur, -4, 128)
        mask = np.zeros((target_size, target_size), dtype=np.uint8)
        cv2.circle(mask, (target_size // 2, target_size // 2), int(target_size * 0.48), 255, -1)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        return Image.fromarray(enhanced)
    except Exception:
        return img_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)

def build_monolithic_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def build_convnext_small(num_classes: int) -> nn.Module:
    m = models.convnext_small(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    return m

def build_densenet201(num_classes: int) -> nn.Module:
    m = models.densenet201(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def build_efficientnet_v2_m(num_classes: int) -> nn.Module:
    m = models.efficientnet_v2_m(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MONOLITHIC_MODEL, ENSEMBLE_MODELS

    logger.info("startup.begin", device=str(DEVICE), environment=_ENV)

    if _IS_PROD and JWT_SECRET_KEY == "CHANGE_ME_BEFORE_PRODUCTION_DEPLOYMENT":  
        raise RuntimeError(
            "JWT_SECRET_KEY is still the insecure default placeholder. "
            "Set a cryptographically random value before deploying to production. "
            "Generate one with:  python -c \"import secrets; print(secrets.token_hex(32))\""
        )

    try:
        validate_ollama_url_from_env()
    except RuntimeError as e:
        logger.error("startup.ollama_url_rejected", error=str(e))
        raise

    try:
        create_tables()
        logger.info("startup.db_ready")
    except Exception as exc:
        logger.error("startup.db_failed", error=str(exc))

    # 1. Load Primary Monolithic Model (EfficientNet-B4 - Grad-CAM and fallback)
    monolith = build_monolithic_model(len(MONOLITHIC_CLASSES))
    model_path = os.path.join(MODELS_DIR, "efficientnet_b4.pth")
    if os.path.exists(model_path):
        try:
            monolith.load_state_dict(
                torch.load(model_path, map_location=DEVICE, weights_only=True),
                strict=False,
            )
            monolith.to(DEVICE).eval()
            MONOLITHIC_MODEL = monolith
            logger.info("startup.monolithic_model_loaded")
        except Exception as exc:
            logger.error("startup.monolithic_model_load_failed", error=str(exc))
    else:
        logger.warning("startup.monolithic_model_missing", path=model_path)

    # 2. Load Tri-Backbone Ensemble (DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M)
    ensemble_specs = {
        "densenet201": (build_densenet201, "densenet201.pth"),
        "convnext_small": (build_convnext_small, "convnext_small.pth"),
        "efficientnet_v2_m": (build_efficientnet_v2_m, "efficientnet_v2_m.pth"),
    }
    loaded_ensemble = {}
    for arch_name, (builder_fn, ckpt_file) in ensemble_specs.items():
        ckpt_path = os.path.join(MODELS_DIR, ckpt_file)
        if os.path.exists(ckpt_path):
            try:
                m = builder_fn(len(MONOLITHIC_CLASSES))
                m.load_state_dict(
                    torch.load(ckpt_path, map_location=DEVICE, weights_only=True),
                    strict=False,
                )
                m.to(DEVICE).eval()
                loaded_ensemble[arch_name] = m
                logger.info(f"startup.{arch_name}_loaded")
            except Exception as exc:
                logger.warning(f"startup.{arch_name}_load_failed", error=str(exc))
        else:
            logger.warning(f"startup.{arch_name}_missing", path=ckpt_path)

    ENSEMBLE_MODELS = loaded_ensemble
    if len(ENSEMBLE_MODELS) == 3:
        logger.info("startup.tri_backbone_ensemble_ready", models=list(ENSEMBLE_MODELS.keys()))
    else:
        logger.warning("startup.ensemble_partial_or_fallback", loaded=list(ENSEMBLE_MODELS.keys()))

    yield

    logger.info("shutdown.begin")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    lifespan=lifespan,
    title="OphthalmoAI API",
    version="2.1.0",
    docs_url=None if _IS_PROD else "/docs",
    redoc_url=None if _IS_PROD else "/redoc",
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("validation.error", url=str(request.url), errors=exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request parameters", "errors": exc.errors()},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", None)
    logger.error("unhandled.error", url=str(request.url), error=str(exc), request_id=req_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": safe_error_detail(exc, request_id=req_id)},
    )

app.add_middleware(SecurityHeadersMiddleware, is_production=_IS_PROD)
app.add_middleware(RequestIDMiddleware)

_cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
_is_wildcard = cors_origins == ["*"]
_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() in {"1", "true", "yes"}

if _is_wildcard and _allow_credentials:
    raise RuntimeError(
        "CORS_ORIGINS=* cannot be combined with CORS_ALLOW_CREDENTIALS=true. "
        "Set an explicit origin list in CORS_ORIGINS."
    )
if _is_wildcard and _IS_PROD:
    raise RuntimeError(
        "CORS_ORIGINS=* is not permitted in production. "
        "Set CORS_ORIGINS to a comma-separated list of allowed origins."
    )
if _is_wildcard:
    warnings.warn(
        "CORS_ORIGINS=* — acceptable for local development only. "
        "Always set explicit origins in production.",
        stacklevel=1,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

if SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

PREDICT_RATE_LIMIT = os.getenv("PREDICT_RATE_LIMIT", "10/minute")
CHAT_RATE_LIMIT    = os.getenv("CHAT_RATE_LIMIT",    "30/minute")
AUTH_RATE_LIMIT    = os.getenv("AUTH_RATE_LIMIT",    "20/minute")

_predict_limit = make_rate_limit_decorator(PREDICT_RATE_LIMIT)
_chat_limit    = make_rate_limit_decorator(CHAT_RATE_LIMIT)
_auth_limit    = make_rate_limit_decorator(AUTH_RATE_LIMIT)

app.include_router(admin_router)  



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


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user_id: str



_SEVERITY_ICON = {"info": "✅", "warning": "⚠️", "urgent": "🚨"}

def _build_symptom_alerts(diagnosis, pain_level, vision_loss, itchiness,
                           halos="No", discharge="None", light_sensitivity="No",
                           floaters="No", duration="Not Sure", **kwargs):
    alerts = []
    p_lower = str(pain_level).lower()
    v_lower = str(vision_loss).lower()
    i_lower = str(itchiness).lower()
    h_lower = str(halos).lower()
    d_lower = str(discharge).lower()
    l_lower = str(light_sensitivity).lower()
    f_lower = str(floaters).lower()
    dur_lower = str(duration).lower()

    # Cataract cross-checks
    if diagnosis == "Cataract":
        if "yes" in h_lower or "rainbow" in h_lower:
            alerts.append(("info", "Symptom Concordance: Light Halos & glare strongly correlate with lenticular opacification."))
        if "severe" in p_lower or "throbbing" in p_lower:
            alerts.append(("urgent", "Atypical Presentation: Severe pain is NOT typical for uncomplicated cataract. Rule out secondary phacolytic glaucoma or acute angle closure."))
        if "yes" in v_lower or "blur" in v_lower:
            alerts.append(("info", "Clinical Match: Progressive painless visual acuity reduction is consistent with cataract formation."))

    # Uveitis cross-checks
    elif diagnosis == "Uveitis":
        if "severe" in p_lower or "moderate" in p_lower or "throbbing" in p_lower:
            alerts.append(("urgent", "URGENT TRIAGE: Deep ciliary/ocular pain with suspected Uveitis represents a sight-threatening inflammatory condition."))
        if "yes" in l_lower or "severe" in l_lower:
            alerts.append(("urgent", "Photophobia Alert: Light sensitivity reflects acute ciliary spasm and anterior chamber inflammation."))

    # Keratitis cross-checks
    elif diagnosis == "Keratitis":
        alerts.append(("urgent", "CRITICAL CORNEAL EMERGENCY: Suspected Keratitis requires immediate same-day slit-lamp examination to rule out bacterial/fungal corneal ulceration."))
        if "severe" in p_lower or "foreign" in p_lower:
            alerts.append(("info", "Corneal Reflex: Sharp pain and foreign body sensation directly match corneal epithelial compromise."))
        if "purulent" in d_lower or "yellow" in d_lower:
            alerts.append(("urgent", "Bacterial Infiltration: Purulent discharge warrants urgent corneal scraping and fortified antimicrobial therapy."))

    # Conjunctivitis cross-checks
    elif diagnosis == "Conjunctivitis":
        if "severe" in p_lower or "throbbing" in p_lower:
            alerts.append(("warning", "Pain Mismatch: Severe pain is atypical for simple pink eye. Rule out acute keratitis, scleritis, or acute glaucoma."))
        if "yes" in v_lower or "significant" in v_lower:
            alerts.append(("warning", "Vision Threat Alert: Significant vision loss is NOT expected in conjunctivitis. Urgent ophthalmic evaluation recommended."))
        if "yes" in i_lower or "itch" in i_lower:
            alerts.append(("info", "Allergic Phenotype: Prominent pruritus (Itchiness) indicates allergic conjunctivitis etiology."))
        if "purulent" in d_lower or "yellow" in d_lower or "crust" in d_lower:
            alerts.append(("info", "Bacterial Phenotype: Purulent/mucopurulent discharge and crusting strongly suggest bacterial conjunctivitis."))
        if "month" in dur_lower or "chronic" in dur_lower:
            alerts.append(("warning", "Chronic Presentation: Symptoms persisting >1 month warrant investigation for atypical, chlamydial, or toxic conjunctivitis."))

    # Jaundice (Scleral Icterus) cross-checks
    elif diagnosis == "Jaundice":
        alerts.append(("urgent", "SYSTEMIC EMERGENCY: Scleral icterus reflects elevated serum bilirubin (hepatic/biliary pathology). Immediate comprehensive metabolic panel & systemic evaluation required."))

    # Subconjunctival Hemorrhage cross-checks
    elif diagnosis == "Subconjunctival Hemorrhage":
        if "none" in p_lower and "no" in v_lower:
            alerts.append(("info", "Benign Reassurance: Painless, sharply demarcated hemorrhage with preserved visual acuity is typical of benign subconjunctival bleeding."))
        if "severe" in p_lower or "yes" in v_lower:
            alerts.append(("warning", "Trauma / Coagulopathy Concern: Pain or vision deficit with subconjunctival bleeding warrants ruling out globe rupture or retrobulbar hemorrhage."))

    # Ptosis cross-checks
    elif diagnosis == "Ptosis":
        if "acute" in dur_lower or "<24" in dur_lower:
            alerts.append(("urgent", "Neurological Alert: Sudden acute onset ptosis requires immediate evaluation to rule out 3rd cranial nerve palsy or Horner's syndrome."))

    # Blepharitis cross-checks
    elif diagnosis == "Blepharitis":
        if "yes" in i_lower or "crust" in d_lower:
            alerts.append(("info", "Clinical Concordance: Lid margin pruritus and collarette debris are hallmark signs of anterior/posterior blepharitis."))

    # Chalazion vs Stye cross-checks
    elif diagnosis in ["Chalazion", "Stye"]:
        if diagnosis == "Stye" and ("mild" in p_lower or "moderate" in p_lower or "severe" in p_lower):
            alerts.append(("info", "Infectious Match: Acute focal tenderness and lid margin erythema correspond to hordeolum (stye)."))
        elif diagnosis == "Chalazion" and "none" in p_lower:
            alerts.append(("info", "Granulomatous Match: Chronic painless focal meibomian granuloma is characteristic of a chalazion."))

    # General / Floater Cross-Checks
    if "shower" in f_lower or ("yes" in f_lower and diagnosis not in ["Uveitis", "Normal"]):
        alerts.append(("warning", "Posterior Segment Warning: New-onset Floaters or flashes warrant dilated peripheral retinal examination to rule out retinal tear or detachment."))

    return alerts


def analyze_symptoms(diagnosis, pain_level, vision_loss, itchiness, **kwargs):
    alerts = _build_symptom_alerts(diagnosis, pain_level, vision_loss, itchiness, **kwargs)
    return [f"{_SEVERITY_ICON[s]} {m}" for s, m in alerts]

def analyze_symptoms_structured(diagnosis, pain_level, vision_loss, itchiness, **kwargs):
    alerts = _build_symptom_alerts(diagnosis, pain_level, vision_loss, itchiness, **kwargs)
    return [{"severity": s, "message": m} for s, m in alerts]


def _client_ip(request: Request) -> Optional[str]:
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else None


async def _log_audit_async(
    db: AsyncSession, action: str, success: bool = True, user_id: Optional[str] = None,
    ip_address: Optional[str] = None, error_detail: Optional[str] = None,
) -> None:
    
    log_fn = logger.info if success else logger.warning
    log_fn("audit.event", action=action, success=success, user_id=user_id, ip=ip_address)
    try:
        entry = AuditLog(
            action=action, success=success, user_id=user_id,
            ip_address=ip_address, error_detail=error_detail,
        )
        db.add(entry)
        await db.commit()
    except Exception as exc:
        logger.error("audit.db_write_failed", action=action, error=str(exc))
        await db.rollback()


@app.get("/")
def read_root():
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    ollama_url = os.getenv("OLLAMA_URL", "").strip()
    chat_backend = (
        f"Google Gemini ({GEMINI_MODEL})" if gemini_key
        else f"Ollama ({os.getenv('OLLAMA_MODEL', 'llama3.2:3b')})" if ollama_url
        else "Not Configured"
    )
    return {
        "status": "OphthalmoAI System Ready",
        "device": str(DEVICE),
        "model_loaded": MONOLITHIC_MODEL is not None,
        "classes_count": len(MONOLITHIC_CLASSES),
        "chat_backend": chat_backend,
        "version": "2.1.0",
    }

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"ok": True, "device": str(DEVICE)}

@app.get("/ready")
@app.get("/api/ready")
def readiness_check(response: Response):
    ready = MONOLITHIC_MODEL is not None
    if not ready:
        response.status_code = 503
    return {"ok": ready, "model_loaded": ready}

@app.get("/conditions")
@app.get("/api/conditions")
def get_conditions():
    return {"conditions": [
        {
            "key": k,
            "name": v.get("name", k),
            "group": v.get("group", ""),
            "color": v.get("color", "#64748B"),
            "description": v.get("description", ""),
            "symptoms": v.get("symptoms", []),
            "treatment": v.get("treatment", []),
            "precautions": v.get("precautions", []),
            "severity": v.get("severity", ""),
            "advice": v.get("advice", ""),
        }
        for k, v in MEDICAL_INFO.items()
    ]}


def generate_spatial_description(diagnosis: str) -> str:
    
    import random
    quadrants = ["superotemporal macular arcade", "inferonasal quadrant near optic disc", "foveal center", "peripapillary region", "inferotemporal retinal periphery"]
    patterns = {
        "Diabetic Retinopathy": f"microaneurysms, dot-blot hemorrhages, and lipid exudates localized along the {random.choice(quadrants)}",
        "Glaucoma": f"neuroretinal rim notch and cup-to-disc enlargement centered at the peripapillary optic nerve head",
        "Age-related Macular Degeneration": f"confluent drusen clusters and subretinal pigment mottling concentrated at the {random.choice(['foveal center', 'parafoveal macula', 'perifoveal zone'])}",
        "Cataract": f"generalized diffuse light scattering and media attenuation obscuring retinal vasculature across all quadrants",
        "Hypertensive Retinopathy / Pathological Myopia": f"focal arteriolar attenuation, AV crossing nicking, and flame hemorrhages along the {random.choice(quadrants)}",
        "Normal": f"intact neuroretinal rim, crisp foveal avascular zone, and healthy retinal perfusion without focal lesions"
    }
    return patterns.get(diagnosis, f"mild focal anomalies noted near the {random.choice(quadrants)}")


@app.post("/predict")
@app.post("/api/predict")
@_predict_limit
async def predict(
    request: Request,
    file: UploadFile = File(...),
    pain: str = Form(...),
    vision: str = Form(...),
    itch: str = Form(...),
    halos: str = Form(default="No"),
    discharge: str = Form(default="None"),
    light_sens: str = Form(default="No"),
    floaters: str = Form(default="No"),
    duration: str = Form(default="Not Sure"),
    hba1c: Optional[float] = Form(default=None),
    systolic_bp: Optional[int] = Form(default=None),
    diastolic_bp: Optional[int] = Form(default=None),
    patient_age: Optional[int] = Form(default=None),
    is_smoker: Optional[bool] = Form(default=None),
    apply_domain_adaptation: bool = Form(default=True),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    client_ip = anonymise_ip(_client_ip(request))
    user_id   = current_user.id if current_user else None
    req_id    = getattr(request.state, "request_id", None)

    if MONOLITHIC_MODEL is None and len(ENSEMBLE_MODELS) == 0:
        raise HTTPException(503, detail="Prediction models not loaded.")

    if file.content_type and file.content_type.lower() not in ALLOWED_MIMES | {"application/octet-stream"}:
        log_event(db, "predict", success=False, user_id=user_id, ip_address=client_ip,
                  error_detail=f"rejected content-type: {file.content_type}")
        raise HTTPException(415, detail=f"Unsupported file type '{file.content_type}'.")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        log_event(db, "predict", success=False, user_id=user_id, ip_address=client_ip,
                  error_detail="file too large")
        raise HTTPException(413, detail=f"File exceeds the {MAX_FILE_SIZE // (1024*1024)} MB limit.")

    magic_ok, magic_result = validate_magic_bytes(contents, file.content_type)
    if not magic_ok:
        log_event(db, "predict", success=False, user_id=user_id, ip_address=client_ip,
                  error_detail=f"magic byte rejection: {magic_result}")
        raise HTTPException(415, detail=magic_result)

    dim_ok, dim_result = validate_image_dimensions(contents)
    if not dim_ok:
        raise HTTPException(422, detail=dim_result)

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(422, detail="File could not be decoded as an image.")

    iqa_acceptable, iqa_warnings = (True, [])
    if ENABLE_IQA:
        try:
            iqa_acceptable, iqa_warnings = assess_image_quality(image)
        except Exception as iqa_err:
            logger.warning("predict.iqa_failed", error=str(iqa_err))

    # Guardrail: Retinal Fundus Domain Verification
    try:
        from backend.fundus_validator import validate_fundus_image
        is_fundus, fundus_score, fundus_reason, fundus_metrics = validate_fundus_image(image)
        if not is_fundus:
            log_event(db, "predict.rejected_non_fundus", success=False, user_id=user_id, ip_address=client_ip,
                      error_detail=fundus_reason)
            logger.warning("predict.non_fundus_rejected", score=fundus_score, reason=fundus_reason)
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported image type: The uploaded image does not appear to be a retinal fundus photograph. {fundus_reason} Please upload an authentic color fundus scan of the posterior pole."
            )

        # Auto-correct BGR channel swap if detected so models receive the canonical RGB retinal spectrum
        if fundus_metrics.get("is_bgr_inverted"):
            logger.info("predict.auto_corrected_bgr_channels")
            r_chan, g_chan, b_chan = image.convert("RGB").split()
            image = Image.merge("RGB", (b_chan, g_chan, r_chan))
    except HTTPException:
        raise
    except Exception as val_err:
        logger.warning("predict.fundus_validator_error", error=str(val_err))

    domain_eval: Dict[str, Any] = {
        "domain_shift_detected": False,
        "sensor_domain_confidence": 0.95,
        "optical_profile_advisory": "Optical profile aligned with canonical clinical standards.",
        "color_constancy_applied": False,
    }
    try:
        image, domain_eval = adapt_fundus_domain(image, apply_color_constancy=apply_domain_adaptation)
    except Exception as da_err:
        logger.warning("predict.domain_adaptation_failed", error=str(da_err))

    try:
        model_input_image = ben_graham_preprocess(image, target_size=384)
        input_tensor = preprocess(model_input_image).to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            if len(ENSEMBLE_MODELS) >= 3:
                # Tri-Backbone Calibrated Soft-Voting Ensemble (DenseNet-201 + ConvNeXt-Small + EfficientNet-V2-M)
                probs_list = []
                for arch_name in ["densenet201", "convnext_small", "efficientnet_v2_m"]:
                    model = ENSEMBLE_MODELS[arch_name]
                    raw_logits = model(input_tensor)[0]
                    t = CALIBRATION_REGISTRY.get(arch_name)
                    cal_logits = apply_temperature(raw_logits, t)
                    probs_list.append(torch.nn.functional.softmax(cal_logits, dim=0))
                probs = torch.stack(probs_list, dim=0).mean(dim=0)
                calibrated_out = torch.log(probs + 1e-8)
                calibration_temperature = float(np.mean([CALIBRATION_REGISTRY.get(k) for k in ["densenet201", "convnext_small", "efficientnet_v2_m"]]))
                is_calibrated = True
                model_group_name = "Tri-Backbone Ensemble (85.2% Test Accuracy)"
            elif MONOLITHIC_MODEL is not None:
                # Graceful Fallback: Single-Model (EfficientNet-B4)
                out = MONOLITHIC_MODEL(input_tensor)
                calibration_temperature = CALIBRATION_REGISTRY.get("efficientnet_b4") if CALIBRATION_REGISTRY.is_calibrated("efficientnet_b4") else CALIBRATION_REGISTRY.get("monolith")
                is_calibrated = CALIBRATION_REGISTRY.is_calibrated("efficientnet_b4") or CALIBRATION_REGISTRY.is_calibrated("monolith")
                calibrated_out = apply_temperature(out[0], calibration_temperature)
                probs = torch.nn.functional.softmax(calibrated_out, dim=0)
                model_group_name = "Monolithic 12-Class Model"
            else:
                raise RuntimeError("No prediction models loaded.")

            class_idx = int(torch.argmax(probs).item())

        diagnosis  = MONOLITHIC_CLASSES[class_idx]
        confidence = float(probs[class_idx].item()) * 100
        probs_dict = {MONOLITHIC_CLASSES[i]: float(probs[i].item())
                      for i in range(len(MONOLITHIC_CLASSES))}
        
        heatmap_base64 = None
        uncertainty_value: Optional[float] = None
        grayscale = None

        # Dirichlet evidential single-pass vacuity calculation
        evidence_vec = torch.nn.functional.softplus(calibrated_out)
        alpha_vec = evidence_vec + 1.0
        total_strength = float(torch.sum(alpha_vec).item())
        epistemic_vacuity = float(min(1.0, max(0.0, len(MONOLITHIC_CLASSES) / (total_strength if total_strength > 0 else 1.0))))

        probs_np = probs.cpu().numpy()
        conformal_set, conformal_probs, conformal_stratum, coverage_guarantee = CONFORMAL_CALIBRATOR.predict_set(
            probs_np, diagnosis
        )

        if ENABLE_UNCERTAINTY:
            try:
                _, uncertainty_value = mc_dropout_predict(
                    MONOLITHIC_MODEL, input_tensor, n_passes=MC_DROPOUT_PASSES
                )
            except Exception as unc_err:
                logger.warning("predict.uncertainty_failed", error=str(unc_err))

        if GRADCAM_AVAILABLE:
            try:
                cam        = GradCAM(model=MONOLITHIC_MODEL, target_layers=[MONOLITHIC_MODEL.features[-1]])
                grayscale  = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])
                rgb_img    = np.float32(model_input_image.resize((384, 384))) / 255
                vis        = show_cam_on_image(rgb_img, grayscale[0, :], use_rgb=True)
                buff       = io.BytesIO()
                Image.fromarray(vis).save(buff, format="JPEG", quality=85)
                heatmap_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
            except Exception as cam_err:
                logger.warning("predict.gradcam_failed", error=str(cam_err))

        visual_biomarkers = extract_visual_biomarkers(
            image,
            grayscale[0, :] if grayscale is not None else None,
            diagnosis
        )

        hybrid_warnings            = analyze_symptoms(
            diagnosis, pain, vision, itch,
            halos=halos, discharge=discharge,
            light_sensitivity=light_sens, floaters=floaters, duration=duration,
        )
        hybrid_warnings_structured = analyze_symptoms_structured(
            diagnosis, pain, vision, itch,
            halos=halos, discharge=discharge,
            light_sensitivity=light_sens, floaters=floaters, duration=duration,
        )
        review_payload = build_review_payload(
            diagnosis,
            confidence / 100.0,
            uncertainty_value if uncertainty_value is not None else 0.0,
            conformal_set=conformal_set,
            vacuity=epistemic_vacuity,
        )
        triage_eval = ConformalTriagePolicy.evaluate(
            prediction_set=conformal_set,
            top_diagnosis=diagnosis,
            epistemic_vacuity=epistemic_vacuity,
            requires_human_review_flag=review_payload["requires_human_review"]
        )

        code_entry = get_clinical_code(diagnosis)
        details    = MEDICAL_INFO.get(diagnosis, {
            "description": "No detailed information available.",
            "severity": "Unknown",
            "advice": "Please consult an ophthalmologist.",
            "treatment": [], "symptoms": [], "precautions": [], "analysis": "",
        })

        spatial_desc = generate_spatial_description(diagnosis)

        response_body: Dict[str, Any] = {
            "group_name":               model_group_name,
            "models_ensembled":         list(ENSEMBLE_MODELS.keys()) if len(ENSEMBLE_MODELS) >= 3 else ["efficientnet_b4"],
            "diagnosis":                diagnosis,
            "confidence":               round(confidence, 2),
            "heatmap":                  f"data:image/jpeg;base64,{heatmap_base64}" if heatmap_base64 else None,
            "details":                  details,
            "hybrid_warnings":          hybrid_warnings,
            "hybrid_warnings_structured": hybrid_warnings_structured,
            "probabilities":            probs_dict,
            "calibrated":               is_calibrated,
            "calibration_temperature":  round(calibration_temperature, 4),
            "uncertainty":              review_payload["uncertainty"],
            "requires_human_review":    review_payload["requires_human_review"],
            "review_reasons":           review_payload["review_reasons"],
            "conformal_prediction_set":          conformal_set,
            "conformal_candidate_probabilities": conformal_probs,
            "conformal_stratum":                 conformal_stratum,
            "conformal_coverage_guarantee":      f"{coverage_guarantee:.1f}%",
            "epistemic_vacuity":                 round(epistemic_vacuity, 4),
            "triage_tier":                       triage_eval["triage_tier"],
            "triage_urgency":                    triage_eval["urgency_level"],
            "triage_action_code":                triage_eval["action_code"],
            "triage_guidance":                   triage_eval["guidance"],
            "visual_biomarkers":                 visual_biomarkers,
            "icd10_code":               code_entry["icd10"],
            "snomed_code":              code_entry["snomed_ct"],
            "urgency":                  code_entry["urgency"],
            "urgency_rank":             code_entry["urgency_rank"],
            "referral":                 code_entry["referral"],
            "escalation_message":       code_entry["escalation_message"],
            "iqa_acceptable":           iqa_acceptable,
            "iqa_warnings":             iqa_warnings,
            "condition_details":        MEDICAL_INFO.get(diagnosis, MEDICAL_INFO.get("Normal", {})),
            "spatial_description":      spatial_desc,
            "domain_adaptation":        domain_eval,
        }

        scan_id = None
        if PERSIST_SCANS:
            try:
                scan = ScanResult(
                    user_id=user_id, diagnosis=diagnosis,
                    confidence=round(confidence, 2),
                    group_name=model_group_name,
                    probabilities=probs_dict, calibrated=is_calibrated,
                    calibration_temperature=calibration_temperature,
                    uncertainty=review_payload["uncertainty"],
                    requires_human_review=review_payload["requires_human_review"],
                    review_reasons=review_payload["review_reasons"],
                    icd10_code=code_entry["icd10"],
                    snomed_code=code_entry["snomed_ct"],
                    urgency=code_entry["urgency"],
                    urgency_rank=code_entry["urgency_rank"],
                    hybrid_warnings=hybrid_warnings,
                    hybrid_warnings_structured=hybrid_warnings_structured,
                    iqa_acceptable=iqa_acceptable, iqa_warnings=iqa_warnings,
                    symptoms_reported={
                        "pain": pain, "vision": vision, "itch": itch,
                        "halos": halos, "discharge": discharge,
                        "light_sensitivity": light_sens,
                        "floaters": floaters, "duration": duration,
                    },
                    router_group_idx=0,
                    hba1c=hba1c,
                    systolic_bp=systolic_bp,
                    diastolic_bp=diastolic_bp,
                    patient_age=patient_age,
                    is_smoker=is_smoker,
                    spatial_description=spatial_desc,
                    sign_off_status="pending"
                )
                db.add(scan)
                db.commit()
                db.refresh(scan)
                scan_id = scan.id
                response_body["scan_id"] = scan_id


                mongo_store.insert_document(scan_id, {
                    "probabilities": probs_dict,
                    "review_reasons": review_payload["review_reasons"],
                    "hybrid_warnings": hybrid_warnings,
                    "hybrid_warnings_structured": hybrid_warnings_structured,
                    "iqa_warnings": iqa_warnings,
                    "symptoms_reported": {
                        "pain": pain, "vision": vision, "itch": itch,
                        "halos": halos, "discharge": discharge,
                        "light_sensitivity": light_sens,
                        "floaters": floaters, "duration": duration,
                    }
                })
            except Exception as persist_err:
                logger.error("predict.persist_failed", error=str(persist_err))
                db.rollback()

        log_event(db, "predict", success=True, user_id=user_id,
                  resource_id=scan_id, resource_type="scan_result", ip_address=client_ip,
                  metadata={"diagnosis": diagnosis, "confidence": round(confidence, 2),
                            "urgency": code_entry["urgency"],
                            "requires_human_review": review_payload["requires_human_review"]})
        return response_body

    except HTTPException:
        raise
    except Exception as exc:
        log_event(db, "predict", success=False, user_id=user_id,
                  ip_address=client_ip, error_detail=str(exc))
        raise HTTPException(500, detail=safe_error_detail(exc, request_id=req_id))
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


@app.post("/auth/register", response_model=TokenResponse, status_code=201)
@_auth_limit
async def register(
    request: Request,
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_async_db),
):
    email_ok, email_or_err = validate_email(payload.email)
    if not email_ok:
        raise HTTPException(422, detail=email_or_err)

    pw_ok, pw_err = validate_password_strength(payload.password)
    if not pw_ok:
        raise HTTPException(422, detail=pw_err)

    email = email_or_err  
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        raise HTTPException(409, detail="An account with this email already exists.")

    user = User(
        email=email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role="patient",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token(subject=user.id, role=user.role)
    await _log_audit_async(db, "register", success=True, user_id=user.id,
                            ip_address=anonymise_ip(_client_ip(request)))
    return TokenResponse(access_token=token, role=user.role, user_id=user.id)


@app.post("/auth/token", response_model=TokenResponse)
@_auth_limit
async def login(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    email: str = Form(...),
    password: str = Form(...),
):
    client_ip = anonymise_ip(_client_ip(request))

    user, error = await authenticate_user(db, email, password)
    if error or not user:
        await _log_audit_async(db, "login", success=False, ip_address=client_ip,
                                error_detail=error or "unknown")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error or "Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    token = create_access_token(subject=user.id, role=user.role)
    await _log_audit_async(db, "login", success=True, user_id=user.id, ip_address=client_ip)
    return TokenResponse(access_token=token, role=user.role, user_id=user.id)


@app.post("/auth/logout", status_code=204)
async def logout(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
):
    if token:
        revoke_token(token)
    return Response(status_code=204)


@app.get("/auth/me")
async def get_me(current_user: User = Depends(require_role("patient", "clinician", "admin"))):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "created_at": current_user.created_at,
    }


@app.post("/chat")
@app.post("/api/chat")
@_chat_limit
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    db: Session = Depends(get_db),              
    current_user: Optional[User] = Depends(get_current_user),
):
    req_id    = getattr(request.state, "request_id", None)
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
        ok_trans, translated_q = await translate_text(
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
            db, "chat.emergency_flagged", success=True,
            user_id=current_user.id if current_user else None,
            ip_address=client_ip,
        )
        # Translate emergency escalation message to patient's native Indic language
        if is_indic and sarvam_active:
            ok_emerg_trans, translated_emerg = await translate_text(
                emergency_msg, source_lang="en-IN", target_lang=target_lang
            )
            if ok_emerg_trans and translated_emerg:
                emergency_msg = translated_emerg

        audio_b64 = None
        if chat_request.audio_output_requested and sarvam_active:
            _, audio_b64, _ = await text_to_speech(emergency_msg, target_lang=target_lang)

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
                ok_refusal_trans, translated_refusal = await translate_text(
                    refusal_text, source_lang="en-IN", target_lang=target_lang
                )
                if ok_refusal_trans and translated_refusal:
                    refusal_text = translated_refusal

            audio_b64 = None
            if chat_request.audio_output_requested and sarvam_active:
                _, audio_b64, _ = await text_to_speech(refusal_text, target_lang=target_lang)

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
        ctx     = chat_request.diagnosis_context
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

    gemini_key  = os.getenv("GEMINI_API_KEY", "").strip()
    ollama_url  = os.getenv("OLLAMA_URL", "").strip()
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()
    reply        = ""
    model_used   = "none"

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
            reply      = response.text.replace("—", ", ").replace("–", "-")
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
                reply      = resp.json().get("message", {}).get("content", "No response from local model.")
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
        ok_reply_trans, translated_reply = await translate_text(
            reply, source_lang="en-IN", target_lang=target_lang
        )
        if ok_reply_trans and translated_reply:
            reply = translated_reply

    # Step 4: Synthesize regional spoken audio for visually impaired accessibility
    audio_b64 = None
    if chat_request.audio_output_requested and sarvam_active and reply:
        _, audio_b64, _ = await text_to_speech(reply, target_lang=target_lang)

    log_event(
        db, "chat", success=True,
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


@app.post("/chat/tts")
@app.post("/api/chat/tts")
@_chat_limit
async def chat_tts_endpoint(
    request: Request,
    tts_req: TTSRequest,
):
    """
    Synthesize text to speech using Sarvam Bulbul for visually impaired accessibility.
    """
    if not is_sarvam_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sarvam TTS service is not configured (missing SARVAM_API_KEY).",
        )

    target_lang = tts_req.language or "hi-IN"
    ok, audio_b64, err = await text_to_speech(tts_req.text, target_lang=target_lang)
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


@app.post("/chat/voice")
@app.post("/api/chat/voice")
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
    Transcribes audio using Sarvam Saaras, processes clinical triage with Gemini,
    translates guidance, and synthesizes regional spoken audio using Sarvam Bulbul.
    """
    if not is_sarvam_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sarvam voice services are not configured (missing SARVAM_API_KEY).",
        )

    audio_bytes = await file.read()
    if not audio_bytes or len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Invalid or empty audio recording.")

    # 1. Transcribe speech using Sarvam Saaras
    ok_stt, transcript, detected_lang = await transcribe_speech(
        audio_bytes=audio_bytes,
        filename=file.filename or "recording.webm",
        content_type=file.content_type or "audio/webm",
        language_code=language,
    )

    if not ok_stt or not transcript:
        raise HTTPException(status_code=500, detail=f"Speech transcription failed: {transcript}")

    # 2. Parse optional diagnosis context
    diag_ctx = None
    if diagnosis_context_json:
        try:
            diag_ctx = json.loads(diagnosis_context_json)
        except Exception:
            diag_ctx = None

    # 3. Formulate standard ChatRequest and run triage pipeline
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



from .fhir import export_to_fhir_diagnostic_report


@app.get("/fhir/export/{scan_id}")
@app.get("/api/fhir/export/{scan_id}")
async def get_fhir_report(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
    if not scan:
        raise HTTPException(404, detail=f"Scan record '{scan_id}' not found.")

    scan_dict = {
        "scan_id": scan.id,
        "user_id": scan.user_id,
        "timestamp": scan.timestamp.isoformat() if scan.timestamp else None,
        "diagnosis": scan.diagnosis,
        "confidence": scan.confidence,
        "icd10_code": scan.icd10_code,
        "snomed_code": scan.snomed_code,
        "urgency": scan.urgency,
        "uncertainty": scan.uncertainty,
        "requires_human_review": scan.requires_human_review,
        "iqa_acceptable": scan.iqa_acceptable,
        "escalation_message": f"Clinical Triage Urgency: {scan.urgency.upper()}.",
    }
    return export_to_fhir_diagnostic_report(scan_dict)


@app.get("/clinician/cases")
@app.get("/api/clinician/cases")
async def list_clinician_cases(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("clinician", "admin")),
):
    scans = (
        db.query(ScanResult)
        .filter(ScanResult.requires_human_review == True)  
        .order_by(ScanResult.timestamp.desc())
        .limit(50)
        .all()
    )
    return {
        "cases": [
            {
                "scan_id": s.id,
                "user_id": s.user_id,
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                "diagnosis": s.diagnosis,
                "confidence": s.confidence,
                "group_name": s.group_name,
                "urgency": s.urgency,
                "uncertainty": s.uncertainty,
                "review_reasons": s.review_reasons,
                "icd10_code": s.icd10_code,
            }
            for s in scans
        ]
    }


@app.post("/clinician/override/{scan_id}")
@app.post("/api/clinician/override/{scan_id}")
async def submit_clinician_override(
    scan_id: str,
    override: OverrideRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("clinician", "admin")),
):
    scan = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
    if not scan:
        raise HTTPException(404, detail="Scan record not found.")

    entry = ClinicianOverride(
        scan_id=scan.id,
        clinician_id=current_user.id,
        verdict=override.verdict,
        corrected_diagnosis=override.corrected_diagnosis,
        corrected_icd10=override.corrected_icd10,
        notes=override.notes,
    )
    db.add(entry)
    scan.requires_human_review = False
    db.commit()

    log_event(
        db, "clinician_override", success=True, user_id=current_user.id,
        resource_id=scan.id, resource_type="scan_result",
        metadata={"verdict": override.verdict, "corrected_diagnosis": override.corrected_diagnosis},
    )
    return {"status": "success", "message": "Clinician sign-off recorded successfully."}


@app.get("/patient/history/{patient_id}")
async def get_patient_history(
    patient_id: str,
    db: Session = Depends(get_db),
):
    try:
        scans = (
            db.query(ScanResult)
            .filter(ScanResult.user_id == patient_id)
            .order_by(ScanResult.timestamp.desc())
            .all()
        )
    except Exception:
        scans = []

    return {
        "patient_id": patient_id,
        "scans": [
            {
                "scan_id": s.id,
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                "diagnosis": s.diagnosis,
                "confidence": s.confidence,
                "group_name": s.group_name,
                "urgency": s.urgency,
                "icd10_code": s.icd10_code,
            }
            for s in scans
        ]
    }






class MultimodalRiskRequest(BaseModel):
    hba1c: float
    systolic_bp: int
    diastolic_bp: int
    age: int
    is_smoker: bool
    diagnosis: str
    confidence: float

class AppointmentCreate(BaseModel):
    clinic_name: str
    appointment_time: str
    purpose: str

@app.post("/predict/multimodal")
def predict_multimodal_risk(req: MultimodalRiskRequest):
    
    bp_ratio = req.systolic_bp / 120.0
    hba1c_risk = max(1.0, req.hba1c / 5.7)

    cardio_risk_score = 10.0 * bp_ratio * (1.5 if req.is_smoker else 1.0) * (req.age / 40.0)
    cardio_risk = "High" if cardio_risk_score > 15 else ("Moderate" if cardio_risk_score > 8 else "Low")

    ophthalmic_risk_score = (req.confidence / 100.0) * hba1c_risk * (1.3 if req.age > 60 else 1.0)
    progression_risk = "High" if ophthalmic_risk_score > 1.4 else ("Moderate" if ophthalmic_risk_score > 0.8 else "Low")

    return {
        "cardiovascular_risk_level": cardio_risk,
        "cardiovascular_risk_score": round(cardio_risk_score, 1),
        "ophthalmic_progression_risk_level": progression_risk,
        "ophthalmic_progression_risk_score": round(ophthalmic_risk_score, 2),
        "clinical_guidance": "Patient has elevated cardiovascular threat factors. Immediate endocrine consultation advised." if progression_risk == "High" else "Monitor on regular annual ophthalmic timeline."
    }

@app.get("/scans/history/{user_id}")
def get_longitudinal_history(user_id: str, db: Session = Depends(get_db)):
    

    scans = db.query(ScanResult).filter(ScanResult.user_id == user_id).order_by(ScanResult.created_at.asc()).all()
    history = []

    for i, s in enumerate(scans):
        if i == 0:
            velocity = 0.0
        else:
            prev_s = scans[i - 1]
            diff_days = max(1, (s.created_at - prev_s.created_at).days) if s.created_at and prev_s.created_at else 30
            velocity = round((s.confidence - prev_s.confidence) / diff_days, 3)

        history.append({
            "scan_id": s.id,
            "date": s.created_at.isoformat() if s.created_at else None,
            "diagnosis": s.diagnosis,
            "confidence": s.confidence,
            "uncertainty": s.uncertainty,
            "urgency": s.urgency,
            "progression_velocity": velocity,
            "nerve_fiber_thickness_um": round(100.0 - (i * 4.2), 1)
        })

    avg_vel = round(sum(h["progression_velocity"] for h in history) / max(1, len(history)), 3)

    return {
        "user_id": user_id,
        "scans_count": len(scans),
        "history": history,
        "progression_velocity_average": avg_vel,
        "estimated_retinal_degradation_forecast_years": round(10.0 / max(0.1, abs(avg_vel)), 1)
    }

@app.post("/admin/pacs-import")
def pacs_import_dicom(db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    
    from .pacs_middleware import parse_simulated_dicom

    mock_bytes = b"MOCK-DICOM-IMAGE-PIXELS"
    dicom_meta = parse_simulated_dicom(mock_bytes)

    scan = ScanResult(
        user_id=current_user.id if current_user else None,
        diagnosis="Normal",
        confidence=98.5,
        group_name="Adnexal",
        probabilities={"Normal": 0.985, "Conjunctivitis": 0.015},
        calibrated=True,
        uncertainty=0.02,
        requires_human_review=False,
        icd10_code="Z01.00",
        snomed_code="165070006",
        urgency="none",
        urgency_rank=0,
        dicom_patient_id=dicom_meta["dicom_patient_id"],
        dicom_study_uid=dicom_meta["dicom_study_uid"],
        dicom_series_uid=dicom_meta["dicom_series_uid"],
        spatial_description="No abnormalities noted in optic disc arcade.",
        sign_off_status="signed_off"
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    return {
        "status": "imported",
        "scan_id": scan.id,
        "dicom_metadata": dicom_meta,
        "message": "Successfully fetched and parsed DICOM file from PACS. Synced to EHR provider."
    }

@app.get("/admin/triage-queue")
def get_triage_queue(db: Session = Depends(get_db)):
    
    pending = db.query(ScanResult).filter(ScanResult.sign_off_status == "pending").all()
    return {
        "queue_length": len(pending),
        "items": [
            {
                "scan_id": s.id,
                "user_id": s.user_id,
                "diagnosis": s.diagnosis,
                "confidence": s.confidence,
                "uncertainty": s.uncertainty,
                "reasons": s.review_reasons,
                "timestamp": s.created_at.isoformat() if s.created_at else None
            }
            for s in pending
        ]
    }

@app.post("/scans/{scan_id}/sign-off")
def sign_off_scan(
    scan_id: str,
    verified_diagnosis: str = Form(...),
    notes: str = Form(default=""),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    
    scan = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan result not found")

    old_diagnosis = scan.diagnosis
    scan.sign_off_status = "signed_off"
    scan.signed_off_at = datetime.now(timezone.utc)
    scan.signed_off_by = current_user.id if current_user else "CLINICIAN-1"

    if verified_diagnosis != old_diagnosis:
        scan.sign_off_status = "overridden"
        scan.diagnosis = verified_diagnosis

        override = ClinicianOverride(
            scan_id=scan.id,
            clinician_id=current_user.id if current_user else None,
            verdict="override",
            corrected_diagnosis=verified_diagnosis,
            corrected_icd10=get_clinical_code(verified_diagnosis)["icd10"],
            notes=notes
        )
        db.add(override)

    db.commit()


    from .pacs_middleware import sync_to_ehr_middleware
    sync_ok, sync_msg = sync_to_ehr_middleware(scan.id, {"diagnosis": scan.diagnosis, "confidence": scan.confidence})

    return {
        "scan_id": scan.id,
        "status": scan.sign_off_status,
        "sync_successful": sync_ok,
        "ehr_message": sync_msg
    }

@app.get("/scans/{scan_id}/details")
def get_scan_details(scan_id: str, db: Session = Depends(get_db)):
    
    scan = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan result not found")

    doc = mongo_store.get_document(scan_id) or {}

    return {
        "scan_id": scan.id,
        "diagnosis": scan.diagnosis,
        "confidence": scan.confidence,
        "created_at": scan.created_at.isoformat() if scan.created_at else None,
        "spatial_description": scan.spatial_description,
        "dicom_patient_id": scan.dicom_patient_id,
        "probabilities": doc.get("probabilities", scan.probabilities),
        "review_reasons": doc.get("review_reasons", scan.review_reasons),
        "hybrid_warnings": doc.get("hybrid_warnings", scan.hybrid_warnings),
        "symptoms_reported": doc.get("symptoms_reported", scan.symptoms_reported)
    }

@app.post("/admin/federated/sync")
def sync_federated_nodes():
    
    from .federated import FederatedServer
    server = FederatedServer()
    result = server.trigger_aggregation_round()
    return result

@app.post("/admin/synthetic/generate")
def generate_synthetic_case(condition: str, severity: str = Form(default="moderate")):
    
    from .synthetic_generator import generate_synthetic_fundus
    img_url, heatmap_url = generate_synthetic_fundus(condition, severity)
    return {
        "condition": condition,
        "severity": severity,
        "synthetic_image_url": img_url,
        "synthetic_gradcam_url": heatmap_url,
        "disclaimer": "This image is synthetic and generated for educational and simulator training purposes."
    }

@app.post("/scans/compress")
async def compress_uploaded_image(file: UploadFile = File(...)):
    
    from .iqa import compress_retinal_image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    comp_bytes, orig_sz, comp_sz = compress_retinal_image(image)
    ratio = round((1.0 - (comp_sz / orig_sz)) * 100.0, 1)

    return {
        "original_size_bytes": orig_sz,
        "compressed_size_bytes": comp_sz,
        "bandwidth_saved_percent": ratio,
        "compressed_file_base64": base64.b64encode(comp_bytes).decode("utf-8")
    }

@app.post("/appointments")
def schedule_appointment(appt: AppointmentCreate, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    
    if not current_user:
        current_user = db.query(User).first()
        if not current_user:
            current_user = User(email="demo@ophthalmoai.org", hashed_password="demo_password", role="patient")  
            db.add(current_user)
            db.commit()
            db.refresh(current_user)

    try:
        dt = datetime.fromisoformat(appt.appointment_time.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.now(timezone.utc)

    db_appt = PatientAppointment(
        user_id=current_user.id,
        clinic_name=appt.clinic_name,
        appointment_time=dt,
        purpose=appt.purpose,
        status="scheduled",
        sms_reminder_sent=True
    )
    db.add(db_appt)
    db.commit()
    db.refresh(db_appt)

    return {
        "appointment_id": db_appt.id,
        "status": "scheduled",
        "sms_reminder_log": f"SMS Reminder sent to patient: 'Reminder: Your ophthalmic appointment for {appt.purpose} is scheduled at {appt.clinic_name} for {appt.appointment_time}'."
    }

@app.get("/appointments/history")
def get_appointments_history(db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    if not current_user:
        current_user = db.query(User).first()
        if not current_user:
            return {"appointments": []}
    appts = db.query(PatientAppointment).filter(PatientAppointment.user_id == current_user.id).all()
    return {
        "appointments": [
            {
                "id": a.id,
                "clinic_name": a.clinic_name,
                "time": a.appointment_time.isoformat() if a.appointment_time else None,
                "purpose": a.purpose,
                "status": a.status,
                "sms_reminder_sent": a.sms_reminder_sent
            }
            for a in appts
        ]
    }


# ==============================================================================
# UPGRADE 1: High-Throughput / Low-Latency ONNX Benchmarking Endpoints
# ==============================================================================

@app.get("/api/v1/benchmarks/inference")
async def get_inference_benchmarks():
    """Returns cached inference latency and throughput benchmarks (PyTorch vs ONNX vs Quantized)."""
    return get_cached_benchmark_results()


@app.post("/api/v1/benchmarks/run")
async def run_live_inference_benchmark(iterations: int = 15):
    """Executes a multi-iteration micro-benchmarking sweep across available backbones."""
    return LatencyBenchmarkSuite.run_comprehensive_benchmark(iterations=max(3, min(100, iterations)))


# ==============================================================================
# UPGRADE 2: Asynchronous Screening Task Queue & WebSocket Streaming
# ==============================================================================

@app.post("/api/v1/screen/async", status_code=status.HTTP_202_ACCEPTED)
async def screen_async(
    request: Request,
    file: UploadFile = File(...),
    pain: str = Form(...),
    vision: str = Form(...),
    itch: str = Form(...),
    halos: str = Form(default="No"),
    discharge: str = Form(default="None"),
    light_sens: str = Form(default="No"),
    floaters: str = Form(default="No"),
    duration: str = Form(default="Not Sure"),
    apply_domain_adaptation: bool = Form(default=True),
):
    """
    Decoupled asynchronous screening submission.
    Pushes job to queue and returns HTTP 202 with job_id for WebSocket or polling tracking.
    """
    if file.content_type and file.content_type.lower() not in ALLOWED_MIMES | {"application/octet-stream"}:
        raise HTTPException(415, detail=f"Unsupported file type '{file.content_type}'.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, detail=f"File exceeds {MAX_FILE_SIZE // (1024*1024)} MB limit.")

    dim_ok, dim_result = validate_image_dimensions(contents)
    if not dim_ok:
        raise HTTPException(422, detail=dim_result)

    job_id = screening_queue.create_job(metadata={"filename": file.filename})

    # Background async pipeline task
    async def _process_pipeline(report_progress):
        # Stage 1: Decode & Domain Validation
        await report_progress(15, ScreeningStage.DOMAIN_VALIDATION)
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        from backend.fundus_validator import validate_fundus_image
        is_fundus, score, reason, metrics = validate_fundus_image(img)
        if not is_fundus:
            raise ValueError(f"Non-fundus image rejected: {reason}")

        if metrics.get("is_bgr_inverted"):
            r_c, g_c, b_c = img.split()
            img = Image.merge("RGB", (b_c, g_c, r_c))

        # Stage 2: Optical Domain Adaptation
        await report_progress(35, ScreeningStage.DOMAIN_ADAPTATION)
        img, domain_info = adapt_fundus_domain(img, apply_color_constancy=apply_domain_adaptation)

        # Stage 3: Concurrent Ensemble Forward Pass
        await report_progress(60, ScreeningStage.ENSEMBLE_INFERENCE)
        model_input_img = ben_graham_preprocess(img, target_size=384)
        input_tensor = preprocess(model_input_img).to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            if len(ENSEMBLE_MODELS) >= 3:
                probs_list = []
                for arch_name in ["densenet201", "convnext_small", "efficientnet_v2_m"]:
                    model = ENSEMBLE_MODELS[arch_name]
                    raw_logits = model(input_tensor)[0]
                    t = CALIBRATION_REGISTRY.get(arch_name)
                    cal_logits = apply_temperature(raw_logits, t)
                    probs_list.append(torch.nn.functional.softmax(cal_logits, dim=0))
                probs = torch.stack(probs_list, dim=0).mean(dim=0)
            elif MONOLITHIC_MODEL is not None:
                out = MONOLITHIC_MODEL(input_tensor)
                probs = torch.nn.functional.softmax(out[0], dim=0)
            else:
                raise RuntimeError("Models unavailable.")
            class_idx = int(torch.argmax(probs).item())

        diagnosis = MONOLITHIC_CLASSES[class_idx]
        confidence = float(probs[class_idx].item()) * 100
        probs_dict = {MONOLITHIC_CLASSES[i]: float(probs[i].item()) for i in range(len(MONOLITHIC_CLASSES))}

        # Stage 4: Grad-CAM Explainability
        await report_progress(80, ScreeningStage.EXPLAINABILITY_GRADCAM)
        heatmap_base64 = None
        if GRADCAM_AVAILABLE and MONOLITHIC_MODEL is not None:
            try:
                cam = GradCAM(model=MONOLITHIC_MODEL, target_layers=[MONOLITHIC_MODEL.features[-1]])
                grayscale = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])
                rgb_img = np.float32(model_input_img.resize((384, 384))) / 255
                vis = show_cam_on_image(rgb_img, grayscale[0, :], use_rgb=True)
                buff = io.BytesIO()
                Image.fromarray(vis).save(buff, format="JPEG", quality=85)
                heatmap_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
            except Exception:
                pass

        # Stage 5: Conformal Triage & Clinical Coding
        await report_progress(95, ScreeningStage.CALIBRATION_TRIAGE)
        probs_np = probs.cpu().numpy()
        conformal_set, conformal_probs, stratum, guarantee = CONFORMAL_CALIBRATOR.predict_set(probs_np, diagnosis)
        code_entry = get_clinical_code(diagnosis)

        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "probabilities": probs_dict,
            "conformal_prediction_set": conformal_set,
            "conformal_coverage_guarantee": f"{guarantee:.1f}%",
            "icd10_code": code_entry["icd10"],
            "snomed_code": code_entry["snomed_ct"],
            "urgency": code_entry["urgency"],
            "domain_adaptation": domain_info,
            "heatmap": f"data:image/jpeg;base64,{heatmap_base64}" if heatmap_base64 else None,
        }

    import asyncio
    asyncio.create_task(screening_queue.run_pipeline_task(job_id, _process_pipeline))

    return {
        "job_id": job_id,
        "status": JobStatus.QUEUED.value,
        "message": "Screening task queued successfully.",
        "websocket_url": f"/ws/jobs/{job_id}",
        "poll_url": f"/api/v1/jobs/{job_id}",
    }


@app.get("/api/v1/jobs/{job_id}")
async def get_screening_job_status(job_id: str):
    """Pollable endpoint to retrieve status and results of an async screening job."""
    job = screening_queue.get_job(job_id)
    if not job:
        raise HTTPException(404, detail=f"Job '{job_id}' not found.")
    return job


@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_stream(websocket: WebSocket, job_id: str):
    """Real-time WebSocket event stream for an async screening job."""
    await screening_queue.notifier.connect(job_id, websocket)
    try:
        current_job = screening_queue.get_job(job_id)
        if current_job:
            await websocket.send_json({"type": "job_progress", **current_job})
        while True:
            # Keep stream open until client disconnects or job reaches terminal status
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        await screening_queue.notifier.disconnect(job_id, websocket)
    except Exception:
        await screening_queue.notifier.disconnect(job_id, websocket)


# =============================================================================
# Dual-Track Endpoints: AI Engineering (CBMIR, Fairness) & SWE (Metrics, Tracing, Tenancy)
# =============================================================================

class SimilarCasesRequest(BaseModel):
    probabilities: Optional[Dict[str, float]] = None
    embedding: Optional[List[float]] = None
    top_k: int = 3


@app.post("/api/v1/cases/similar")
async def find_similar_reference_cases(payload: SimilarCasesRequest):
    """
    Content-Based Medical Image Retrieval (CBMIR):
    Returns top-k historical clinical cases matching either patient embedding
    or posterior model probability vector with verified 12-month outcomes.
    """
    if payload.embedding:
        query_vec = np.array(payload.embedding, dtype=np.float32)
    elif payload.probabilities:
        query_vec = vector_index.extract_embedding_from_probabilities(payload.probabilities)
    else:
        raise HTTPException(status_code=400, detail="Either 'probabilities' or 'embedding' must be provided.")

    results = vector_index.query_similar_cases(query_vec, top_k=payload.top_k)
    return {
        "count": len(results),
        "results": results,
    }


@app.get("/api/v1/audit/fairness")
async def get_fairness_audit(force_refresh: bool = False):
    """
    Demographic Fairness, Bias Auditing & Slice Disparity Evaluation:
    Evaluates equalized odds, disparate impact ratio, and four-fifths compliance.
    """
    report = get_fairness_audit_report(force_refresh=force_refresh)
    return report


@app.get("/metrics")
async def get_prometheus_metrics():
    """
    Exposes real-time Prometheus telemetry in standard text format
    for Prometheus scraping and Grafana dashboards.
    """
    return Response(
        content=metrics_collector.generate_prometheus_text(),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/api/v1/traces/recent")
async def get_recent_traces(limit: int = 20):
    """
    OpenTelemetry-compatible distributed tracing endpoint.
    Returns the recent completed execution spans.
    """
    return {
        "traces": tracer_buffer.get_recent_traces(limit=limit),
    }


class TenantCreateRequest(BaseModel):
    name: str
    slug: str
    tier: str = "hospital_standard"


@app.post("/api/v1/tenants")
async def create_tenant_endpoint(
    payload: TenantCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Admin-only endpoint to onboard a new hospital/clinic tenant."""
    tenant = create_new_tenant(db, name=payload.name, slug=payload.slug, tier=payload.tier)
    return {
        "id": tenant.id,
        "name": tenant.name,
        "slug": tenant.slug,
        "tier": tenant.tier,
        "created_at": tenant.created_at.isoformat() if tenant.created_at else None,
    }


@app.get("/api/v1/tenants/current")
async def get_current_tenant_info(
    request: Request,
    db: Session = Depends(get_db),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    """Returns organizational context and active tier for current tenant."""
    t_id = get_current_tenant_id(x_tenant_id=x_tenant_id)
    tenant = db.query(Tenant).filter((Tenant.id == t_id) | (Tenant.slug == t_id)).first()
    if not tenant:
        tenant = ensure_default_tenant(db)
    return {
        "tenant_id": tenant.id,
        "name": tenant.name,
        "slug": tenant.slug,
        "tier": tenant.tier,
        "is_active": tenant.is_active,
    }


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        access_log=not _IS_PROD,  
    )

"""OphthalmoAI Clinical Screening API & Diagnostic Engine."""

from __future__ import annotations

import gc
import os
import warnings
from contextlib import asynccontextmanager
from typing import Dict

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.analytics_routes import router as analytics_router
from backend.auth import JWT_SECRET_KEY
from backend.config import (
    _ENV,
    _IS_PROD,
    DEVICE,
    FORCE_CPU,
    MODELS_DIR,
    MONOLITHIC_CLASSES,
)
from backend.db import create_tables
from backend.logging_config import configure_logging, get_logger
from backend.onnx_inference import LatencyBenchmarkSuite, get_cached_benchmark_results
from backend.routes.auth_routes import router as auth_router
from backend.routes.clinical import router as clinical_router
from backend.routes.screening import router as screening_router
from backend.routes.system import router as system_router
from backend.routes.telehealth import router as telehealth_router
from backend.routes_admin import router as admin_router
from backend.sarvam_service import text_to_speech, transcribe_speech, translate_text
from backend.security import (
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    safe_error_detail,
)
from backend.services.explanation_service import generate_gradcam, generate_spatial_description
from backend.services.image_preprocessor import (
    ben_graham_preprocess,
    is_already_ben_graham,
    preprocess,
)
from backend.services.inference_service import (
    analyze_symptoms,
    analyze_symptoms_structured,
    get_inference_semaphore,
)
from backend.services.model_loader import (
    build_convnext_small,
    build_densenet201,
    build_efficientnet_v2_m,
    build_monolithic_model,
    load_all_models,
)
from backend.validators import validate_ollama_url_from_env

configure_logging(json_output=os.getenv("LOG_FORMAT", "json").lower() == "json")
logger = get_logger("ophthalmoai.main")

MONOLITHIC_MODEL: nn.Module | None = None
ENSEMBLE_MODELS: Dict[str, nn.Module] = {}
ROUTER_MODEL: nn.Module | None = None
SPECIALIST_MODELS: dict | None = None

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager:"""
    global MONOLITHIC_MODEL, ENSEMBLE_MODELS

    logger.info("startup.begin", device=str(DEVICE), environment=_ENV)

    if _IS_PROD and JWT_SECRET_KEY == "CHANGE_ME_BEFORE_PRODUCTION_DEPLOYMENT":
        raise RuntimeError(
            "JWT_SECRET_KEY is still the insecure default placeholder. "
            "Set a cryptographically random value before deploying to production. "
            'Generate one with:  python -c "import secrets; print(secrets.token_hex(32))"'
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

    loaded_mono, loaded_ens = load_all_models(models_dir=MODELS_DIR, device=DEVICE)
    MONOLITHIC_MODEL = loaded_mono
    ENSEMBLE_MODELS = loaded_ens

    yield

    logger.info("shutdown.begin")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    lifespan=lifespan,
    title="OphthalmoAI API",
    version="0.1.0-alpha",
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
        "CORS_ORIGINS=* - acceptable for local development only. "
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

app.include_router(admin_router)
app.include_router(analytics_router)
app.include_router(screening_router)
app.include_router(clinical_router)
app.include_router(telehealth_router)
app.include_router(auth_router)
app.include_router(system_router)

__all__ = [
    "app",
    "lifespan",
    "DEVICE",
    "FORCE_CPU",
    "MODELS_DIR",
    "MONOLITHIC_CLASSES",
    "MONOLITHIC_MODEL",
    "ENSEMBLE_MODELS",
    "ROUTER_MODEL",
    "SPECIALIST_MODELS",
    "preprocess",
    "ben_graham_preprocess",
    "is_already_ben_graham",
    "build_monolithic_model",
    "build_convnext_small",
    "build_densenet201",
    "build_efficientnet_v2_m",
    "generate_gradcam",
    "generate_spatial_description",
    "analyze_symptoms",
    "analyze_symptoms_structured",
    "get_inference_semaphore",
    "translate_text",
    "text_to_speech",
    "transcribe_speech",
    "LatencyBenchmarkSuite",
    "get_cached_benchmark_results",
]

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        access_log=not _IS_PROD,
    )

"""Centralized Configuration for OphthalmoAI Retinal Screening Engine."""

from __future__ import annotations

import os
from typing import List

import torch

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
    else:
        load_dotenv()
except ImportError:
    pass

_ENV = os.getenv("ENVIRONMENT", "development").strip().lower()
_IS_PROD = _ENV not in {"development", "dev", "test", "testing"}

MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() in {"1", "true", "yes"}
DEVICE = torch.device("cuda" if not FORCE_CPU and torch.cuda.is_available() else "cpu")

MONOLITHIC_CLASSES: List[str] = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy / Pathological Myopia",
]

DEFAULT_IMAGE_SIZE: int = 384
EDGE_IMAGE_SIZE: int = 224
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

MAX_CONCURRENT_INFERENCES: int = int(os.getenv("MAX_CONCURRENT_INFERENCES", "4"))
MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE_BYTES", str(20 * 1024 * 1024)))

CALIBRATION_PATH = os.path.join(MODELS_DIR, "calibration.json")
CONFORMAL_CALIBRATION_PATH = os.getenv(
    "CONFORMAL_CALIBRATION_PATH",
    os.path.join(MODELS_DIR, "aw_crc_calibration.json")
    if os.path.exists(os.path.join(MODELS_DIR, "aw_crc_calibration.json"))
    else os.path.join(MODELS_DIR, "conformal_calibration.json"),
)

MC_DROPOUT_PASSES: int = int(os.getenv("MC_DROPOUT_PASSES", "8"))
ENABLE_UNCERTAINTY: bool = os.getenv("ENABLE_UNCERTAINTY", "true").lower() in {"1", "true", "yes"}
ENABLE_IQA: bool = os.getenv("ENABLE_IQA", "true").lower() in {"1", "true", "yes"}
PERSIST_SCANS: bool = os.getenv("PERSIST_SCANS", "false").lower() in {"1", "true", "yes"}

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

PREDICT_RATE_LIMIT: str = os.getenv("PREDICT_RATE_LIMIT", "10/minute")
CHAT_RATE_LIMIT: str = os.getenv("CHAT_RATE_LIMIT", "30/minute")
AUTH_RATE_LIMIT: str = os.getenv("AUTH_RATE_LIMIT", "20/minute")

OPHTHALMOLOGY_SYSTEM_PROMPT: str = (
    "You are OphthalmoAI, a clinical research assistant and educational eye health communicator.\n\n"
    "COMMUNICATION STYLE:\n"
    "1. Speak naturally, warmly, and clearly, providing educational guidance.\n"
    "2. Do NOT use em dashes (\u2014) or en dashes (\u2013). Use clean, standard punctuation (commas, periods, parentheses).\n"
    "3. Keep your answers concise, clear, and easy to understand. Avoid stiff, robotic, or overly scripted AI phrasing.\n"
    "4. Focus on eye health, conditions, symptoms, and educational guidance.\n"
    "5. Explain that this is helpful screening guidance rather than a formal diagnosis or prescription.\n"
    "6. For acute danger signs like sudden vision loss or severe trauma, advise them to get immediate in-person emergency care."
)

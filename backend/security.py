from __future__ import annotations

import hashlib
import ipaddress
import os
import time
import uuid
from typing import Dict, Optional, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .logging_config import get_logger

logger = get_logger("security")

_ENV = os.getenv("ENVIRONMENT", "development").strip().lower()
_IS_PROD = _ENV not in {"development", "dev", "test", "testing"}

_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self' data:; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "upgrade-insecure-requests"
)

_SECURITY_HEADERS: Dict[str, str] = {
    "X-Frame-Options":           "DENY",
    "X-Content-Type-Options":    "nosniff",
    "Referrer-Policy":           "strict-origin-when-cross-origin",
    "Permissions-Policy":        "camera=(self), microphone=(), geolocation=()",
    "Content-Security-Policy":   _CSP,
    "X-XSS-Protection":          "0",
    "Cache-Control":             "no-store",
    "Pragma":                    "no-cache",
}

_HSTS = "max-age=63072000; includeSubDomains; preload"

_HEADERS_TO_REMOVE = {"x-powered-by", "server"}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: ASGIApp, is_production: bool = _IS_PROD) -> None:
        super().__init__(app)
        self._is_production = is_production

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)

        for name, value in _SECURITY_HEADERS.items():
            response.headers[name] = value

        if self._is_production:
            response.headers["Strict-Transport-Security"] = _HSTS

        for h in _HEADERS_TO_REMOVE:
            if h in response.headers:
                del response.headers[h]

        response.headers["Server"] = "OphthalmoAI"

        return response


class RequestIDMiddleware(BaseHTTPMiddleware):

    _HEADER = "X-Request-ID"
    _UUID_RE = __import__("re").compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        __import__("re").IGNORECASE,
    )

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get(self._HEADER, "")
        request_id = (
            incoming if self._UUID_RE.match(incoming) else str(uuid.uuid4())
        )
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers[self._HEADER] = request_id
        return response


_MAGIC: Dict[str, list] = {
    "image/jpeg": [(0, b"\xff\xd8\xff")],
    "image/png":  [(0, b"\x89PNG\r\n\x1a\n")],
    "image/bmp":  [(0, b"BM")],
    "image/webp": [(0, b"RIFF"), (8, b"WEBP")],
    "image/gif":  [(0, b"GIF87a"), (0, b"GIF89a")],
}

ALLOWED_MIMES = frozenset(_MAGIC.keys()) | {"image/jpg"}

_MAX_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))
MIN_IMAGE_DIMENSION_PX = 128
MAX_IMAGE_AXIS_PX = 6000
MAX_IMAGE_ASPECT_RATIO = 4.0


def validate_magic_bytes(
    data: bytes,
    declared_mime: Optional[str] = None,
) -> Tuple[bool, str]:
    if len(data) < 12:
        return False, "File too small to be a valid image"

    detected: Optional[str] = None

    for mime, sigs in _MAGIC.items():
        if mime == "image/gif":
            if any(
                len(data) >= off + len(sig) and data[off : off + len(sig)] == sig
                for off, sig in sigs
            ):
                detected = mime
                break
        else:
            if all(
                len(data) >= off + len(sig) and data[off : off + len(sig)] == sig
                for off, sig in sigs
            ):
                detected = mime
                break

    if detected is None:
        return False, "File content does not match any permitted image format"

    if declared_mime:
        normalised = declared_mime.lower().replace("image/jpg", "image/jpeg")
        if normalised not in ("application/octet-stream", detected):
            logger.warning(
                "security.mime_mismatch",
                declared=declared_mime,
                detected=detected,
            )
            return False, (
                f"Content-Type mismatch: declared '{declared_mime}' "
                f"but magic bytes indicate '{detected}'. Upload rejected."
            )

    return True, detected


def validate_image_dimensions(data: bytes) -> Tuple[bool, str]:
    try:
        import io
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = _MAX_PIXELS
        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
            if w < MIN_IMAGE_DIMENSION_PX or h < MIN_IMAGE_DIMENSION_PX:
                return False, (
                    f"Image dimensions {w}x{h} are below the minimum required resolution "
                    f"({MIN_IMAGE_DIMENSION_PX}x{MIN_IMAGE_DIMENSION_PX} px) for clinical screening."
                )

            if w > MAX_IMAGE_AXIS_PX or h > MAX_IMAGE_AXIS_PX:
                return False, (
                    f"Image axis dimension ({max(w, h)} px) exceeds maximum allowed size "
                    f"({MAX_IMAGE_AXIS_PX} px)."
                )

            aspect = max(w / max(h, 1), h / max(w, 1))
            if aspect > MAX_IMAGE_ASPECT_RATIO:
                return False, (
                    f"Abnormal image aspect ratio ({aspect:.2f}:1). Retinal fundus photography "
                    f"requires standard circular or 4:3 optical aspect ratios."
                )

            total = w * h
            if total > _MAX_PIXELS:
                return False, (
                    f"Image dimensions {w}x{h} ({total:,} px) exceed the "
                    f"{_MAX_PIXELS:,} pixel limit"
                )
        return True, f"{w}x{h}"
    except Exception as exc:
        return False, f"Image dimension check failed: {exc}"


def sanitize_filename(raw_filename: Optional[str]) -> str:
    """Sanitizes user-supplied filenames, stripping path traversal and dangerous characters."""
    import re
    if not raw_filename or not isinstance(raw_filename, str):
        return "fundus_scan.jpg"
    base = os.path.basename(raw_filename.replace("\\", "/")).strip()
    base = re.sub(r"[\x00-\x1f\x7f]", "", base)
    name, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        ext = ".jpg"
    clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:64]
    return f"{clean_name or 'scan'}{ext}"


def strip_image_metadata(img: Any) -> Any:
    """Strips EXIF, GPS, camera serials, and custom metadata tags, returning clean RGB image."""
    from PIL import Image
    if img is None:
        return None
    rgb = img.convert("RGB")
    clean_img = Image.new("RGB", rgb.size)
    clean_img.paste(rgb)
    return clean_img


def anonymise_ip(ip: Optional[str]) -> Optional[str]:
    if not ip:
        return None
    try:
        addr = ipaddress.ip_address(ip)
        if isinstance(addr, ipaddress.IPv4Address):
            parts = str(addr).split(".")
            parts[-1] = "0"
            return ".".join(parts)
        packed = addr.packed[:6] + b"\x00" * 10
        return str(ipaddress.IPv6Address(packed))
    except ValueError:
        return None


def safe_error_detail(exc: Exception, *, request_id: Optional[str] = None) -> str:
    logger.error(
        "unhandled_exception",
        exc_type=type(exc).__name__,
        exc_msg=str(exc),
        request_id=request_id,
    )
    if _IS_PROD:
        ref = f" (ref: {request_id})" if request_id else ""
        return f"An internal error occurred{ref}. Please try again or contact support."
    return f"[{type(exc).__name__}] {exc}"


class TokenBlacklist:

    def __init__(self) -> None:
        self._store: Dict[str, float] = {}

    def revoke(self, jti: str, exp: float) -> None:
        self._store[jti] = exp
        self._prune()

    def is_revoked(self, jti: str) -> bool:
        self._prune()
        return jti in self._store

    def _prune(self) -> None:
        now = time.time()
        stale = [k for k, v in self._store.items() if v < now]
        for k in stale:
            del self._store[k]

    def __len__(self) -> int:
        self._prune()
        return len(self._store)

token_blacklist = TokenBlacklist()


class LoginAttemptTracker:

    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: int = 300,
        lockout_seconds: int = 900,
    ) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        self._attempts: Dict[str, list] = {}
        self._lockouts: Dict[str, float] = {}

    @staticmethod
    def _key(email: str) -> str:
        return hashlib.sha256(email.lower().strip().encode()).hexdigest()[:24]

    def is_locked(self, email: str) -> Tuple[bool, int]:
        key = self._key(email)
        if key in self._lockouts:
            remaining = self._lockouts[key] - time.time()
            if remaining > 0:
                return True, int(remaining)
            del self._lockouts[key]
        return False, 0

    def record_failure(self, email: str) -> Tuple[bool, int]:
        key = self._key(email)
        now = time.time()
        cutoff = now - self.window_seconds

        bucket = self._attempts.setdefault(key, [])
        bucket[:] = [t for t in bucket if t > cutoff]
        bucket.append(now)

        if len(bucket) >= self.max_attempts:
            self._lockouts[key] = now + self.lockout_seconds
            del self._attempts[key]
            logger.warning(
                "security.login_locked",
                email_hash=key,
                lockout_seconds=self.lockout_seconds,
            )
            return True, self.lockout_seconds
        return False, 0

    def record_success(self, email: str) -> None:
        key = self._key(email)
        self._attempts.pop(key, None)
        self._lockouts.pop(key, None)

login_tracker = LoginAttemptTracker()


def make_rate_limit_decorator(limit_string: str):
    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address

        _limiter = Limiter(key_func=get_remote_address)
        return _limiter.limit(limit_string)
    except ImportError:
        if _IS_PROD:
            raise RuntimeError(
                "slowapi is required but not installed. "
                "Run: pip install slowapi. "
                "Rate limiting MUST be active in production."
            )
        logger.warning(
            "security.rate_limit_disabled",
            reason="slowapi not installed",
            environment=_ENV,
            limit=limit_string,
        )

        def _noop_decorator(func):
            return func

        return _noop_decorator

from __future__ import annotations
import os
from typing import Optional
from .logging_config import get_logger

logger = get_logger("storage")
BUCKET = os.getenv("SCAN_STORAGE_BUCKET", "")
KMS_KEY_ID = os.getenv("SCAN_STORAGE_KMS_KEY_ID", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import boto3
        from botocore.config import Config
        _client = boto3.client("s3", region_name=AWS_REGION, config=Config(signature_version="s3v4"))
        return _client
    except ImportError:
        logger.warning("storage.boto3_unavailable")
        return None


def store(data: bytes, key: str, content_type: str = "image/jpeg") -> Optional[str]:
    if not BUCKET:
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        extra = {"ServerSideEncryption": "aws:kms"}
        if KMS_KEY_ID:
            extra["SSEKMSKeyId"] = KMS_KEY_ID
        client.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type, **extra)
        return key
    except Exception as exc:
        logger.error("storage.store_failed", key=key, error=str(exc))
        return None


def fetch(key: str) -> Optional[bytes]:
    if not BUCKET or not key:
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.get_object(Bucket=BUCKET, Key=key)
        return response["Body"].read()
    except Exception as exc:
        logger.error("storage.fetch_failed", key=key, error=str(exc))
        return None


def presigned_url(key: str, expires_seconds: int = 300) -> Optional[str]:
    if not BUCKET or not key:
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        return client.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=expires_seconds)
    except Exception as exc:
        logger.error("storage.presign_failed", key=key, error=str(exc))
        return None


def is_configured() -> bool:
    return bool(BUCKET)

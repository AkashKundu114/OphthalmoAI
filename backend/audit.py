from __future__ import annotations
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session
from .db import AuditLog
from .logging_config import get_logger

logger = get_logger("audit")


def log_event(db: Session, action: str, success: bool = True, user_id=None, resource_id=None,
              resource_type=None, ip_address=None, user_agent=None, error_detail=None,
              metadata: Optional[Dict[str, Any]] = None) -> None:
    log_fn = logger.info if success else logger.warning
    log_fn("audit.event", action=action, success=success, user_id=user_id, resource_id=resource_id,
           resource_type=resource_type, ip=ip_address, **(metadata or {}))
    try:
        entry = AuditLog(action=action, success=success, user_id=user_id, resource_id=resource_id,
                          resource_type=resource_type, ip_address=ip_address, user_agent=user_agent,
                          error_detail=error_detail, metadata_=metadata)
        db.add(entry)
        db.commit()
    except Exception as exc:
        logger.error("audit.db_write_failed", action=action, error=str(exc))
        db.rollback()

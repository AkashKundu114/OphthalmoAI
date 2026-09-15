"""
Asynchronous Screening Task Queue & Real-Time WebSocket Streaming Engine.

Provides:
1. Decoupled job queue and worker execution for heavy multi-backbone inference
   without blocking the primary HTTP event loop.
2. Granular stage-by-stage progress reporting (Domain Validation -> Ensemble -> Grad-CAM -> Triage).
3. WebSocket streaming broadcaster for real-time client UI progress updates.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ScreeningStage(str, Enum):
    INITIALIZED = "Job Initialized in Queue"
    DOMAIN_VALIDATION = "Aperture & Chromophore Domain Validation"
    DOMAIN_ADAPTATION = "Optical Color Constancy Normalization"
    ENSEMBLE_INFERENCE = "Concurrent Tri-Backbone Soft-Voting Forward Pass"
    EXPLAINABILITY_GRADCAM = "Generating High-Resolution Grad-CAM Heatmap"
    CALIBRATION_TRIAGE = "Platt Temperature Calibration & Conformal Triage"
    FINALIZING = "Synthesizing Biomarkers & Clinical Codes"
    COMPLETED = "Screening Complete"
    FAILED = "Execution Failed"


class WebSocketJobNotifier:
    """Manages active WebSocket subscriber connections keyed by job_id."""

    def __init__(self):
        self._subscribers: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = set()
            self._subscribers[job_id].add(websocket)

    async def disconnect(self, job_id: str, websocket: WebSocket):
        async with self._lock:
            if job_id in self._subscribers:
                self._subscribers[job_id].discard(websocket)
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]

    async def broadcast(self, job_id: str, message: Dict[str, Any]):
        async with self._lock:
            websockets = list(self._subscribers.get(job_id, []))

        for ws in websockets:
            try:
                await ws.send_json(message)
            except Exception:
                # Disconnection handled in outer loop
                pass


class ScreeningJobManager:
    """
    In-memory task coordinator with WebSocket event broadcasting.
    Architecture is ready for drop-in Redis Pub/Sub and Celery worker queues in distributed clusters.
    """

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.notifier = WebSocketJobNotifier()

    def create_job(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "progress_percent": 0,
            "current_stage": ScreeningStage.INITIALIZED.value,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    async def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress_percent: Optional[int] = None,
        stage: Optional[ScreeningStage] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        if job_id not in self.jobs:
            return

        job = self.jobs[job_id]
        if status is not None:
            job["status"] = status.value
        if progress_percent is not None:
            job["progress_percent"] = max(0, min(100, progress_percent))
        if stage is not None:
            job["current_stage"] = stage.value
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        job["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Broadcast update to any connected WebSockets
        await self.notifier.broadcast(job_id, {
            "type": "job_progress",
            "job_id": job_id,
            "status": job["status"],
            "progress_percent": job["progress_percent"],
            "current_stage": job["current_stage"],
            "result": job["result"],
            "error": job["error"],
            "updated_at": job["updated_at"],
        })

    async def run_pipeline_task(
        self,
        job_id: str,
        pipeline_fn: Callable[[Callable[[int, ScreeningStage], Coroutine]], Coroutine],
    ):
        """
        Executes an asynchronous screening pipeline with real-time stage updates.
        """
        await self.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            progress_percent=5,
            stage=ScreeningStage.INITIALIZED,
        )

        async def report_progress(percent: int, stage: ScreeningStage):
            await self.update_job(
                job_id,
                status=JobStatus.PROCESSING,
                progress_percent=percent,
                stage=stage,
            )

        try:
            result = await pipeline_fn(report_progress)
            await self.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress_percent=100,
                stage=ScreeningStage.COMPLETED,
                result=result,
            )
        except Exception as exc:
            await self.update_job(
                job_id,
                status=JobStatus.FAILED,
                stage=ScreeningStage.FAILED,
                error=str(exc),
            )


# Global singleton manager
screening_queue = ScreeningJobManager()

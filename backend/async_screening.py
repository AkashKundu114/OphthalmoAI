"""
Asynchronous Screening Task Coordinator and Real-Time WebSocket Streaming Engine.

Provides decoupled execution for compute-intensive multi-backbone inference,
bounded job state retention to prevent memory leaks, and real-time WebSocket stage
broadcasting.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set
import uuid

from fastapi import WebSocket


MAX_RETAINED_JOBS = 1000


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
    """Manages active WebSocket subscriber connections with auto-cleanup of dead sockets."""

    def __init__(self):
        self._subscribers: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        """Accepts and registers a client WebSocket for progress updates."""
        await websocket.accept()
        async with self._lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = set()
            self._subscribers[job_id].add(websocket)

    async def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        """Deregisters a WebSocket connection."""
        async with self._lock:
            if job_id in self._subscribers:
                self._subscribers[job_id].discard(websocket)
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]

    async def broadcast(self, job_id: str, message: Dict[str, Any]) -> None:
        """Broadcasts event payload to all active subscribers, pruning broken connections."""
        async with self._lock:
            websockets = list(self._subscribers.get(job_id, []))

        dead_sockets: List[WebSocket] = []
        for ws in websockets:
            try:
                await ws.send_json(message)
            except Exception:
                dead_sockets.append(ws)

        if dead_sockets:
            async with self._lock:
                if job_id in self._subscribers:
                    for ws in dead_sockets:
                        self._subscribers[job_id].discard(ws)
                    if not self._subscribers[job_id]:
                        del self._subscribers[job_id]


class ScreeningJobManager:
    """
    Task coordinator maintaining bounded job state history and orchestrating
    pipeline workers.
    """

    def __init__(self, max_retained_jobs: int = MAX_RETAINED_JOBS):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.notifier = WebSocketJobNotifier()
        self._max_jobs = max(50, max_retained_jobs)
        self._lock = asyncio.Lock()

    def create_job(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Instantiates a new queued job, pruning stale records if limit exceeded."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        if len(self.jobs) >= self._max_jobs:
            # Prune oldest completed or failed jobs
            prune_candidates = [
                k for k, v in self.jobs.items()
                if v.get("status") in {JobStatus.COMPLETED.value, JobStatus.FAILED.value}
            ]
            for old_id in prune_candidates[: len(self.jobs) - self._max_jobs + 10]:
                self.jobs.pop(old_id, None)

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
        """Retrieves current job record by ID."""
        return self.jobs.get(job_id)

    async def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress_percent: Optional[int] = None,
        stage: Optional[ScreeningStage] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Updates job state attributes and broadcasts snapshot to active subscribers."""
        if job_id not in self.jobs:
            return

        job = self.jobs[job_id]
        if status is not None:
            job["status"] = status.value
        if progress_percent is not None:
            job["progress_percent"] = max(0, min(100, int(progress_percent)))
        if stage is not None:
            job["current_stage"] = stage.value
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = str(error)
        job["updated_at"] = datetime.now(timezone.utc).isoformat()

        await self.notifier.broadcast(
            job_id,
            {
                "type": "job_progress",
                "job_id": job_id,
                "status": job["status"],
                "progress_percent": job["progress_percent"],
                "current_stage": job["current_stage"],
                "result": job["result"],
                "error": job["error"],
                "updated_at": job["updated_at"],
            },
        )

    async def run_pipeline_task(
        self,
        job_id: str,
        pipeline_fn: Callable[[Callable[[int, ScreeningStage], Coroutine]], Coroutine],
    ) -> None:
        """Executes asynchronous screening pipeline with granular stage reporting."""
        await self.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            progress_percent=5,
            stage=ScreeningStage.INITIALIZED,
        )

        async def report_progress(percent: int, stage: ScreeningStage) -> None:
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

"""
Tests for backend/async_screening.py (Asynchronous Job Queue & WebSocket Notifier).
"""

import asyncio
import pytest

from backend.async_screening import (
    JobStatus,
    ScreeningJobManager,
    ScreeningStage,
    WebSocketJobNotifier,
)


@pytest.mark.asyncio
async def test_job_lifecycle():
    manager = ScreeningJobManager()
    job_id = manager.create_job(metadata={"test": 123})
    assert job_id.startswith("job_")

    job = manager.get_job(job_id)
    assert job["status"] == JobStatus.QUEUED.value
    assert job["progress_percent"] == 0

    await manager.update_job(
        job_id,
        status=JobStatus.PROCESSING,
        progress_percent=45,
        stage=ScreeningStage.ENSEMBLE_INFERENCE,
    )
    job = manager.get_job(job_id)
    assert job["status"] == JobStatus.PROCESSING.value
    assert job["progress_percent"] == 45
    assert job["current_stage"] == ScreeningStage.ENSEMBLE_INFERENCE.value

    # Mark completed
    await manager.update_job(
        job_id,
        status=JobStatus.COMPLETED,
        progress_percent=100,
        result={"diagnosis": "Normal"},
    )
    job = manager.get_job(job_id)
    assert job["status"] == JobStatus.COMPLETED.value
    assert job["result"]["diagnosis"] == "Normal"


@pytest.mark.asyncio
async def test_run_pipeline_task_success():
    manager = ScreeningJobManager()
    job_id = manager.create_job()

    stages_recorded = []

    async def mock_pipeline(report_progress):
        await report_progress(30, ScreeningStage.DOMAIN_VALIDATION)
        stages_recorded.append("validation")
        await report_progress(70, ScreeningStage.ENSEMBLE_INFERENCE)
        stages_recorded.append("ensemble")
        return {"top_class": "Diabetic Retinopathy", "confidence": 0.95}

    await manager.run_pipeline_task(job_id, mock_pipeline)
    job = manager.get_job(job_id)
    assert job["status"] == JobStatus.COMPLETED.value
    assert job["progress_percent"] == 100
    assert job["result"]["top_class"] == "Diabetic Retinopathy"
    assert stages_recorded == ["validation", "ensemble"]


@pytest.mark.asyncio
async def test_run_pipeline_task_failure():
    manager = ScreeningJobManager()
    job_id = manager.create_job()

    async def failing_pipeline(report_progress):
        await report_progress(20, ScreeningStage.DOMAIN_VALIDATION)
        raise ValueError("Corrupt image data detected")

    await manager.run_pipeline_task(job_id, failing_pipeline)
    job = manager.get_job(job_id)
    assert job["status"] == JobStatus.FAILED.value
    assert "Corrupt image data" in job["error"]

"""Retinal Screening & Prediction Route Handlers."""

from __future__ import annotations

import asyncio
import base64
import gc
import io
from typing import Any, Dict

import numpy as np
import torch
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from PIL import Image
from sqlalchemy.orm import Session

from backend.async_screening import JobStatus, ScreeningStage, screening_queue
from backend.audit import log_event
from backend.auth import get_current_user
from backend.biomarker_extractor import extract_visual_biomarkers
from backend.calibration import apply_temperature
from backend.clinical_codes import get_clinical_code
from backend.config import (
    DEVICE,
    ENABLE_IQA,
    ENABLE_UNCERTAINTY,
    MAX_FILE_SIZE,
    MC_DROPOUT_PASSES,
    MONOLITHIC_CLASSES,
    PERSIST_SCANS,
    PREDICT_RATE_LIMIT,
)
from backend.conformal import ConformalTriagePolicy
from backend.db import ScanResult, User, get_db
from backend.domain_adaptation import adapt_fundus_domain
from backend.fundus_validator import validate_fundus_image
from backend.iqa import assess_image_quality, compress_retinal_image
from backend.logging_config import get_logger
from backend.medical_data import MEDICAL_INFO
from backend.mongodb_client import mongo_store
from backend.multimodal_service import fuse_image_and_biodata
from backend.security import (
    ALLOWED_MIMES,
    anonymise_ip,
    make_rate_limit_decorator,
    safe_error_detail,
    sanitize_filename,
    strip_image_metadata,
    validate_image_dimensions,
    validate_magic_bytes,
)
from backend.services.explanation_service import generate_gradcam, generate_spatial_description
from backend.services.image_preprocessor import ben_graham_preprocess, preprocess
from backend.services.inference_service import (
    CALIBRATION_REGISTRY,
    CONFORMAL_CALIBRATOR,
    analyze_symptoms,
    analyze_symptoms_structured,
    get_inference_semaphore,
)
from backend.uncertainty import build_review_payload, mc_dropout_predict

logger = get_logger("ophthalmoai.screening_routes")
router = APIRouter(tags=["Screening"])

_predict_limit = make_rate_limit_decorator(PREDICT_RATE_LIMIT)


def _client_ip(request: Request) -> str | None:
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else None


@router.post("/predict")
@router.post("/api/predict")
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
    hba1c: float | None = Form(default=None),
    systolic_bp: int | None = Form(default=None),
    diastolic_bp: int | None = Form(default=None),
    patient_age: int | None = Form(default=None),
    iop: float | None = Form(default=None),
    eye_side: str | None = Form(default="OD"),
    visual_acuity_logmar: float | None = Form(default=None),
    is_smoker: bool | None = Form(default=None),
    apply_domain_adaptation: bool = Form(default=True),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    import backend.main as bm

    client_ip = anonymise_ip(_client_ip(request))
    user_id = current_user.id if current_user else None
    req_id = getattr(request.state, "request_id", None)

    monolithic_model = getattr(bm, "MONOLITHIC_MODEL", None)
    ensemble_models = getattr(bm, "ENSEMBLE_MODELS", {})

    if monolithic_model is None and len(ensemble_models) == 0:
        raise HTTPException(503, detail="Prediction models not loaded.")

    if file.content_type and file.content_type.lower() not in ALLOWED_MIMES | {"application/octet-stream"}:
        log_event(
            db,
            "predict",
            success=False,
            user_id=user_id,
            ip_address=client_ip,
            error_detail=f"rejected content-type: {file.content_type}",
        )
        raise HTTPException(415, detail=f"Unsupported file type '{file.content_type}'.")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        log_event(
            db,
            "predict",
            success=False,
            user_id=user_id,
            ip_address=client_ip,
            error_detail="file too large",
        )
        raise HTTPException(413, detail=f"File exceeds the {MAX_FILE_SIZE // (1024 * 1024)} MB limit.")

    magic_ok, magic_result = validate_magic_bytes(contents, file.content_type)
    if not magic_ok:
        log_event(
            db,
            "predict",
            success=False,
            user_id=user_id,
            ip_address=client_ip,
            error_detail=f"magic byte rejection: {magic_result}",
        )
        raise HTTPException(415, detail=magic_result)

    dim_ok, dim_result = validate_image_dimensions(contents)
    if not dim_ok:
        raise HTTPException(422, detail=dim_result)

    try:
        raw_image = Image.open(io.BytesIO(contents))
        image = strip_image_metadata(raw_image)
    except Exception:
        raise HTTPException(422, detail="File could not be decoded as an image.")

    iqa_acceptable: bool = True
    iqa_warnings: list[str] = []
    if ENABLE_IQA:
        try:
            iqa_acceptable, iqa_warnings = assess_image_quality(image)
        except Exception as iqa_err:
            logger.warning("predict.iqa_failed", error=str(iqa_err))

    try:
        is_fundus, fundus_score, fundus_reason, fundus_metrics = validate_fundus_image(image)
        if not is_fundus:
            log_event(
                db,
                "predict.rejected_non_fundus",
                success=False,
                user_id=user_id,
                ip_address=client_ip,
                error_detail=fundus_reason,
            )
            logger.warning("predict.non_fundus_rejected", score=fundus_score, reason=fundus_reason)
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported image type: The uploaded image does not appear to be a retinal fundus photograph. "
                    f"{fundus_reason} Please upload an authentic color fundus scan of the posterior pole."
                ),
            )

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

    sem = get_inference_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=20.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Clinical inference capacity saturated. Please retry in a few moments.",
        )

    try:
        model_input_image = ben_graham_preprocess(image, target_size=384)
        input_tensor = preprocess(model_input_image).to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            if len(ensemble_models) >= 3:
                probs_list = []
                for arch_name in ["densenet201", "convnext_small", "efficientnet_v2_m"]:
                    model = ensemble_models[arch_name]
                    raw_logits = model(input_tensor)[0]
                    t = CALIBRATION_REGISTRY.get(arch_name)
                    cal_logits = apply_temperature(raw_logits, t)
                    probs_list.append(torch.nn.functional.softmax(cal_logits, dim=0))
                probs = torch.stack(probs_list, dim=0).mean(dim=0)
                calibrated_out = torch.log(probs + 1e-8)
                calibration_temperature = float(
                    np.mean([CALIBRATION_REGISTRY.get(k) for k in ["densenet201", "convnext_small", "efficientnet_v2_m"]])
                )
                is_calibrated = True
                model_group_name = "Tri-Backbone Ensemble (85.2% Test Accuracy)"
            elif monolithic_model is not None:
                out = monolithic_model(input_tensor)
                calibration_temperature = (
                    CALIBRATION_REGISTRY.get("efficientnet_b4")
                    if CALIBRATION_REGISTRY.is_calibrated("efficientnet_b4")
                    else CALIBRATION_REGISTRY.get("monolith")
                )
                is_calibrated = CALIBRATION_REGISTRY.is_calibrated("efficientnet_b4") or CALIBRATION_REGISTRY.is_calibrated("monolith")
                calibrated_out = apply_temperature(out[0], calibration_temperature)
                probs = torch.nn.functional.softmax(calibrated_out, dim=0)
                model_group_name = "Monolithic 12-Class Model"
            else:
                raise RuntimeError("No prediction models loaded.")

            if torch.isnan(probs).any() or torch.isinf(probs).any():
                logger.error("predict.numerical_instability", probs=str(probs))
                raise HTTPException(status_code=500, detail="Inference encountered numerical instability in output tensors.")

            class_idx = int(torch.argmax(probs).item())

        diagnosis = MONOLITHIC_CLASSES[class_idx]
        confidence = float(probs[class_idx].item()) * 100
        probs_dict = {MONOLITHIC_CLASSES[i]: float(probs[i].item()) for i in range(len(MONOLITHIC_CLASSES))}

        heatmap_base64: str | None = None
        uncertainty_value: float | None = None
        grayscale: np.ndarray | None = None

        evidence_vec = torch.nn.functional.softplus(calibrated_out)
        alpha_vec = evidence_vec + 1.0
        total_strength = float(torch.sum(alpha_vec).item())
        epistemic_vacuity = float(
            min(1.0, max(0.0, len(MONOLITHIC_CLASSES) / (total_strength if total_strength > 0 else 1.0)))
        )

        probs_np = probs.cpu().numpy()
        conformal_set, conformal_probs, conformal_stratum, coverage_guarantee = CONFORMAL_CALIBRATOR.predict_set_aw_crc(
            probs_np, diagnosis, admissibility_score=fundus_score
        )

        if ENABLE_UNCERTAINTY and monolithic_model is not None:
            try:
                _, uncertainty_value = mc_dropout_predict(
                    monolithic_model, input_tensor, n_passes=MC_DROPOUT_PASSES
                )
            except Exception as unc_err:
                logger.warning("predict.uncertainty_failed", error=str(unc_err))

        preprocessed_base64 = None
        try:
            p_buff = io.BytesIO()
            model_input_image.resize((384, 384)).save(p_buff, format="JPEG", quality=85)
            preprocessed_base64 = base64.b64encode(p_buff.getvalue()).decode("utf-8")
        except Exception:
            pass

        if monolithic_model is not None:
            heatmap_base64, grayscale = generate_gradcam(
                model=monolithic_model,
                input_tensor=input_tensor,
                class_idx=class_idx,
                model_input_image=model_input_image,
            )

        visual_biomarkers = extract_visual_biomarkers(
            image,
            grayscale if grayscale is not None else None,
            diagnosis,
        )

        hybrid_warnings = analyze_symptoms(
            diagnosis,
            pain,
            vision,
            itch,
            halos=halos,
            discharge=discharge,
            light_sensitivity=light_sens,
            floaters=floaters,
            duration=duration,
        )
        hybrid_warnings_structured = analyze_symptoms_structured(
            diagnosis,
            pain,
            vision,
            itch,
            halos=halos,
            discharge=discharge,
            light_sensitivity=light_sens,
            floaters=floaters,
            duration=duration,
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
            requires_human_review_flag=review_payload["requires_human_review"],
        )

        code_entry = get_clinical_code(diagnosis)
        details = MEDICAL_INFO.get(
            diagnosis,
            {
                "description": "No detailed information available.",
                "severity": "Unknown",
                "advice": "Please consult an ophthalmologist.",
                "treatment": [],
                "symptoms": [],
                "precautions": [],
                "analysis": "",
            },
        )

        spatial_desc = generate_spatial_description(diagnosis)

        patient_bio = {
            "patient_age": patient_age,
            "sex": "Female" if str(request.query_params.get("sex", "")).lower() == "female" else "Male",
            "eye_laterality": eye_side,
            "hba1c": hba1c,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "intraocular_pressure": iop,
            "visual_acuity_logmar": visual_acuity_logmar,
        }
        multimodal_eval = fuse_image_and_biodata(probs_dict, patient_bio)

        response_body: Dict[str, Any] = {
            "group_name": model_group_name,
            "models_ensembled": list(ensemble_models.keys()) if len(ensemble_models) >= 3 else ["efficientnet_b4"],
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "preprocessed_image": f"data:image/jpeg;base64,{preprocessed_base64}" if preprocessed_base64 else None,
            "heatmap": f"data:image/jpeg;base64,{heatmap_base64}" if heatmap_base64 else None,
            "details": details,
            "hybrid_warnings": hybrid_warnings,
            "hybrid_warnings_structured": hybrid_warnings_structured,
            "probabilities": probs_dict,
            "multimodal_evaluation": multimodal_eval,
            "patient_biodata_received": patient_bio,
            "calibrated": is_calibrated,
            "calibration_temperature": round(calibration_temperature, 4),
            "uncertainty": review_payload["uncertainty"],
            "requires_human_review": review_payload["requires_human_review"],
            "review_reasons": review_payload["review_reasons"],
            "conformal_prediction_set": conformal_set,
            "conformal_candidate_probabilities": conformal_probs,
            "conformal_stratum": conformal_stratum,
            "conformal_coverage_guarantee": f"{coverage_guarantee:.1f}%",
            "epistemic_vacuity": round(epistemic_vacuity, 4),
            "triage_tier": triage_eval["triage_tier"],
            "triage_urgency": triage_eval["urgency_level"],
            "triage_action_code": triage_eval["action_code"],
            "triage_guidance": triage_eval["guidance"],
            "visual_biomarkers": visual_biomarkers,
            "icd10_code": code_entry["icd10"],
            "snomed_code": code_entry["snomed_ct"],
            "urgency": code_entry["urgency"],
            "urgency_rank": code_entry["urgency_rank"],
            "referral": code_entry["referral"],
            "escalation_message": code_entry["escalation_message"],
            "iqa_acceptable": iqa_acceptable,
            "iqa_warnings": iqa_warnings,
            "optical_admissibility_score": round(fundus_score, 4),
            "guardrail_status": "PASSED" if is_fundus else "FLAGGED",
            "condition_details": MEDICAL_INFO.get(diagnosis, MEDICAL_INFO.get("Normal", {})),
            "spatial_description": spatial_desc,
            "domain_adaptation": domain_eval,
        }

        scan_id = None
        if PERSIST_SCANS:
            try:
                scan = ScanResult(
                    user_id=user_id,
                    diagnosis=diagnosis,
                    confidence=round(confidence, 2),
                    group_name=model_group_name,
                    probabilities=probs_dict,
                    calibrated=is_calibrated,
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
                    iqa_acceptable=iqa_acceptable,
                    iqa_warnings=iqa_warnings,
                    symptoms_reported={
                        "pain": pain,
                        "vision": vision,
                        "itch": itch,
                        "halos": halos,
                        "discharge": discharge,
                        "light_sensitivity": light_sens,
                        "floaters": floaters,
                        "duration": duration,
                    },
                    router_group_idx=0,
                    hba1c=hba1c,
                    systolic_bp=systolic_bp,
                    diastolic_bp=diastolic_bp,
                    patient_age=patient_age,
                    is_smoker=is_smoker,
                    spatial_description=spatial_desc,
                    sign_off_status="pending",
                )
                db.add(scan)
                db.commit()
                db.refresh(scan)
                scan_id = scan.id
                response_body["scan_id"] = scan_id

                mongo_store.insert_document(
                    scan_id,
                    {
                        "probabilities": probs_dict,
                        "review_reasons": review_payload["review_reasons"],
                        "hybrid_warnings": hybrid_warnings,
                        "hybrid_warnings_structured": hybrid_warnings_structured,
                        "iqa_warnings": iqa_warnings,
                        "symptoms_reported": {
                            "pain": pain,
                            "vision": vision,
                            "itch": itch,
                            "halos": halos,
                            "discharge": discharge,
                            "light_sensitivity": light_sens,
                            "floaters": floaters,
                            "duration": duration,
                        },
                    },
                )
            except Exception as persist_err:
                logger.error("predict.persist_failed", error=str(persist_err))
                db.rollback()

        log_event(
            db,
            "predict",
            success=True,
            user_id=user_id,
            resource_id=scan_id,
            resource_type="scan_result",
            ip_address=client_ip,
            metadata={
                "diagnosis": diagnosis,
                "confidence": round(confidence, 2),
                "urgency": code_entry["urgency"],
                "requires_human_review": review_payload["requires_human_review"],
            },
        )
        return response_body

    except HTTPException:
        raise
    except Exception as exc:
        log_event(
            db,
            "predict",
            success=False,
            user_id=user_id,
            ip_address=client_ip,
            error_detail=str(exc),
        )
        raise HTTPException(500, detail=safe_error_detail(exc, request_id=req_id))
    finally:
        try:
            sem.release()
        except (ValueError, RuntimeError):
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


@router.post("/api/v1/screen/async", status_code=status.HTTP_202_ACCEPTED)
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
    """Decoupled asynchronous screening submission."""
    import backend.main as bm

    if file.content_type and file.content_type.lower() not in ALLOWED_MIMES | {"application/octet-stream"}:
        raise HTTPException(415, detail=f"Unsupported file type '{file.content_type}'.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, detail=f"File exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB limit.")

    safe_name = sanitize_filename(file.filename or "scan.jpg")
    magic_ok, magic_result = validate_magic_bytes(contents, file.content_type)
    if not magic_ok:
        raise HTTPException(415, detail=magic_result)

    dim_ok, dim_result = validate_image_dimensions(contents)
    if not dim_ok:
        raise HTTPException(422, detail=dim_result)

    job_id = screening_queue.create_job(metadata={"filename": safe_name})

    async def _process_pipeline(report_progress: Any):
        try:
            await report_progress(15, ScreeningStage.DOMAIN_VALIDATION)
            raw_img = Image.open(io.BytesIO(contents))
            img = strip_image_metadata(raw_img)

            is_fundus, score, reason, metrics = validate_fundus_image(img)
            if not is_fundus:
                raise ValueError(f"Non-fundus image rejected: {reason}")

            if metrics.get("is_bgr_inverted"):
                r_c, g_c, b_c = img.split()
                img = Image.merge("RGB", (b_c, g_c, r_c))

            await report_progress(35, ScreeningStage.DOMAIN_ADAPTATION)
            img, domain_info = adapt_fundus_domain(img, apply_color_constancy=apply_domain_adaptation)

            await report_progress(60, ScreeningStage.ENSEMBLE_INFERENCE)
            model_input_img = ben_graham_preprocess(img, target_size=384)
            input_tensor = preprocess(model_input_img).to(DEVICE).unsqueeze(0)

            monolithic_model = getattr(bm, "MONOLITHIC_MODEL", None)
            ensemble_models = getattr(bm, "ENSEMBLE_MODELS", {})

            with torch.no_grad():
                if len(ensemble_models) >= 3:
                    probs_list = []
                    for arch_name in ["densenet201", "convnext_small", "efficientnet_v2_m"]:
                        model = ensemble_models[arch_name]
                        raw_logits = model(input_tensor)[0]
                        if torch.isnan(raw_logits).any() or torch.isinf(raw_logits).any():
                            raise RuntimeError(f"Numerical instability in {arch_name}: model produced NaN or Inf logits.")
                        t = CALIBRATION_REGISTRY.get(arch_name)
                        cal_logits = apply_temperature(raw_logits, t)
                        probs_list.append(torch.nn.functional.softmax(cal_logits, dim=0))
                    probs = torch.stack(probs_list, dim=0).mean(dim=0)
                elif monolithic_model is not None:
                    out = monolithic_model(input_tensor)
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        raise RuntimeError("Numerical instability: monolithic model produced NaN or Inf logits.")
                    probs = torch.nn.functional.softmax(out[0], dim=0)
                else:
                    raise RuntimeError("Models unavailable.")
                class_idx = int(torch.argmax(probs).item())

            diagnosis = MONOLITHIC_CLASSES[class_idx]
            confidence = float(probs[class_idx].item()) * 100
            probs_dict = {MONOLITHIC_CLASSES[i]: float(probs[i].item()) for i in range(len(MONOLITHIC_CLASSES))}

            await report_progress(80, ScreeningStage.EXPLAINABILITY_GRADCAM)
            heatmap_base64, _ = generate_gradcam(
                model=monolithic_model,
                input_tensor=input_tensor,
                class_idx=class_idx,
                model_input_image=model_input_img,
            )

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
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    asyncio.create_task(screening_queue.run_pipeline_task(job_id, _process_pipeline))

    return {
        "job_id": job_id,
        "status": JobStatus.QUEUED.value,
        "message": "Screening task queued successfully.",
        "websocket_url": f"/ws/jobs/{job_id}",
        "poll_url": f"/api/v1/jobs/{job_id}",
    }


@router.get("/api/v1/jobs/{job_id}")
async def get_screening_job_status(job_id: str):
    """Pollable endpoint to retrieve status and results of an async screening job."""
    job = screening_queue.get_job(job_id)
    if not job:
        raise HTTPException(404, detail=f"Job '{job_id}' not found.")
    return job


@router.websocket("/ws/jobs/{job_id}")
async def websocket_job_stream(websocket: WebSocket, job_id: str):
    """Real-time WebSocket event stream for an async screening job."""
    await screening_queue.notifier.connect(job_id, websocket)
    try:
        current_job = screening_queue.get_job(job_id)
        if current_job:
            await websocket.send_json({"type": "job_progress", **current_job})
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        await screening_queue.notifier.disconnect(job_id, websocket)
    except Exception:
        await screening_queue.notifier.disconnect(job_id, websocket)


@router.post("/scans/compress")
async def compress_uploaded_image(file: UploadFile = File(...)):
    """Lossless/near-lossless retinal image compression endpoint."""
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image file exceeds maximum limit of {MAX_FILE_SIZE // (1024 * 1024)} MB",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    safe_name = sanitize_filename(file.filename or "fundus.jpg")
    magic_ok, magic_result = validate_magic_bytes(contents, file.content_type)
    if not magic_ok:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=magic_result)

    dim_ok, dim_result = validate_image_dimensions(contents)
    if not dim_ok:
        raise HTTPException(status_code=422, detail=dim_result)

    raw_img = Image.open(io.BytesIO(contents))
    image = strip_image_metadata(raw_img)

    comp_bytes, orig_sz, comp_sz = compress_retinal_image(image)
    ratio = round((1.0 - (comp_sz / orig_sz)) * 100.0, 1)

    return {
        "original_size_bytes": orig_sz,
        "compressed_size_bytes": comp_sz,
        "bandwidth_saved_percent": ratio,
        "compressed_file_base64": base64.b64encode(comp_bytes).decode("utf-8"),
    }

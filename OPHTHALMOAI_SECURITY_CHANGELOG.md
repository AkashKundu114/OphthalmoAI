# OphthalmoAI: Security & Privacy Hardening Changelog

> **Release Version**: v1.2.0-secure  
> **Security Baseline**: Red-Team Validated & Automated Regression Tested  
> **Target Scope**: Backend API, Medical Image Ingestion, Model Inference, Persistence, Zero-Egress Network  

---

## 1. Summary of Changes

This release hardens OphthalmoAI for responsible external testing by unfamiliar clinicians, researchers, and penetration testers. The modifications eliminate malicious upload vectors, decompression bombs, path traversal in mock persistence stores, Insecure Direct Object References (IDOR) across patient scan records, unstripped EXIF/PHI metadata leaks, GPU memory exhaustion, and external network dependencies.

---

## 2. Granular Security Changelog by Component

### 2.1 Medical-Image Upload & Sanitization (`backend/security.py`)
- **Decompression Bomb Protection**: Reduced the default pixel ceiling `_MAX_PIXELS` from `89,478,485` to `25,000,000` (5000x5000 px). This caps uncompressed RAM usage while accommodating high-resolution 16 MP clinical fundus photography.
- **Minimum Clinical Dimension**: Enforced `MIN_IMAGE_DIMENSION_PX = 128` to reject sub-resolution icons, empty tiles, and corrupted small inputs.
- **Maximum Axis Boundary**: Enforced `MAX_IMAGE_AXIS_PX = 6000` to prevent single-axis dimension anomalies from causing GPU driver texture allocation timeouts.
- **Aspect Ratio Constraint**: Enforced `MAX_IMAGE_ASPECT_RATIO = 4.0` (rejecting extreme aspect ratios like 10:1 or 20:1 that do not correspond to circular posterior pole optics).
- **Filename Path Traversal Sanitization**: Implemented `sanitize_filename(raw_filename)` to strip directory paths (`../`, `..\`), strip ASCII control characters, and force safe extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
- **EXIF PHI Scrubbing**: Implemented `strip_image_metadata(img)` to re-allocate a fresh PIL `Image` buffer and copy only RGB pixels, purging all private EXIF tags, GPS coordinates, and camera serial numbers before in-memory processing.
- **Air-Gapped Zero-Egress CSP**: Fortified `SecurityHeadersMiddleware` with zero-egress Content-Security-Policy headers (`style-src 'self' 'unsafe-inline'; font-src 'self' data:;`), removing external Google Fonts CDN links.

### 2.2 Endpoint Hardening & Access Control (`backend/main.py`)
- **Concurrency Throttling**: Added `get_inference_semaphore()` (`MAX_CONCURRENT_INFERENCES = 4`, 20.0s timeout). When inference capacity is saturated, incoming requests receive `HTTP 503 Service Unavailable` with a clinical advisory rather than crashing the worker process with GPU OOM.
- **CUDA Cache Cleanup**: Added systematic `torch.cuda.empty_cache()` and `gc.collect()` in `finally:` blocks following all model executions in `/predict` and `/api/v1/screen/async`.
- **Numerical Instability Detection**: Added explicit validation checks for `NaN` and `Inf` values in output probability and logit tensors, returning structured error responses before triage processing.
- **Longitudinal History IDOR Defense**: Fortified `/scans/history/{user_id}` with regex identifier validation (`^[a-zA-Z0-9_\-]+$`) and role-based access control (patients can only view their own history; unauthorized access attempts return `HTTP 403 Forbidden`).
- **Scan Details IDOR Defense**: Fortified `/scans/{scan_id}/details` with format checks and ownership validation against `scan.user_id` and `scan.dicom_patient_id`.
- **Image Compression Hardening**: Fortified `/scans/compress` with `MAX_FILE_SIZE` enforcement (20 MB), magic bytes verification, dimension constraints, empty file checks, and EXIF metadata stripping.
- **Async Screening Hardening**: Fortified `/api/v1/screen/async` with magic bytes validation, filename sanitization, EXIF scrubbing, and CUDA cache reclamation.

### 2.3 Document Store Path Traversal Protection (`backend/mongodb_client.py`)
- **Mock Store Traversal Defense**: Replaced raw string concatenation in `_resolve_mock_path()` with strict regex validation (`^[a-zA-Z0-9_\-]+$`) and canonical directory confinement checks (`os.path.abspath(target).startswith(self.mock_dir)`). Arbitrary path traversal payloads (e.g., `../../etc/passwd`) raise `ValueError`.

### 2.4 Model Deserialization & Edge ONNX Export (`models/`, `scripts/`)
- **Safe PyTorch Deserialization**: Enforced `weights_only=True` on all `torch.load()` invocations across backend, evaluation scripts, and edge export scripts.
- **PyTorch-to-ONNX FP32 Parity**: Exported MobileNetV3-Small edge model (`models/edge_model_fp32.onnx`) and verified mathematical parity against PyTorch eager checkpoint ($\Delta_{max} = 3.81 \times 10^{-6}$, cosine similarity $0.99999998$).
- **Registry Checksum Tracking**: Updated `models/registry.json` with registered edge model architecture metadata and SHA256 integrity verification.

### 2.5 Automated Test Suite (`tests/backend/`)
- Created `tests/backend/test_security_red_team_and_privacy.py` containing 19 automated red-team security, privacy, decompression bomb, and IDOR unit tests.
- Created `tests/backend/test_onnx_parity_and_security.py` verifying numerical parity, safe deserialization, and air-gapped zero-egress fonts.
- Created `scripts/benchmark_resource_profiles.py` to capture genuine empirical resource profiles (RAM, VRAM, CPU utilization, latency percentiles).

# OphthalmoAI: Medical-Image Privacy & Clinical Dataflow Architecture

> **Document Status**: Release Security & Privacy Specification  
> **System Classification**: Investigational Decision-Support Prototype (Academic Manuscript in Preparation)  
> **Compliance Context**: HIPAA Security Rule Safeguards, GDPR Art. 9 (Health Data Protection), Zero-Egress Air-Gap Architecture  

---

## 1. Executive Summary & Privacy Threat Model

OphthalmoAI processes sensitive ophthalmic data, including high-resolution digital fundus photography, optical coherence tomography scans, patient demographic indicators, and ocular biomarkers. In clinical screening workflows, medical images frequently carry Protected Health Information (PHI) embedded in EXIF dictionaries, DICOM headers, or filename strings.

This document traces the complete data lifecycle across ingestion, memory transformation, inference, persistent storage, and egress boundaries.

```
[ Client / Clinician Browser ]
            |  (HTTPS / TLS 1.3 - Form Multipart)
            v
[ SecurityHeadersMiddleware & RequestIDMiddleware ]
    - Injects Strict-Transport-Security, X-Content-Type-Options
    - Air-Gapped Zero-Egress CSP (no external font or script CDNs)
    - X-Request-ID propagation for audit logging
            |
            v
[ Upload Sanitization & Ingestion Boundary ]
    - Magic Bytes Verification: Discards spoofed extensions (validate_magic_bytes)
    - Decompression Bomb Defense: Bounds decoded pixels <= 25,000,000 px (_MAX_PIXELS)
    - Aspect Ratio Guard: Rejects pathological aspect ratios > 4.0:1
    - Filename Sanitizer: Strips path traversal sequences (sanitize_filename)
            |
            v
[ In-Memory Privacy & EXIF Scrubbing ]
    - strip_image_metadata(): Reallocates clean RGB buffer, purging EXIF, GPS, camera serials
    - Anonymise IP: Hashes/truncates client IP before structured audit log entry
            |
            v
[ Inference Pipeline & Ephemeral Execution ]
    - Concurrency Semaphore: Bounds active inferences (MAX_CONCURRENT_INFERENCES = 4)
    - Ben Graham CLAHE Preprocessing & Tensor Normalization
    - Tri-Backbone Soft-Voting Ensemble / ONNX Runtime
    - CUDA Cache Eviction: torch.cuda.empty_cache() in finally blocks
            |
            +--------------------------------+
            |                                |
            v                                v
[ Ephemeral / Transient Response ]   [ Encrypted Persistence (Optional) ]
    - Diagnosis, Confidence, Set        - PostgreSQL / SQLite with Tenant Isolation
    - Calibrated Probabilities          - MongoStore (_resolve_mock_path path traversal defense)
    - Grad-CAM Heatmap Base64           - Role-Based Access Control (RBAC) & IDOR Defense
```

---

## 2. Ingestion Boundary & PHI Stripping

### 2.1 File Ingestion Constraints
Every endpoint receiving binary image data (`/predict`, `/scans/compress`, `/api/v1/screen/async`) enforces strict input constraints before any image decoding occurs:

| Property | Clinical Constraint | Security Rationale | Enforcement Mechanism |
| :--- | :--- | :--- | :--- |
| **Max Payload Size** | 20 MB (`MAX_FILE_SIZE`) | Mitigates Denial-of-Service memory exhaustion | `backend.main.MAX_FILE_SIZE` |
| **Magic Byte Check** | JPEG, PNG, BMP, WEBP signatures | Prevents executable upload masquerading as image | `backend.security.validate_magic_bytes` |
| **Pixel Dimension Cap** | 25,000,000 pixels (5000x5000 px) | Defends against decompression bomb (zip/image bomb) | `backend.security._MAX_PIXELS` |
| **Min Dimension** | 128x128 pixels | Rejects sub-resolution icons or empty tiles | `backend.security.MIN_IMAGE_DIMENSION_PX` |
| **Max Axis Limit** | 6,000 pixels | Bounds single-axis aspect anomalies | `backend.security.MAX_IMAGE_AXIS_PX` |
| **Aspect Ratio Ratio**| Max 4.0:1 | Rejects pathological strips (e.g., 10000x1 px) | `backend.security.MAX_IMAGE_ASPECT_RATIO` |

### 2.2 EXIF & Metadata Scrubbing (`strip_image_metadata`)
Fundus cameras (e.g., Topcon, Zeiss, Optos) and smartphones operating with fundus adapters frequently encode device serial numbers, patient IDs, and geolocation coordinates into image headers. 

`backend/security.py` implements surgical metadata purging:
```python
def strip_image_metadata(img: Any) -> Any:
    """Strips EXIF, GPS, camera serials, and custom metadata tags, returning clean RGB image."""
    from PIL import Image
    if img is None:
        return None
    rgb = img.convert("RGB")
    clean_img = Image.new("RGB", rgb.size)
    clean_img.paste(rgb)
    return clean_img
```
By re-allocating a new `Image` object with freshly allocated raster pixels and pasting only RGB channels, all internal dictionary structures (`exif`, `info`, `gps`, `maker_note`) are unlinked and destroyed in memory.

---

## 3. IDOR & Access Isolation Architecture

### 3.1 Patient Isolation
Patient health data is isolated via two distinct mechanisms:
1. **Longitudinal History Endpoint (`/scans/history/{user_id}`)**:
   - Enforces regex validation `^[a-zA-Z0-9_\-]+$`.
   - Compares the authenticated `current_user.id` against requested `user_id`.
   - If `current_user.role == "patient"` and IDs do not match, immediately returns `HTTP 403 Forbidden`.
   - Clinicians, ophthalmologists, and administrators are authorized to review patient records.

2. **Scan Result Details (`/scans/{scan_id}/details`)**:
   - Enforces format validation on `scan_id`.
   - Checks ownership against `scan.user_id` and `scan.dicom_patient_id`.
   - Prevents patients from enumerating scan IDs across the clinical database.

### 3.2 Mock Document Store Traversal Hardening
When executing in mock or local development environments without an active MongoDB replica set, `mongo_store` stores documents in a dedicated directory. Path traversal attempts (e.g., `../../etc/passwd`) are blocked via canonical resolution:
```python
def _resolve_mock_path(self, doc_id: str) -> str:
    if not doc_id or not _VALID_DOC_ID_RE.match(doc_id):
        raise ValueError(f"Invalid document identifier format: {doc_id}")
    resolved = os.path.abspath(os.path.join(self.mock_dir, f"{doc_id}.json"))
    if not resolved.startswith(self.mock_dir + os.sep) and resolved != self.mock_dir:
        raise ValueError(f"Directory traversal detected: {doc_id}")
    return resolved
```

---

## 4. Air-Gapped Zero-Egress Network Architecture

### 4.1 Strict Content-Security-Policy (CSP)
For offline clinical deployments in remote rural clinics or air-gapped hospital intranets, the application is prohibited from making outbound requests to third-party CDNs or telemetry endpoints.

`backend/security.py` enforces the following zero-egress CSP:
```http
Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self' data:; img-src 'self' data: blob:; connect-src 'self' ws: wss:; frame-ancestors 'none'; object-src 'none'; base-uri 'self';
```
- **External CDN Fonts Removed**: All fonts (`Inter`, `Cinzel`, system fallbacks) are served locally or as system fonts. No requests to `fonts.googleapis.com` or `fonts.gstatic.com`.
- **Telemetry Disallowed**: No third-party tracking scripts, analytics beacons, or remote tag managers exist in either frontend or backend.

---

## 5. Ephemeral Compute & Memory Hygiene

### 5.1 Concurrency Throttling
To protect GPU and CPU host memory from resource exhaustion attacks, `/predict` executes through an async semaphore (`MAX_CONCURRENT_INFERENCES = 4`, 20.0s acquisition timeout). When saturated, incoming requests receive `HTTP 503 Service Unavailable` with a clinical advisory header rather than crashing the worker process.

### 5.2 GPU Cache & Garbage Collection
Following every inference pass, the worker executes memory reclamation:
```python
finally:
    try:
        sem.release()
    except (ValueError, RuntimeError):
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
```
This guarantees that patient feature maps and activations in VRAM cannot bleed into subsequent screening sessions.

---

## 6. Audit Logging & IP Anonymization

All sensitive screening interactions are recorded in the structured audit log with sanitized attributes:
- **Client IP**: Truncated to `/24` for IPv4 (`192.168.1.0`) and `/48` for IPv6 using `anonymise_ip()`.
- **Sensitive Inputs**: Passwords, raw base64 images, and medical history texts are excluded from log messages.
- **Trace Correlation**: Correlated using UUIDv4 `X-Request-ID` across middleware, backend spans, and DB audit tables.

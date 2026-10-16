# OphthalmoAI: Current-State Security Audit & Threat Model

**Audit Date**: 2026-10-17  
**Scope**: FastAPI Screening Backend, Image Ingestion Pipeline, React 19 Frontend, Model Storage & Serving  
**Current Posture**: Production Hardened for Research Prototype Deployment  

---

## 1. Threat Model & Asset Inventory

| Asset | Sensitivity | Primary Threat Vectors | Applied Mitigations |
| :--- | :--- | :--- | :--- |
| **Patient Retinal Scans** | High (Biometric / Health) | Unauthorized storage, cloud egress, EXIF/GPS leaks | `PERSIST_SCANS=false` (ephemeral RAM buffers), automated EXIF stripping, zero egress edge mode |
| **Model Weights (.pth, .onnx)** | High (IP / Integrity) | Insecure deserialization, weight tampering | SHA256 checksum validation, `weights_only=True` in `torch.load()`, read-only container binds |
| **API Endpoints (/predict, /chat)** | Medium | Denial-of-Service, decompression bombs, prompt injection | SlowAPI rate limiting, Pillow max pixels ($89,478,485$), sanitization, medical emergency interceptors |
| **User & Clinician Accounts** | High (RBAC) | Brute-force credential stuffing, token forgery | Bcrypt password hashing, JWT HS256 with 256-bit secrets, role-based decorators (`require_role`) |

---

## 2. Ingestion Pipeline & Image Upload Hardening

### 2.1 Magic Byte & MIME Validation
- Files uploaded to `POST /predict` are inspected using magic byte headers rather than trust in user-supplied file extensions:
  - JPEG: `\xFF\xD8\xFF`
  - PNG: `\x89PNG\r\n\x1a\n`
  - WebP: `RIFF....WEBP`
- Reject executables, SVGs, HTML/XML files disguised with image extensions.

### 2.2 Decompression Bomb Safeguards
- Pillow `Image.MAX_IMAGE_PIXELS` set strictly to $89,478,485$ pixels.
- Axis dimension limit enforced: any image with width or height exceeding $10,000$ pixels is rejected before rasterization.
- File upload payload size capped at 20 MB via reverse proxy and middleware.

### 2.3 EXIF & Metadata Sanitization
- Prior to memory array conversion or downstream inference, `strip_image_metadata()` extracts raw pixel buffers into a fresh RGB PIL canvas, stripping:
  - GPS latitude, longitude, and elevation tags.
  - Camera make, model, and serial number.
  - Patient name and DICOM tags.

---

## 3. Ephemeral In-Memory Processing Posture

- By default, `PERSIST_SCANS=false` in `backend/config.py`.
- Uploaded scans are evaluated entirely within volatile RAM memory buffers and immediately released after response generation.
- No patient imagery is written to persistent block storage or external cloud buckets unless the deployer explicitly configures S3/GCS credentials.

---

## 4. Authentication, Authorization & RBAC

- **Password Hashing**: Bcrypt with auto-salted key derivation.
- **Token Management**: JWT tokens signed using HMAC-SHA256 with random tokens generated via `secrets.token_hex(32)`.
- **Role Isolation**: Strict separation of roles (`patient`, `clinician`, `admin`). Administrative overrides and batch review queues enforce `require_role(["clinician", "admin"])`.
- **Audit Logging**: Immutable event log tracking actions, timestamps, and anonymized client IPs.

---

## 5. Dependency & Supply Chain Verification

- **Python**: `pip-audit -r backend/requirements.txt` executed on Python 3.12: **0 known vulnerabilities found**. (Resolved CVE-2024-33663 by removing deprecated `python-jose` in favor of `pyjwt`).
- **Node.js**: `npm audit --audit-level=high` executed on frontend dependencies: **0 high/critical vulnerabilities found**.

---

## 6. Open / Deferred Security Items

1. **Database Row-Level Security (RLS)**: Currently enforced at the SQLAlchemy ORM layer (`backend/tenancy.py`). Database-native PostgreSQL `CREATE POLICY` is deferred to `v0.2.0` (see `REMEDIATION/DEFERRED.md`).
2. **KMS Hardware Key Integration**: Local signing uses environment secrets; hardware-backed HSM integration is deferred to enterprise cloud deployments.

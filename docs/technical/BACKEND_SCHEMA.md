# Backend Schema & API Reference: OphthalmoAI

**Base URL (local dev)**: `http://localhost:8000`  
**Base URL (Docker prod)**: `http://localhost:8000` (or via reverse proxy)  

---

## 1. System Architecture & Startup Sequence

The backend is built with **FastAPI** (Python 3.10+) running an asynchronous ASGI server with PyTorch GPU acceleration.

### Startup Lifecycle (`lifespan` handler)
1. **Device Detection**: Identifies CUDA GPU acceleration (e.g. NVIDIA RTX 5060 Laptop GPU, 8GB VRAM) or falls back to multi-core CPU.
2. **Model Loading (Tri-Backbone + Grad-CAM)**:
   - `models/densenet201_best.pth` -> DenseNet-201 (eval mode)
   - `models/convnext_best.pth` -> ConvNeXt-Small (eval mode)
   - `models/efficientnet_v2_best.pth` -> EfficientNet-V2-M (eval mode)
   - `models/efficientnet_b4_best.pth` -> EfficientNet-B4 (eval mode, used for Grad-CAM saliency heatmaps)
3. **Calibration Loading**: Reads Platt temperature scaling parameters from `models/calibration.json`:
   - `efficientnet_v2_m`: $T = 1.0654$
   - `densenet201`: $T = 1.2616$
   - `convnext_small`: $T = 1.3407$
   - `efficientnet_b4`: $T = 1.3275$
   - `resnet50`: $T = 1.0947$
4. **Database Initialisation**: Asynchronous SQLAlchemy connection pooling (`asyncpg` / `aiosqlite`) with Alembic version tracking.
5. **Shutdown Cleanup**: Explicit `gc.collect()` and `torch.cuda.empty_cache()`.

---

## 2. API Endpoints

### 2.1 Core Inference: `POST /predict`
Runs multi-backbone inference, temperature-calibrated soft voting, and Grad-CAM generation.

- **Request**: `multipart/form-data`
  - `file`: Image file (JPEG/PNG, validated via magic bytes)
  - `symptoms`: Optional JSON-encoded clinical symptoms dictionary
- **Inference Pipeline**:
  1. Image validation (magic bytes, dimension bounds $\le 10,000 \times 10,000$) & resizing to $384 \times 384$ with Ben Graham circular illumination subtraction.
  2. Image Quality Assessment (IQA) checking sharpness and exposure.
  3. **Optical Aperture & Chromophore Domain Guardrail (OAC-DG)**: Validates optical aperture circularity, chorioretinal red-to-blue backscatter ($\bar{R}/\bar{B} \ge 1.05$), and spatial autocorrelation. Non-fundus photographs (everyday scenes, pets, noise, documents) are rejected immediately with HTTP 422.
  4. Parallel forward pass across DenseNet-201, ConvNeXt-Small, and EfficientNet-V2-M.
  5. Platt temperature scaling per model: $z_m / T_m^*$.
  6. Soft-voting probability averaging across all active backbones.
  7. Dedicated EfficientNet-B4 Grad-CAM activation map extraction and viridis colormap generation.
  8. Clinical metadata mapping (ICD-10, SNOMED-CT, urgency level).
- **Response (`200 OK`)**: Returns prediction with condition, calibrated confidence, probabilities for all 6 retinal classes, urgency, and base64-encoded Grad-CAM heatmap.
- **Error Response (`422 Unprocessable Content`)**: Rejection for non-fundus photographs, blurry images, or invalid formats.

### 2.2 AI Clinical Assistant: `POST /chat`
- **Request**: `{"message": "...", "scan_context": {...}}`
- **Engine**: Google Gemini 2.0 Flash (free tier) with fallback to local Ollama.
- **Guardrails**:
  - Medical emergency interceptor (e.g. chemical splash, sudden blindness) routing to emergency instructions immediately.
  - Sanitization against prompt injections ("ignore previous instructions"), jailbreaks ("DAN"), and developer persona overrides.
  - Rejection of prescription requests and off-topic queries (code generation, essays).

### 2.3 System & Diagnostic Metadata
- `GET /health`: Liveness probe.
- `GET /ready`: Readiness probe (confirms backbones and database connectivity).
- `GET /conditions`: Authoritative list of 6 retinal conditions with descriptions and ICD-10 codes.
- `GET /metrics/system`: Runtime hardware telemetry (GPU VRAM, temperatures, throughput).

### 2.4 Clinician Review & Audit Trail
- `POST /auth/token`: JWT authentication with role-based access control (`patient`, `clinician`, `admin`).
- `POST /scans/{id}/override`: Clinician secondary opinion and diagnostic correction.
- `GET /admin/audit-logs`: Immutable administrative audit trail.

# OphthalmoAI: Security, Medical Privacy & Resource Optimization Release Report

> **Document Status**: Official Release Security & Optimization Audit  
> **Release Target**: OphthalmoAI Clinical Screening Pipeline (PyTorch CUDA / ONNX / FastAPI / React 19)  
> **Publication & Regulatory Status**: Academic Research Manuscript in Preparation (Investigational Decision-Support Prototype). Not published in IEEE; not cleared for standalone diagnostic use.  
> **Automated Verification**: **306 Passed**, 0 Failed across full regression suite.  

---

## 1. Executive Summary

This comprehensive audit and release report summarizes the hardening, medical-image privacy protections, adversarial testing, and resource optimization performed on the OphthalmoAI platform to prepare it for responsible external testing by unfamiliar clinicians, researchers, and technical auditors.

The hardening cycle achieved:
1. **Zero-Trust Image Upload Pipeline**: Strict multi-stage validation terminating decompression bombs (>25 MP), spoofed file extensions, MIME mismatches, pathological aspect ratios (>4.0:1), and sub-minimum clinical dimensions (<128 px).
2. **Medical Privacy & Zero-Egress Air-Gapping**: Surgical scrubbing of EXIF, GPS, and camera metadata (`strip_image_metadata`), IP address anonymization (`anonymise_ip`), and strict zero-egress Content-Security-Policy headers with complete removal of external CDN font calls.
3. **Inference Stability & Resource Throttling**: Asynchronous concurrency semaphore (`MAX_CONCURRENT_INFERENCES = 4`, 20s timeout) preventing GPU/CPU memory exhaustion, `NaN`/`Inf` tensor detection, systematic `torch.cuda.empty_cache()` cache reclamation, and mathematically bounded PyTorch-to-ONNX FP32 parity ($\Delta_{max} = 3.81 \times 10^{-6}$, Cosine Similarity $0.99999998$).
4. **Access Isolation & IDOR Defenses**: Elimination of Insecure Direct Object References on longitudinal history (`/scans/history/{user_id}`) and scan details (`/scans/{scan_id}/details`), plus strict directory confinement on mock document persistence (`mongo_store`).
5. **Empirical Benchmarking**: Captured real latency, RAM/VRAM, and throughput profiles across both GPU (NVIDIA RTX 5060 Laptop GPU) and CPU runtimes.

---

## 2. Security & Privacy Threat Model

```
   [ UNTRUSTED EXTERNAL ENVIRONMENT ]
                   |
   (Form Uploads, Malicious Scans, Attack Scripts)
                   |
                   v
+-------------------------------------------------------------+
| 1. PERIMETER & SECURITY HEADERS                             |
|    - SecurityHeadersMiddleware (HSTS, NoSniff, FrameGuard)  |
|    - Zero-Egress Air-Gapped CSP (No external CDNs/telemetry)|
|    - RequestIDMiddleware (UUIDv4 distributed tracing)       |
+-------------------------------------------------------------+
                   |
                   v
+-------------------------------------------------------------+
| 2. INGESTION RED-TEAM DEFENSES                              |
|    - Max File Size: 20 MB ceiling                           |
|    - validate_magic_bytes(): Header signature inspection     |
|    - validate_image_dimensions(): 25 MP & 4:1 aspect cap    |
|    - sanitize_filename(): Path traversal neutralization     |
|    - strip_image_metadata(): EXIF & PHI scrubbing           |
+-------------------------------------------------------------+
                   |
                   v
+-------------------------------------------------------------+
| 3. DOMAIN VERIFICATION & IQA                                |
|    - validate_fundus_image(): Circular FOV & optical checks |
|    - assess_image_quality(): Blur, contrast, illumination   |
+-------------------------------------------------------------+
                   |
                   v
+-------------------------------------------------------------+
| 4. INFERENCE ENGINE & RESOURCE ISOLATION                    |
|    - Concurrency Semaphore: Max 4 active leases             |
|    - Ben Graham CLAHE Preprocessing                         |
|    - Tri-Backbone Soft-Voting Ensemble / ONNX Runtime       |
|    - Numerical Check: NaN/Inf logit detection               |
|    - Cache Reclamation: torch.cuda.empty_cache() & gc       |
+-------------------------------------------------------------+
                   |
                   v
+-------------------------------------------------------------+
| 5. PERSISTENCE, TENANCY & RBAC                              |
|    - IDOR Defense: Strict patient-level scan isolation      |
|    - Row-Level Tenant Isolation                             |
|    - MongoStore: _resolve_mock_path directory confinement   |
|    - Safe Deserialization: torch.load(weights_only=True)    |
+-------------------------------------------------------------+
```

---

## 3. Key Findings & Remediation Ledger

| Finding ID | Severity | Subsystem | Vulnerability Description | Remediation Implemented | Verification Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SEC-01** | **High** | Upload Ingestion | Decompression bomb vulnerability: default 89.4 MP pixel ceiling permitted uncompressed ~270 MB image buffers in RAM, creating an OOM vector. | Capped `_MAX_PIXELS` at `25,000,000` (5000x5000 px). Added axis limit (6000 px) and aspect ratio limit (4.0:1). | `test_decompression_bomb_pixel_limit`, `test_max_axis_dimension_rejection` |
| **SEC-02** | **High** | Access Control | Insecure Direct Object Reference (IDOR) on `/scans/history/{user_id}` and `/scans/{scan_id}/details` permitted any caller to query other patients' scan records. | Enforced regex validation and role-based checks ensuring patients can only inspect their own records. | `test_idor_patient_cannot_access_other_patient_history`, `test_idor_patient_cannot_access_other_patient_scan_details` |
| **SEC-03** | **Medium** | Patient Privacy | EXIF headers (e.g. camera model, serial numbers, patient MRN in ImageDescription) were passed unstripped to in-memory buffers. | Created `strip_image_metadata()` to re-allocate clean RGB raster buffers and purge EXIF dictionaries. | `test_exif_metadata_stripping` |
| **SEC-04** | **Medium** | Mock Persistence | `mongo_store._resolve_mock_path()` used unvalidated string concatenation, allowing path traversal (`../../etc/passwd`). | Implemented strict regex `^[a-zA-Z0-9_\-]+$` and verified `os.path.abspath(target)` is strictly confined within `self.mock_dir`. | `test_mongodb_mock_store_path_traversal_defense` |
| **SEC-05** | **Medium** | Air-Gap Network | CSP headers referenced external Google Fonts (`fonts.googleapis.com`), violating zero-egress air-gap clinical compliance. | Removed external fonts from CSP; localized all font stacks to local system fonts. | `test_air_gapped_zero_egress_font_integrity`, `test_zero_egress_csp_header_enforced` |
| **SEC-06** | **Medium** | Compression Endpoint | `/scans/compress` lacked file size checks, magic byte verification, dimension bounds, and metadata stripping. | Applied full sanitization pipeline (`MAX_FILE_SIZE`, `validate_magic_bytes`, `validate_image_dimensions`, `strip_image_metadata`). | `test_compress_endpoint_decompression_bomb_rejected`, `test_compress_endpoint_empty_file_rejected` |
| **SEC-07** | **High** | Inference Stability | Unbounded concurrent requests could exhaust GPU VRAM, triggering unhandled CUDA OOM process termination. | Implemented `get_inference_semaphore()` (`MAX_CONCURRENT_INFERENCES = 4`, 20s timeout) and systematic CUDA cache reclamation. | `test_predict` & concurrency benchmarks |
| **SEC-08** | **High** | Model Deserialization | Potential arbitrary code execution via Python pickle unpickling during checkpoint loading. | Enforced `torch.load(..., weights_only=True)` on all model checkpoint loads across all backend and script workflows. | `test_weights_only_deserialization_security` |

---

## 4. Empirical Resource & Latency Profiles

*Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 12.8) / 32 Logical Cores / 16 GB RAM*

### 4.1 Inference Latency & Memory Footprint

| Architecture | Deployment Target | Checkpoint Size | Cold Load | p50 Latency | Throughput | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | Edge (GPU) | 5.95 MB | 253.1 ms | **4.30 ms** | **230.4 QPS** | 48.1 MB |
| **MobileNetV3-Small** | Edge (CPU) | 5.95 MB | 68.0 ms | **6.41 ms** | **152.8 QPS** | 0.0 MB |
| **EfficientNet-B4** | Monolith (GPU) | 67.71 MB | 489.0 ms | **16.80 ms** | **58.7 QPS** | 213.3 MB |
| **EfficientNet-B4** | Monolith (CPU) | 67.71 MB | 330.7 ms | **66.25 ms** | **14.6 QPS** | 0.0 MB |
| **Tri-Backbone Ensemble** | Clinical Ensemble (GPU)| 462.35 MB | 2.47 s | **67.00 ms** | **14.9 QPS** | ~1.12 GB |

### 4.2 Image Preprocessing Timings
- **Standard PIL Resize**: 10.76 ms mean (11.60 ms p95)
- **Ben Graham CLAHE Preprocessing**: 69.49 ms mean (74.65 ms p95)

---

## 5. Verification & Test Evidence

All remediations are verified by automated regression test suites:
- `tests/backend/test_security_red_team_and_privacy.py` (19 passed)
- `tests/backend/test_onnx_parity_and_security.py` (3 passed)
- Full Test Suite `tests/`: **306 passed, 0 failed** in ~80 seconds.

---

## 6. Academic Research & Regulatory Status

> [!IMPORTANT]
> - OphthalmoAI is an **investigational decision-support prototype** developed for biomedical informatics research.
> - The research paper is currently an **academic manuscript in preparation** and is **not published in IEEE**.
> - The system does not claim FDA 510(k), CE mark, or international regulatory clearance for autonomous clinical diagnosis.
> - All algorithmic predictions must be reviewed by a licensed ophthalmologist or optometrist before clinical intervention.

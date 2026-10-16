# OphthalmoAI: Adversarial Upload & Red-Team Security Assessment

> **Classification**: Release Red-Team & Adversarial Ingestion Report  
> **Target Application**: OphthalmoAI Clinical Screening Pipeline (FastAPI / PyTorch / ONNX)  
> **Status**: Verified Against Test Suite (19/19 Red-Team Tests Passing)  
> **Scope**: Malicious file payloads, decompression bombs, path traversal, EXIF PHI leakage, and domain spoofing  

---

## 1. Executive Summary

This report documents the red-team penetration tests executed against the OphthalmoAI image ingestion endpoints (`/predict`, `/scans/compress`, `/api/v1/screen/async`). In medical AI systems, image ingestion constitutes the primary untrusted boundary. Malicious or malformed inputs can cause Denial of Service (DoS) through unbounded decompression, execute remote commands via file-extension spoofing, leak patient privacy through unstripped EXIF tags, or corrupt model inference through non-ophthalmic image injection.

All 7 core adversarial vectors have been empirically tested, fortified, and validated with automated regression tests in `tests/backend/test_security_red_team_and_privacy.py`.

---

## 2. Attack Vectors & Verification Matrix

| Vector ID | Attack Payload Description | Vulnerability Mechanism | Remediation Applied | Automated Test Status |
| :--- | :--- | :--- | :--- | :--- |
| **ADV-01** | **Magic Byte Spoofing**: PHP script disguised as `fundus.jpg` (`<?php echo '...'; ?>`) | Client-supplied MIME type accepted without verifying file signature | Multi-byte file signature validation (`validate_magic_bytes`) checking JPEG, PNG, BMP, WEBP, GIF headers | **PASSED** (`test_magic_bytes_spoofed_extension`) |
| **ADV-02** | **MIME / Magic Byte Mismatch**: Valid PNG binary declared as `image/jpeg` | Bypasses format-specific decoders or triggers undefined decoder behavior | Strict cross-check between detected magic bytes and declared `Content-Type` header | **PASSED** (`test_magic_bytes_mime_mismatch`) |
| **ADV-03** | **Decompression Bomb**: 5500 x 5000 px image (27.5 MP > 25 MP limit) | Allocates ~82.5 MB raw uncompressed pixel buffer in RAM, creating OOM crash risk | Header-only size extraction with strict `_MAX_PIXELS = 25_000_000` cap | **PASSED** (`test_decompression_bomb_pixel_limit`) |
| **ADV-04** | **Sub-Resolution Icon Injection**: 64 x 64 px image | Insufficient resolution for optic disc or macula evaluation; degrades model predictions | Enforced `MIN_IMAGE_DIMENSION_PX = 128` rejecting sub-clinical resolutions | **PASSED** (`test_subminimum_dimension_rejection`) |
| **ADV-05** | **Pathological Aspect Ratio**: 2000 x 200 px strip (10:1 aspect ratio) | Non-physiological retinal scan format; distorts convolutional receptive fields | Enforced `MAX_IMAGE_ASPECT_RATIO = 4.0` requiring circular or standard 4:3 optical profiles | **PASSED** (`test_pathological_aspect_ratio_rejection`) |
| **ADV-06** | **Max-Axis Dimension Attack**: 6500 x 5000 px image | Exceeds GPU texture allocation buffers and causes driver timeouts | Enforced `MAX_IMAGE_AXIS_PX = 6000` capping maximum single-axis dimension | **PASSED** (`test_max_axis_dimension_rejection`) |
| **ADV-07** | **Filename Path Traversal**: `../../etc/passwd` or `..\..\cmd.exe` | Could traverse storage roots or overwrite system binaries when writing to disk | `sanitize_filename()` strips directory paths, non-ASCII control characters, and enforces whitelisted extension | **PASSED** (`test_filename_sanitization_path_traversal`) |
| **ADV-08** | **EXIF PHI & GPS Extraction**: Image carrying MRN in `ImageDescription` & GPS tags | Metadata retains sensitive patient identifiers across clinical networks | `strip_image_metadata()` re-allocates clean RGB buffer, completely purging EXIF dictionaries | **PASSED** (`test_exif_metadata_stripping`) |
| **ADV-09** | **Mock DB Traversal**: Calling `mongo_store.get_document("../../etc/passwd")` | Arbitrary file read in mock development environments without live MongoDB | `_resolve_mock_path()` validates regex `^[a-zA-Z0-9_\-]+$` and verifies canonical directory confinement | **PASSED** (`test_mongodb_mock_store_path_traversal_defense`) |
| **ADV-10** | **Empty Payload DoS**: 0-byte file upload to `/scans/compress` | Uncaught EOF / decode exception crashes endpoint | Explicit payload length check (`len(contents) == 0`) raising `HTTP 400 Bad Request` | **PASSED** (`test_compress_endpoint_empty_file_rejected`) |

---

## 3. Detailed Technical Remediation

### 3.1 Two-Stage Image Validation Flow
```
[ Incoming Raw Bytes ]
         |
         v
[ Stage 1: Fast Byte Inspection (< 0.05 ms) ]
  - Byte length check: len(data) >= 12 and len(data) <= MAX_FILE_SIZE (20 MB)
  - Signature check: validate_magic_bytes(data, declared_mime)
  - Reject immediately if signature fails (No PIL decoding)
         |
         v
[ Stage 2: Header-Only Dimension Inspection (< 0.2 ms) ]
  - Image.MAX_IMAGE_PIXELS = 25_000_000
  - with Image.open(io.BytesIO(data)) as img:
      w, h = img.size
      check: w >= 128 and h >= 128
      check: max(w, h) <= 6000
      check: max(w/h, h/w) <= 4.0
      check: w * h <= 25_000_000
         |
         v
[ Stage 3: Privacy & Sanitization (< 2.0 ms) ]
  - strip_image_metadata(img): Purges EXIF, GPS, camera metadata
  - sanitize_filename(filename): Normalizes safe identifier
```

### 3.2 Non-Fundus Domain Guardrail
To prevent attackers from submitting non-retinal images (e.g. photos of people, animals, vehicles, skin lesions) to induce confident but invalid classifications:
- `backend/fundus_validator.py` evaluates:
  1. Circular field-of-view (FOV) aperture mask.
  2. Red-channel dominance characteristic of vascularized choroid/retina ($R_{mean} > G_{mean} > B_{mean}$).
  3. High-frequency edge density in macula/disc vascular arcades.
- Images failing fundus validation are rejected with `HTTP 422 Unprocessable Entity` and a clinical guidance message.

---

## 4. Empirical Test Verification

All security mechanisms are tested continuously in the automated test suite:
```powershell
& ".venv_gpu\Scripts\python.exe" -m pytest tests/backend/test_security_red_team_and_privacy.py -v
```
**Outcome**: 19 passed, 0 failed, 0 regressions.

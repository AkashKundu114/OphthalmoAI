# Security Audit & Vulnerability Assessment: Project B — OphthalmoAI

**Target Repository:** `d:\OphthalmoAI`  
**Date:** October 16, 2026  
**Auditor:** Principal Medical AI Security Engineer  
**Standard:** HIPAA Security Rule, OWASP API Security Top 10, CWE / SANS Top 25

---

## 1. Executive Summary

OphthalmoAI handles sensitive biometric health information (retinal fundus imagery, diagnostic predictions, patient clinical notes). The security audit evaluated API boundaries, authentication models, upload processing, third-party egress, and machine learning model deserialization.

Two significant issues were identified and resolved, and operational security boundaries have been formally documented.

---

## 2. Vulnerability Ledger & Remediation Analysis

### 2.1 [OPH-SEC-01] [HIGH] Silent Clinical Data Egress via Unconfigured Cloudflare Tunnel Fallback

- **CWE:** CWE-200 (Exposure of Sensitive Information to an Unauthorized Actor)
- **Component:** `frontend/src/apiConfig.js` (`FALLBACK_TUNNEL_URL`)
- **Severity:** **High (CVSS 7.5 / HIPAA Violation Risk)**
- **Vulnerability Description:**  
  `apiConfig.js` hardcoded `export const FALLBACK_TUNNEL_URL = 'https://api.trycloudflare.com'`. If the local backend or primary API was unavailable or slow to respond, client-side fetch logic silently fell back to transmitting user `FormData`—containing unencrypted retinal fundus images and clinical metadata—to an external Cloudflare tunnel endpoint.
- **Exploitation Impact:**  
  Biometric retinal imagery could be transmitted across public third-party infrastructure without explicit clinician or patient consent, violating HIPAA biometric privacy and zero-egress commitments.
- **Remediation Implemented:**  
  Updated `apiConfig.js`: `FALLBACK_TUNNEL_URL = import.meta.env.VITE_FALLBACK_TUNNEL_URL || ''`. By default, no fallback tunnel URL is configured, and all requests remain strictly bound to local `/api` routes.

---

### 2.2 [OPH-SEC-02] [MEDIUM] Account Contamination on Unauthenticated Appointment Scheduling & History

- **CWE:** CWE-284 (Improper Access Control), CWE-200 (Information Disclosure)
- **Component:** `backend/main.py` (`schedule_appointment`, `get_appointments_history`)
- **Severity:** **Medium (CVSS 6.5)**
- **Vulnerability Description:**  
  In `POST /appointments` and `GET /appointments/history`, unauthenticated requests fell back to `current_user = db.query(User).first()`. If the first user record in the database was the clinic administrator or a real patient, unauthenticated users could schedule appointments on their behalf or inspect their full appointment history.
- **Remediation Implemented:**  
  Restricted fallback strictly to querying or creating a designated isolated demo user (`User.email == "demo@ophthalmoai.org"`) with hashed bcrypt credentials:
  ```python
  if not current_user:
      demo_user = db.query(User).filter(User.email == "demo@ophthalmoai.org").first()
      if not demo_user:
          demo_user = User(email="demo@ophthalmoai.org", hashed_password=hash_password("demo_password"), role="patient")
          db.add(demo_user)
          db.commit()
          db.refresh(demo_user)
      current_user = demo_user
  ```
- **Regression Verification:**  
  Added and verified `test_appointment_unauthenticated_isolation` in `tests/backend/test_security_and_auth.py`.

---

### 2.3 [OPH-SEC-03] [LOW] Insecure Model Deserialization in Auxiliary Scripts

- **CWE:** CWE-502 (Deserialization of Untrusted Data)
- **Component:** `scripts/evaluate_ensemble.py`, `scripts/train_ensemble.py`, `scripts/calibrate_models.py`, `scripts/evaluate_models.py`, `scripts/train_model.py`, `scripts/finetune_retfound_comparison.py`, `scripts/fine_tune_external_ensemble.py`, `scripts/run_conformal_calib.py`, `scripts/run_statistical_tests.py`
- **Severity:** **Low (Internal Script Context)**
- **Vulnerability Description:**  
  While runtime inference in `backend/main.py` enforced `weights_only=True`, multiple offline training scripts in `scripts/` invoked `torch.load()` without setting `weights_only=True` or explicitly set `weights_only=False`.
- **Remediation Implemented:**  
  Audited all 25 checkpoints in `models/*.pth` (all verified to be pure tensor state dicts). Added `weights_only=True` to every `torch.load` invocation across the entire repository (0 bare or `weights_only=False` calls remain).
- **Regression Verification:**  
  Verified via `test_weights_only_deserialization_security` in `tests/backend/test_onnx_parity_and_security.py`.
- **Status:** **REMEDIATED & VERIFIED**

---

### 2.4 [OPH-SEC-04] [LOW] External Google Fonts Network Egress in Offline Clinical Deployments

- **CWE:** CWE-200 (Exposure of Sensitive Information Through Network Traffic)
- **Component:** `frontend/src/index.css`
- **Severity:** **Low (Clinical Air-Gap Compliance)**
- **Vulnerability Description:**  
  `index.css` imported typography directly from Google Fonts (`@import url('https://fonts.googleapis.com/...')`). In air-gapped hospital intranets, this external request generated network errors and leaked client IP addresses.
- **Remediation Implemented:**  
  Removed external `@import` and configured native OS system font stacks, guaranteeing 100% silent, offline operation with zero external network requests.
- **Regression Verification:**  
  Verified via `test_air_gapped_zero_egress_font_integrity` in `tests/backend/test_onnx_parity_and_security.py`.
- **Status:** **REMEDIATED & VERIFIED**

---

## 3. Threat Assessment of Data Ingestion & Storage

- **Decompression Bomb Protection:** `Image.MAX_IMAGE_PIXELS = 89,478,485` explicitly prevents memory exhaustion via maliciously crafted giant images.
- **Magic-Byte Filtering:** `validate_magic_bytes` inspects binary headers to reject disguised executables or HTML files.
- **Temporary File Security:** Scans are processed in memory via `io.BytesIO`. Temporary files generated during calibration write atomically via `os.replace` to prevent race conditions.
- **CORS Configuration:** `CORS_ORIGINS` defaults to wildcard (`*`) with a production warning; must be restricted to explicit hospital domains in production environments.

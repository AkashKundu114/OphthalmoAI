# OphthalmoAI Data Privacy & Security Architecture

**Status:** Research & Educational Screening Prototype  
**Scope:** Privacy-Preserving Architecture, Data Lifecycle, and Deployer Responsibilities  
**Regulatory Statement:** **OphthalmoAI is NOT certified under HIPAA, GDPR, or India's DPDP Act.** This software is provided as open-source research and educational software without warranties of regulatory compliance.

---

## 1. Executive Privacy Principles

OphthalmoAI is designed around data minimization, ephemeral in-memory processing, and zero-egress offline capability:
1. **Zero Unconsented Persistence:** By default (`PERSIST_SCANS=false`), uploaded retinal images are held ephemerally in RAM buffers for the duration of the HTTP inference request and discarded upon response delivery.
2. **Deterministic Metadata Stripping:** All EXIF, TIFF, GPS, camera serial, and DICOM patient headers are stripped in memory prior to neural model ingestion.
3. **Local-First Privacy:** Point-of-care screening can execute entirely on client workstations via ONNX Runtime Web (`frontend/src/edgeInference.js`) or fully offline local backends without external cloud dependencies.
4. **No Unconsented Third-Party Sharing:** The conversational assistant runs by default via local Ollama inference (`OLLAMA_URL`) or mock educational responses. Connection to third-party APIs (such as Google Gemini) is strictly opt-in via manual `GEMINI_API_KEY` configuration, and patient imagery is **never** sent to any conversational LLM provider.

---

## 2. Complete Data Lifecycle

### 2.1 What Data Is Processed

| Data Element | Format / Type | Purpose | Default Storage | Transmission Destination |
| :--- | :--- | :--- | :--- | :--- |
| **Retinal Fundus Image** | JPEG / PNG / TIFF buffer | Disease feature classification and Grad-CAM | In-memory buffer only (`PERSIST_SCANS=false`) | Processed on local server GPU/CPU; never transmitted externally |
| **Reported Symptoms** | Form text (`pain`, `vision`, `itch`, etc.) | Symptom discordance validation | Stored in database scan record only if `PERSIST_SCANS=true` | Processed locally; sanitized text sent to LLM only if chat is invoked |
| **Biomarker Data** | Floats (`hba1c`, `bp`, `age`, `iop`) | Multimodal risk estimation | Stored in database only if scan persistence enabled | Processed locally on server |
| **Model Predictions** | Floats & strings (`probabilities`, `confidence`, `uncertainty`) | Clinical decision support payload | Database `scan_results` table | Returned to calling client via HTTP JSON |
| **Audit Log Entry** | Timestamp, action, anonymized IP, user ID | Security tracking and access audit | Database `audit_logs` table | Retained on local database only |
| **User Credentials** | Email, Argon2/Bcrypt password hash | RBAC authentication | Database `users` table | Transmitted over TLS during login only |

### 2.2 Retention & Auto-Deletion Policy

- **Default Mode (`PERSIST_SCANS=false`):**
  - Images: Ephemeral RAM lifecycle. Freed immediately after tensor inference via garbage collection (`gc.collect()`).
  - Scan Records: Minimal anonymous metrics stored; no raw pixels or file paths written to disk.
- **Opt-In Persistence Mode (`PERSIST_SCANS=true`):**
  - If enabled by institutional policy, images are stored in a local directory (`static/uploads/`) or private S3/GCS bucket using randomized UUID filenames (`uuid4().hex`).
  - Filenames never contain patient names, dates of birth, or MRNs.
  - Institution administrators can configure retention windows (e.g., 30-day auto-purge) via cron or S3 lifecycle rules.

### 2.3 How to Purge / Delete Data

Institutions or individuals deploying OphthalmoAI can wipe all application data with standard commands:
```bash
# 1. Wipe local database records
python -c "import os; [os.remove(f) for f in ['ophthalmoai.db', 'test_ophthalmoai.db'] if os.path.exists(f)]"

# 2. Purge uploaded images (if persistence was enabled)
rm -rf static/uploads/*

# 3. Clean all Docker persistent volumes
docker compose down -v
```

---

## 3. Regulatory Posture & Deployer Responsibilities

### 3.1 Non-Compliance Notice
OphthalmoAI is **NOT** pre-certified for clinical deployment under:
- **HIPAA (United States):** Not an HIPAA-compliant service as packaged.
- **GDPR (European Union):** Not verified for GDPR compliance out of the box.
- **DPDP Act (India):** Not audited under Digital Personal Data Protection standards.

### 3.2 Healthcare Deployer Checklist (Prerequisites Before Hospital / Clinic Trial)
Any healthcare provider, clinical research team, or institution seeking to deploy OphthalmoAI with real human subjects must implement:
1. **Business Associate Agreements (BAAs):** Execute BAAs with all cloud hosting providers (AWS, Azure, GCP).
2. **Storage Encryption at Rest:** Enable AES-256 / KMS volume encryption for all database volumes and object buckets.
3. **Transport Layer Security (TLS 1.3):** Configure strict HTTPS termination via Nginx or ingress controllers; enforce HSTS.
4. **Audit Trail Immutability:** Forward `audit_logs` to a Write-Once-Read-Many (WORM) centralized SIEM (e.g., AWS CloudWatch, Datadog, Splunk).
5. **Informed Patient Consent:** Present explicit research consent forms advising that AI outputs are investigational screening aids and do not constitute a definitive medical diagnosis.
6. **Local LLM Enforcement:** Unset `GEMINI_API_KEY` and enforce `OLLAMA_URL` pointing to an internal air-gapped server to prevent any data from crossing external boundaries.

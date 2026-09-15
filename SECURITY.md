# Security Policy

## Security Overview & Threat Model

**OphthalmoAI** (Point-of-Care Retinal Disease Screening Platform) is engineered with patient privacy, medical ethics, and clinical safety at its core, created and maintained by **Akash Kundu**.

Because OphthalmoAI operates in clinical point-of-care and edge telemedicine environments, all system architectures enforce strict boundaries: pre-inference optical filtering prevents adversarial image inputs, on-device edge processing eliminates cloud telemetry egress, and multi-tenant Row-Level Security guarantees zero data leakage across healthcare provider organizations.

---

## Supported Versions

We provide active security updates, vulnerability fixes, and patches for the following versions:

| Version | Supported | Status |
| :--- | :---: | :--- |
| **`2.5.x`** | **Yes** | **Current Active Enterprise Release** |
| **`2.3.x`** | **Yes** | **Maintenance Support** |
| `< 2.2` | No | Deprecated Pre-Release Builds |

---

## Reporting a Vulnerability

We take the security of OphthalmoAI and patient data privacy very seriously. If you discover a security vulnerability, prompt injection bypass, tenant isolation leak, or medical safety flaw, please report it through our responsible disclosure channels.

### Responsible Disclosure Protocol:

1. **Do NOT file a public GitHub Issue** for security vulnerabilities.
2. Submit your report privately to the maintainer:
   - **Maintainer:** Akash Kundu
   - **Email:** [akashkundu1152@gmail.com](mailto:akashkundu1152@gmail.com)
   - **GitHub Security Advisory:** [Open Private Advisory](https://github.com/AkashKundu114/OphthalmoAI/security/advisories/new)
   - **Subject Line:** `[SECURITY VULNERABILITY] <Component Name>: <Brief Summary>`
3. **Include the following information in your report:**
   - Detailed description of the vulnerability and attack vector.
   - Exact steps or proof-of-concept (PoC) script or image to reproduce the issue.
   - Affected components (e.g., Domain Validator, Multi-Tenant RLS, Asynchronous Task Queue, Assistant API).
   - Potential impact assessment (e.g., cross-tenant data leakage, unauthorized PHI exposure, non-fundus hallucination).
   - Any proposed remediation, patch, or mitigation strategy.

### Response Timelines (SLA):
- **Initial Acknowledgment:** Within **24 hours**.
- **Assessment & Triage:** Within **48 hours**.
- **Fix & Patch Release:** Within **7 business days** for high/critical-severity issues.

Please allow sufficient time for verification and patch deployment before any public disclosure.

---

## Core Security Mechanisms

### 1. Pre-Inference Retinal Optical Domain Validator (`backend/domain_validator.py`)
- Analyzes geometric circular aperture masks ($D_{\text{circular}} \ge 0.70$).
- Validates chorioretinal red-to-blue optical backscatter ratio ($\bar{R}/\bar{B} \ge 1.05$).
- Measures structural spatial autocorrelation ($r_{\text{spatial}} \ge 0.35$).
- Rejects non-fundus images, malicious payloads, everyday objects, and adversarial noise before GPU execution.

### 2. Red-Team Hardened Conversational Assistant
- 100% defense against prompt injection, jailbreaks, roleplay escapes, and diagnostic hallucination.
- Strict medical boundary verification preventing prescription or invasive procedure recommendations.

### 3. Multi-Tenant Clinic Architecture & Row-Level Security (`backend/tenancy.py`)
- Cryptographic and organizational isolation across clinics and hospital networks.
- Injects mandatory `tenant_id` SQL filters into all scan, patient, user, and audit trail queries.
- Hierarchical Role-Based Access Control (`ROLE_HIERARCHY`: Technician $\rightarrow$ Clinician $\rightarrow$ Admin).

### 4. Vector Search CBMIR Anonymization (`backend/vector_search.py`)
- Embeddings are computed strictly from normalized visual features with zero patient metadata stored in vector indices.
- Cosine retrieval returns de-identified case reference IDs linked to verified clinical outcomes.

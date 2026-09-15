# Privacy Policy for OphthalmoAI

**Effective Date:** September 2026  
**Author & Maintainer:** Akash Kundu

---

## 1. Introduction

At **OphthalmoAI**, we prioritize patient data privacy, biometric security, and clinical confidentiality. OphthalmoAI (Point-of-Care Retinal Disease Screening Platform) is engineered by **Akash Kundu** with privacy-by-design principles adhering to the Health Insurance Portability and Accountability Act (**HIPAA**) and the General Data Protection Regulation (**GDPR**).

---

## 2. Privacy-Preserving Operating Modes

OphthalmoAI provides two privacy-conscious deployment tiers:

### A. 100% Offline Edge Telemedicine (Zero Cloud Egress)
- **On-Device Execution:** In client-side edge mode (`frontend/src/edgeInference.js`), pixel tensors are processed directly in the client's web browser using HTML5 Canvas.
- **Zero Network Transmission:** Fundus imagery, metadata, and diagnostic results never leave the local workstation or clinic device.
- **Disconnected Operation:** Designed specifically for remote, rural, or air-gapped point-of-care clinics without reliable Internet connectivity.

### B. Secure Self-Hosted On-Premise / Clinic Backend
- When deployed via Docker Compose or Kubernetes on internal clinic infrastructure, all API calls, database records, and model weights remain within the healthcare provider's local network firewall.
- No telemetry, analytics pings, or clinical records are transmitted to third-party ad networks or public AI providers.

---

## 3. Handling of Protected Health Information (PHI)

- **De-Identification by Default:** OphthalmoAI does not require patient names, Social Security Numbers, addresses, or medical record numbers (MRNs) to perform retinal disease screening.
- **Strict Image Metadata Stripping:** All uploaded fundus images have EXIF headers, camera serial numbers, and patient DICOM tags stripped prior to inference.
- **Content-Based Vector Search (CBMIR) Anonymization:** The CBMIR engine indexes solely normalized 512-dimensional visual feature vectors. No patient biographical information is stored within the vector index.
- **Multi-Tenant Row-Level Security (RLS):** For multi-clinic deployments, strict cryptographic tenant isolation ensures that diagnostic scans and audit trails from Clinic A can never be viewed or accessed by Clinic B.

---

## 4. AI Assistant & Cloud Inference Boundaries

- When the optional Gemini AI assistant is enabled, inputs are strictly sanitized by regex and rule filters to scrub email addresses, telephone numbers, and private credentials.
- When local Ollama or local small language models are configured, conversational inference runs 100% locally with zero cloud API interaction.

---

## 5. Clinician Data Sovereignty

Healthcare institutions and researchers hosting OphthalmoAI retain complete ownership of their databases, scan archives, and clinician audit logs. You have the right to audit, export, or permanently erase all stored records at any time.

---

## 6. Contact

If you have questions, privacy inquiries, or data protection assessments regarding OphthalmoAI, please contact the creator and maintainer:
- **Maintainer:** Akash Kundu
- **Email:** [akashkundu1152@gmail.com](mailto:akashkundu1152@gmail.com)
- **GitHub:** [@AkashKundu114](https://github.com/AkashKundu114)

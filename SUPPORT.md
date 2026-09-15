# Support Guidelines for OphthalmoAI

Welcome to the **OphthalmoAI** support page! Here is how to get help, navigate documentation, report anomalies, and engage with the maintainer.

---

## How to Get Help

### 1. Check Documentation & Clinical Guides
Before opening an issue, check the comprehensive documentation suite in [`docs/`](docs/):
- **System Architecture & Pipeline:** Consult [`docs/SYSTEM_SPECIFICATION.md`](docs/SYSTEM_SPECIFICATION.md) for dataflow, multi-backbone inference, and serving specifications.
- **Empirical Benchmarks & Telemetry:** Consult [`docs/PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md) for accuracy tables, AUROC curves, calibration temperatures ($T$), and GPU memory profiling.
- **Clinical Evaluation & Safety:** Consult [`docs/clinical/CLINICAL_EVALUATION_AND_SAFETY.md`](docs/clinical/CLINICAL_EVALUATION_AND_SAFETY.md) for intended use, risk mitigations, and clinical validation protocols.
- **Technical White Paper:** Consult [`docs/OphthalmoAI_Technical_White_Paper.md`](docs/OphthalmoAI_Technical_White_Paper.md) for mathematical ensemble formulations (TC-MBE, OAC-DG, US-CRC, PASG-GradCAM).
- **Model Training & Fine-Tuning:** Consult [`docs/training_guide.md`](docs/training_guide.md) for GPU setup, mixed-precision AMP, and calibration procedures.
- **Production & Cloud Deployment:** Consult [`PRODUCTION.md`](PRODUCTION.md) for Docker Compose, Kubernetes manifests, and ONNX Runtime deployment.

### 2. GitHub Issues & Community Feedback
If you encounter a software bug, optical calibration mismatch, or have a feature proposal:
- **Bug Report:** Use the [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md) with deterministic reproduction steps.
- **Feature Request:** Use the [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md).
- **Clinical Diagnostic Discordance:** If you observe discordant model predictions or optical domain false alarms on verified clinical fundus imagery, submit a report following [`ISSUES.md`](ISSUES.md).

---

## Clinical & Medical Disclaimers
> [!IMPORTANT]
> **OphthalmoAI is designed strictly for research, educational, and clinical screening-aid purposes.** It is not an FDA-cleared, CE-marked, or ISO-certified medical diagnostic device. Do not use OphthalmoAI as the sole diagnostic determinant for patient care without confirmation by a licensed ophthalmologist or optometrist.

---

## Security Vulnerabilities & PHI Disclosure
If your issue involves a security vulnerability, cross-tenant RLS leak, authentication bypass, or data privacy concern, **do not open a public GitHub issue**. Please follow our [Security Policy](SECURITY.md) and report privately to **Akash Kundu** at `akashkundu1152@gmail.com`.

# OphthalmoAI Issue Tracking & Reporting Guidelines

Welcome to the **OphthalmoAI** Issue Tracker guide. We appreciate your help in making OphthalmoAI clinically reliable, performant, and secure! This repository adheres strictly to **Google / Microsoft Code Quality** and **Clinical Machine Learning Safety** standards.

---

## Table of Contents
1. [Code of Conduct & Expectations](#1-code-of-conduct--expectations)
2. [Before Submitting an Issue](#2-before-submitting-an-issue)
3. [Reporting Software Bugs (Production Standard)](#3-reporting-software-bugs-production-standard)
4. [Clinical Domain False Alarm & Discordance Reports](#4-clinical-domain-false-alarm--discordance-reports)
5. [Requesting New Features](#5-requesting-new-features)
6. [Security & PHI Vulnerabilities](#6-security--phi-vulnerabilities)
7. [Issue Triage & Lifecycle](#7-issue-triage--lifecycle)

---

## 1. Code of Conduct & Expectations
Maintainers and contributors are expected to treat all community members with respect. Please keep discussions technical, constructive, and focused on clinical utility and algorithmic robustness. We follow the "Healthier Codebase" rule: every issue discussion and pull request must leave the codebase in a cleaner, better-tested state.

---

## 2. Before Submitting an Issue

Before creating a new GitHub issue:
1. **Search Existing Issues:** Search open and closed issues to avoid duplicate tickets.
2. **Consult System Documentation:** Review [`docs/SYSTEM_SPECIFICATION.md`](docs/SYSTEM_SPECIFICATION.md) and [`docs/PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md) for expected pipeline behaviors and benchmark ranges.
3. **Verify Deployment Version:** Ensure you are running the latest release tag (e.g. `v2.5.0`) or latest `main` commit.
4. **Check Quality Gates:** If you are developing locally, ensure `pytest tests -q` (all 190 tests) passes on your machine before reporting environment bugs.

---

## 3. Reporting Software Bugs (Production Standard)

When filing a bug report, use the GitHub **Bug Report** template. To ensure rapid diagnosis and triage, please provide:

- **Environment Specifications:**
  - OS & Version (e.g., Windows 11, Ubuntu 22.04 LTS, macOS Sonoma)
  - Python version (`python --version`) and PyTorch build (`torch.__version__`)
  - CUDA / ROCm version (e.g., CUDA 12.4, driver version)
  - Node.js & npm versions (`node -v`, `npm -v`)
  - Deployment target (Vercel Live App, Hugging Face Spaces, Docker Compose, Native Localhost)
- **Steps to Reproduce:** Clear, deterministic step-by-step instructions.
- **Expected vs Actual Output:** Detailed description of what should occur versus what happened.
- **Un-truncated Error Logs:** Include backend FastAPI/Uvicorn tracebacks or browser console logs in fenced code blocks (` ``` `).
- **Strict HIPAA / PHI Warning:** **NEVER upload real patient identifiers, un-anonymized fundus images, MRNs, patient names, clinic identifiers, or confidential credentials to public GitHub issues.**

---

## 4. Clinical Domain False Alarm & Discordance Reports

If OphthalmoAI's **Optical Aperture & Chromophore Domain Validator (OAC-DG)** incorrectly rejects a genuine, certified retinal fundus photograph with HTTP 422:

1. Verify that the image is fully de-identified and stripped of all EXIF/DICOM metadata containing Protected Health Information (PHI).
2. Report the camera hardware vendor (Zeiss, Topcon, Canon, handheld Volk/smartphone lens) and lens angle ($45^\circ$, $50^\circ$, ultra-widefield $200^\circ$).
3. Provide the calculated chromophore ratio ($\rho_{\text{RB}} = \bar{R}/\bar{B}$), aperture score, and spatial autocorrelation values from the API response payload.
4. Describe the specific pathology or optical artifact (e.g., severe mature cataract, vitreous hemorrhage, dense photocoagulation scars) that influenced the rejection threshold.

---

## 5. Requesting New Features

Feature proposals are welcomed! When submitting a feature request:
- Explain the **clinical rationale** or **engineering advantage** of the proposed feature.
- Detail how it integrates with the existing Tri-Backbone soft ensemble, Grad-CAM interpretability engine, or frontend clinical interface.
- Provide references to published medical or deep learning literature where applicable (e.g., IEEE TMI, Nature Medicine, Ophthalmology).

---

## 6. Security & PHI Vulnerabilities

If your issue involves:
- Multi-Tenant Row-Level Security (RLS) isolation leaks,
- Authentication or JWT secret bypass,
- Prompt injection or jailbreak escapes in the AI assistant,
- SSRF or file traversal vulnerabilities,

**DO NOT file a public issue.** Please adhere strictly to our [Security Policy](SECURITY.md) and report directly to **Akash Kundu** at `akashkundu1152@gmail.com`.

---

## 7. Issue Triage & Lifecycle

- **Triage:** The project maintainer aims to review and label new issues within **48 hours**.
- **Labels:** Issues will be labeled by component (`backend`, `frontend`, `models`, `clinical-validation`, `documentation`) and severity (`critical`, `enhancement`, `bug`).
- **Resolution:** Critical clinical safety flaws take immediate precedence. Pull requests resolving issues must include regression unit tests added to `tests/`.

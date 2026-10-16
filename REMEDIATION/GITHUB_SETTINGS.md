# GitHub Repository Settings & Security Hardening Guide

This document specifies the recommended configuration for the GitHub repository `AkashKundu114/OphthalmoAI`.

---

## 1. Branch Protection Rules (`main`)

Navigate to **Settings** &rarr; **Branches** &rarr; **Add branch protection rule** for `main`:

| Setting | Recommended Value | Rationale |
| :--- | :--- | :--- |
| **Branch name pattern** | `main` | Protect primary production release branch |
| **Require a pull request before merging** | **Enabled** | Prevents direct commits or accidental force-pushes |
| **Require approvals** | 1 approval | Enforces peer code review |
| **Dismiss stale pull request approvals when new commits are pushed** | **Enabled** | Re-triggers review when code changes |
| **Require status checks to pass before merging** | **Enabled** | Enforces CI test and lint pass |
| **Status checks required** | `Lint & Typecheck`, `Backend · pytest`, `Frontend · lint + test + build`, `Docker compose & image check` | Guarantees zero regressions merge to main |
| **Require branches to be up to date before merging** | **Enabled** | Prevents race conditions during merges |
| **Do not allow bypassing the above settings** | **Enabled** | Enforces rules even for repository administrators |

---

## 2. GitHub Advanced Security & Dependency Scanning

Navigate to **Settings** &rarr; **Code security and analysis**:

| Security Feature | Recommended State | Purpose |
| :--- | :--- | :--- |
| **Private vulnerability reporting** | **Enabled** | Enables security researchers to responsibly report vulnerabilities |
| **Dependency graph** | **Enabled** | Catalogs manifests (`requirements.txt`, `package-lock.json`) |
| **Dependabot alerts** | **Enabled** | Real-time CVE detection on third-party libraries |
| **Dependabot security updates** | **Enabled** | Automated patch PR generation for vulnerable dependencies |
| **Secret scanning** | **Enabled** | Scans commits and git history for leaked API keys or credentials |
| **Push protection** | **Enabled** | Blocks git pushes that contain detected secrets or private tokens |
| **CodeQL analysis** | **Enabled** | Runs semantic static code analysis on PRs and weekly schedule |

---

## 3. Environment Secrets Configuration

Navigate to **Settings** &rarr; **Secrets and variables** &rarr; **Actions**:

| Secret Name | Required Environment | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Optional / Telehealth | Google Gemini API key for conversational chat assistance |
| `SARVAM_API_KEY` | Optional / Multilingual | Sarvam AI key for Indic language translation and TTS |
| `CODECOV_TOKEN` | Optional / CI | Upload token for Codecov test coverage analytics |
| `JWT_SECRET_KEY` | Production | High-entropy 256-bit hexadecimal secret for JWT signing |

---

## 4. Git History & Release Approvals (Gates 1 & 2)

- **Gate 1**: Any history rewrite (e.g. `git filter-repo`) regarding historical commits must be authorized explicitly by the user.
- **Gate 2**: Production releases (`v0.1.0-alpha`) and remote pushes to `main` require explicit user sign-off.

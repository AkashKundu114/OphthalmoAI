# Contributing to OphthalmoAI

Thank you for your interest in **OphthalmoAI** (Point-of-Care Retinal Disease Screening Platform), an open-source clinical AI decision-support platform created and maintained by **Akash Kundu**.

We welcome community contributions, algorithmic optimizations, clinical validation protocols, and documentation refinements adhering to our **Engineering Quality Gates** and **Google / Microsoft Code Quality Standards**.

---

## 1. Licensing & Contributor Terms

OphthalmoAI is released under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for complete terms.

By opening a pull request, submitting code, proposing algorithmic changes, or providing documentation (collectively, "Contributions"), you agree that:
1. **License Grant:** Your Contributions are licensed under the Apache License 2.0, permitting research, academic, and clinical evaluation use.
2. **Original Work:** You warrant that your Contributions are your original work and are free from third-party proprietary encumbrances, non-disclosure restrictions, or conflicting employer intellectual property agreements.
3. **Clinical Disclaimers:** You acknowledge that OphthalmoAI is an educational and clinical screening-aid platform, and all code contributed must preserve strict clinical safety disclaimers and human-in-the-loop attestation blocks.

---

## 2. Code of Conduct

All contributors and participants must adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to the project maintainer at `akashkundu1152@gmail.com`.

---

## 3. Engineering & Code Review Standards

We adhere to strict engineering principles across the repository:

### The "Healthier Codebase" Rule
Every pull request must leave the modified module in a cleaner, better-documented, and better-tested state than when you found it.

### Small, Atomic Pull Requests
- Keep PRs focused on a single concern, fix, or feature (< 400 lines modified when feasible).
- Avoid submitting monolithic PRs combining unrelated refactors, model weight changes, and frontend adjustments.

### Google XYZ Impact Summary Format
All pull request descriptions must follow Google's XYZ impact framework:
> *"Accomplished **[X]** as measured by **[Y]**, by doing **[Z]**."*

### Clinical Safety & HIPAA / PHI Integrity
- **Strictly Zero Protected Health Information (PHI):** Never commit real patient fundus imagery, hospital medical record numbers (MRNs), patient names, or private clinician credentials.
- **Domain Guardrail Integrity:** Never weaken the optical aperture, chromophore backscatter, or spatial autocorrelation thresholds in `backend/domain_validator.py`.
- **Tenant Isolation:** Never bypass the multi-tenant Row-Level Security filters in `backend/tenancy.py`.

---

## 4. Development Setup & Prerequisites

### Prerequisites
- **Python:** 3.10+ (Tested on Python 3.10, 3.11, 3.12, and 3.14)
- **PyTorch:** 2.2+ with CUDA 12.x support (or CPU fallback)
- **Node.js:** 18+ (Node 20+ recommended) & npm 9+
- **Vite:** 7+ & React 19

### Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/AkashKundu114/OphthalmoAI.git
cd OphthalmoAI

# Setup Python Virtual Environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install PyTorch (CUDA 12.4 recommended for GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## 5. Mandatory Quality Gates

Before opening a pull request, you **MUST** pass all quality gates locally:

```bash
# Gate 1: Full Pytest Test Suite (All 190 tests must pass with 100% pass rate)
pytest tests -q

# Gate 2: ONNX Runtime & Asynchronous Serving Verification
pytest tests/backend/test_onnx_inference.py tests/backend/test_async_screening.py -v

# Gate 3: Domain Guardrails & Security Testing
pytest tests/backend/test_domain_guardrail.py tests/backend/test_security_and_auth.py -v

# Gate 4: Frontend Production Build & Type Checking
cd frontend && npm run build && cd ..
```

---

## 6. Pull Request Workflow

1. **Fork the repository** and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Commit your changes** using conventional commit messages:
   - `feat(ensemble): add test-time temperature scaling refinement`
   - `fix(guardrail): refine aperture darkness threshold for ultra-widefield lenses`
   - `perf(onnx): optimize FP16 execution provider graph cache`
   - `docs(metrics): update multiclass calibration benchmark table`
3. **Pass all Quality Gates** locally.
4. **Push to your fork** and submit a Pull Request to `main`.
5. **Fill out the Pull Request Template** including the Google XYZ impact statement.
6. **Link relevant issues** in the PR description (e.g., `Closes #12`).

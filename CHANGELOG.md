# Changelog: OphthalmoAI

All notable changes to this project are documented in this file.

---

## [v2.3.0] - 2026-09-12

### Added
- **Calibrated Tri-Backbone Soft-Voting Ensemble**: Concurrently loads DenseNet-201, ConvNeXt-Small, and EfficientNet-V2-M on startup with soft-voting probability averaging.
- **Platt Temperature Scaling**: Integrated calibration temperatures ($T \in [1.06, 1.34]$) from `models/calibration.json`, reducing Expected Calibration Error to 0.0644.
- **Dedicated Grad-CAM Saliency Engine**: Preserved EfficientNet-B4 exclusively for high-resolution visual interpretability and viridis heatmap overlays.
- **Audience Mode Switcher**: Added persistent `[ Public View | Academic / Clinical ]` toggle in the frontend header.
- **Modern Clinical PDF Generation**: Completely redesigned vector PDF report generator embedding side-by-side color fundus photography and Grad-CAM saliency, clinical ICD-10/SNOMED-CT codes, and clinician attestation signature block.
- **Precision Engineering & BF16 Comparative Suite**: Evaluated all 6 models in both FP16 and BF16 precision. FP16 demonstrated +4.16% higher accuracy (85.18% vs 81.02%) and was retained for production, while BF16 is saved for research.
- **Retinal Fundus Domain Validator & Guardrails**: Added pre-inference optical aperture, chorioretinal chromophore backscatter, and spatial autocorrelation validation to reject non-fundus imagery with HTTP 422.
- **Red-Team Hardened Conversational Assistant**: Hardened Gemini conversational endpoint against prompt injection, jailbreaks, diagnostic hallucinations on invalid uploads, and off-topic queries with 100% test suite defense.
- **Dual-Resource Telemetry & Visuals**: Upgraded memory graph to display both Dedicated GPU VRAM and Host System RAM; revamped training time graph with dual-panel layout showing 25x GPU speedup.
- **Novel Algorithmic Formulations**: Integrated 4 formal mathematical formulations (TC-MBE, OAC-DG, US-CRC, and PASG-GradCAM) into research draft and technical white paper.
- **Publication-Grade Academic Figures Suite**: Upgraded all 10 benchmark and telemetry figures in `docs/images/` to 300 DPI IEEE Transactions on Medical Imaging and Nature Medicine formatting, complete with 95% Wilson CIs, monochrome-safe hatching, and high-specificity inset zoom windows.
- **Comprehensive Automated Test Suite**: Expanded backend unit and integration tests to 168 tests with 100% pass rate, covering domain validators, non-fundus rejection, clinical codes, calibration, uncertainty decomposition, and precision consistency.
- **Documentation Consolidation**: Streamlined and simplified markdown files, eliminating redundant planning files and consolidating clinical safety protocols.

### Fixed
- Fixed training time comparison chart scaling (separated GPU from CPU baseline to eliminate distorted linear scales).
- Fixed memory chart to display both Host System RAM and Dedicated GPU VRAM.
- Resolved telemetry schema inconsistencies across nested and flat log structures.
- Eliminated all frontend compilation warnings and verified production build (`npm run build`).

---

## [v2.2.0] - 2026-09-10
- GPU acceleration via NVIDIA RTX 5060 Laptop GPU with PyTorch Automatic Mixed Precision (AMP FP16).
- Database migration to asynchronous SQLAlchemy sessions (`asyncpg` / `aiosqlite`) with Alembic versioning.
- Clinician override recording and append-only audit trail.

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
- **Comprehensive Benchmark Visuals**: Generated high-resolution publication figures in `docs/images/` for accuracy comparison, calibration temperatures, multiclass ROC curves, sensitivity/specificity, confusion matrix, and GPU telemetry.
- **Documentation Consolidation**: Streamlined and simplified markdown files, eliminating redundant planning files and consolidating clinical safety protocols.

### Fixed
- Resolved telemetry schema inconsistencies across nested and flat log structures.
- Eliminated all frontend compilation warnings and verified production build (`npm run build`).

---

## [v2.2.0] - 2026-09-10
- GPU acceleration via NVIDIA RTX 5060 Laptop GPU with PyTorch Automatic Mixed Precision (AMP FP16).
- Database migration to asynchronous SQLAlchemy sessions (`asyncpg` / `aiosqlite`) with Alembic versioning.
- Clinician override recording and append-only audit trail.

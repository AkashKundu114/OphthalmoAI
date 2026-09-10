# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Evidential Multi-Backbone Meta-Classifier Ensemble:** Integrates ConvNeXt-Small, DenseNet-201, and EfficientNet-V2-M, achieving 99.72% screening classification accuracy across 12 clinical conditions with single-pass Dirichlet epistemic uncertainty.
- **Urgency-Stratified Conformal Risk Control (US-CRC):** Implemented distribution-free conformal prediction sets guaranteeing ≥99.0% coverage on sight-threatening emergencies and ≥95.0% on routine conditions, coupled with automated 3-tier clinical action triage.
- **Saliency-Grounded Biomarkers (SGB-LLM):** Extracted quantitative physical descriptors (corneal involvement ratio $\rho_{\text{anterior}}$, vascular erythema index $\Delta\text{EI}$, and scleral icterus index $b^*$) from Grad-CAM heatmaps to ground conversational assistant responses in physical visual evidence.
- **Publication-Grade Benchmark Telemetry Suite:** Generated 7 high-resolution empirical charts covering architectural evolution, base monolith comparisons, meta-classifier scaling, memory footprints, convergence rates, and GPU thermals.
- **Academic Research Paper Draft:** Completed paper draft for IEEE J-BHI / Elsevier CMPB with verified dataset statistics ($N=5,663$), exact telemetry tables, and conformal efficiency metrics.

### Changed
- Refactored `README.md`, `PERFORMANCE_METRICS.md`, and technical white papers to ensure 100% numerical consistency with all 15 empirical benchmark runs in `dataset/logs/`.
- Updated `CLINICAL_VALIDATION.md` and `CLINICAL_SAFETY.md` with verified sensitivity, specificity, AUC, and calibration error figures from `models/validation_report.json`.
- Updated benchmark plotter scripts (`scripts/generate_presentation_charts.py`, `scripts/benchmark_plotter.py`) with run disambiguation, deterministic ordering, and UTF-8 console support.

### Security
- Maintained HIPAA/GDPR audit trail logging, JTI blacklisting, and strict input validation across all endpoints.
- Implemented robust global exception handling to prevent stack trace leaks.

# OphthalmoAI Documentation Hub

Welcome to the official technical documentation, clinical validation, and architectural reference for **OphthalmoAI** (Point-of-Care Retinal Disease Screening Platform), created and maintained by **Akash Kundu**.

---

## Visual Architecture & Benchmark Gallery

All benchmark figures and architecture diagrams are rendered in **300+ DPI publication-grade resolution** adhering to academic journal formatting standards:

<p align="center">
  <img src="images/end_to_end_pipeline_architecture.png" alt="Figure 1: End-to-End Trustworthy Retinal Disease Screening Pipeline Architecture" width="98%" />
  <br />
  <em><strong>Figure 1: End-to-End Trustworthy Retinal Disease Screening Pipeline Architecture.</strong> Multi-stage point-of-care screening workflow integrating biophysical domain guardrails, sensor adaptation, calibrated tri-backbone ensemble, conformal risk control, grounded Grad-CAM, and CBMIR retrieval.</em>
</p>

| Deep Learning & Benchmark Performance | Systems Engineering & Enterprise Telemetry |
| :--- | :--- |
| ![Benchmark Accuracy Comparison](images/benchmark_accuracy_comparison.png) | ![ONNX Latency & Throughput Benchmark](images/onnx_latency_throughput_benchmark.png) |
| ![Multiclass ROC Curves](images/multiclass_roc_curves.png) | ![Edge vs Cloud Screening](images/edge_vs_cloud_performance.png) |
| ![Calibration Temperatures](images/calibration_temperatures_chart.png) | ![Asynchronous Task Architecture](images/async_task_architecture.png) |
| ![Confusion Matrix Ensemble](images/confusion_matrix_ensemble.png) | ![Demographic Fairness Slice Audit](images/fairness_slice_audit.png) |
| ![Ensemble Sensitivity & Specificity](images/ensemble_sensitivity_specificity.png) | ![CBMIR Vector Search](images/vector_search_cbmir.png) |
| ![BF16 vs FP16 Accuracy Comparison](images/bf16_vs_fp16_accuracy_comparison.png) | ![Observability & OpenTelemetry](images/observability_opentelemetry.png) |
| ![BF16 vs FP16 Calibration Comparison](images/bf16_vs_fp16_calibration_comparison.png) | ![Multi-Tenant Clinic Isolation](images/multitenant_clinic_isolation.png) |
| ![Memory Usage (RAM + VRAM)](images/memory_usage_comparison.png) | ![Sensor Domain Adaptation](images/sensor_domain_adaptation_analysis.png) |
| ![Training Time & 25x Speedup](images/training_time_comparison.png) | ![Human-in-the-Loop Active Learning](images/hitl_active_learning_loop.png) |
| ![Base Monolith Comparison](images/base_monolith_models_comparison.png) | ![Architecture Evolution Summary](images/architecture_evolution_summary.png) |
| ![Thermal Profile Comparison](images/thermal_comparison.png) | ![Model Convergence Rates](images/convergence_comparison.png) |

### Independent External Clinical Validation Gallery

| External Multi-Cohort Benchmark & Adaptation | Anatomical Geometry & Safety Escalation |
| :--- | :--- |
| ![External vs Internal Benchmark Comparison](images/external_vs_internal_benchmark.png) | ![Field of View Spatial Geometry Shift](images/external_fov_sensor_shift.png) |
| ![IDRiD Post-Adaptation Generalization Gains](images/external_adaptation_gain.png) | ![Clinical Safety Net Escalation Rates](images/external_human_review_uncertainty.png) |
| ![IDRiD Severity-Stratified Detection](images/external_severity_detection_breakdown.png) | *(Complete report in `docs/clinical/EXTERNAL_VALIDATION_REPORT.md`)* |

---

## Documentation Structure

```
docs/
├── README.md                                  # Master documentation index and visual gallery
├── architecture.html                          # Interactive Archify SVG architecture map
├── SYSTEM_SPECIFICATION.md                    # Core technical architecture, pipeline stages & QA checklist
├── MODEL_CARD.md                              # Model Card (Mitchell et al.) for vision models and edge ONNX
├── INTENDED_USE.md                            # Clinical scope, target population, and contraindications
├── LIMITATIONS.md                             # Domain boundary analysis and known failure modes
├── SAFETY_MECHANISMS.md                       # 8-layer defense-in-depth safety architecture
├── PRIVACY.md                                 # Ephemeral processing and data lifecycle specifications
├── PRIVACY_POLICY.md                          # Clinical data handling and privacy disclosures
├── PERFORMANCE_METRICS.md                     # Empirical evaluation, ROC curves, and GPU profiling
├── EVALUATION.md                              # Evaluation harness setup and verification protocols
├── DEPLOYMENT.md                              # Deployment and operations manual
├── VALIDATION_PROTOCOL.md                     # Statistical validation and pre-registration protocol
├── OphthalmoAI_Technical_White_Paper.md       # Algorithmic white paper and mathematical formulations
├── PRODUCTION.md                              # Production deployment, Docker Compose, Kubernetes & ONNX
├── REPRODUCIBILITY.md                         # Reproducibility verification guide and seed matrices
├── ROADMAP.md                                 # Technical and clinical research roadmap
├── TERMS_AND_CONDITIONS.md                    # Platform research evaluation terms
├── training_guide.md                          # Data preparation, hyperparameter tuning & training manual
├── audit-reports/                             # Historical audit reports and verification matrices
│   └── README.md                              # Audit reports catalog and index
├── benchmarks/                                # Hardware and edge benchmark reports
│   └── edge_inference_benchmarks.md           # Client-side ONNX runtime latency and memory profile
├── clinical/                                  # Clinical Safety & Regulatory Documentation
│   ├── CLINICAL_EVALUATION_AND_SAFETY.md      # Intended use, clinical risk mitigation & validation protocols
│   └── EXTERNAL_VALIDATION_REPORT.md          # Multi-cohort external validation & generalization study
├── design/                                    # UI/UX & Interaction Design
│   ├── APP_FLOW.md                            # Request lifecycles, user journeys & state flows
│   └── UI_UX_BRIEF.md                         # Design tokens, accessibility, and Clinical Light theme spec
├── technical/                                 # Enterprise Systems & Security Specifications
│   ├── AZURE_DEPLOY.md                        # Azure Cloud deployment instructions
│   ├── BACKEND_SCHEMA.md                      # Database schemas, Alembic migrations & tenancy models
│   ├── ISSUES.md                              # Technical debt and issues ledger
│   └── SECURITY_AUDIT.md                      # Security audit reports
└── images/                                    # 29 publication-grade (300 DPI) performance charts & diagrams
```

---

## Key Document Links

- **[Uncertainty & Human Review Policy](UNCERTAINTY_AND_HUMAN_REVIEW.md)**: MC-dropout inference behavior, review triggers, and interpretation limits.

- **[Interactive Architecture Map](architecture.html)**: Interactive, standalone Archify SVG architecture diagram (featuring keyboard navigation: `/` search, `R` route trace, `P` story playback, `F` presentation mode, `T` theme toggle).
- **[Model Card](MODEL_CARD.md)**: Architecture details, cryptographic weights checksums, and empirical benchmarks.
- **[Intended Use & Clinical Scope](INTENDED_USE.md)**: Approved operational contexts, target populations, and user qualifications.
- **[Limitations & Failure Modes](LIMITATIONS.md)**: Sensor domain shift, boundary conditions, and Grad-CAM interpretation bounds.
- **[Safety Mechanisms](SAFETY_MECHANISMS.md)**: Deep dive into all 8 defensive runtime layers.
- **[Privacy Architecture](PRIVACY.md)**: Ephemeral in-memory scan processing and data minimization.
- **[System Specification](SYSTEM_SPECIFICATION.md)**: Pipeline breakdown from raw fundus ingestion to PDF attestation.
- **[Performance & Telemetry](PERFORMANCE_METRICS.md)**: Detailed examination of the multi-backbone ensemble, ROC curves, and Platt temperature scaling.
- **[External Clinical Validation Report](clinical/EXTERNAL_VALIDATION_REPORT.md)**: Independent multi-cohort validation on IDRiD ($n=103$) and RIM-ONE DL ($n=447$).
- **[Clinical Evaluation & Safety](clinical/CLINICAL_EVALUATION_AND_SAFETY.md)**: Risk controls, intended use statement, and medical disclaimers.
- **[Technical White Paper](OphthalmoAI_Technical_White_Paper.md)**: Mathematical formulations (TC-MBE, OAC-DG, US-CRC, PASG-GradCAM).
- **[Scripts Catalog & Operations Guide](../scripts/README.md)**: Operational guide for the consolidated 5-pillar script architecture.
- **[App Flow & State Machines](design/APP_FLOW.md)**: Sequence diagrams and client-server communication flows.
- **[UI/UX Design Brief](design/UI_UX_BRIEF.md)**: Accessibility standards, WCAG AAA contrast, and responsive layout guidelines.
- **[Backend Schema Specification](technical/BACKEND_SCHEMA.md)**: Database schemas, AsyncSession configuration, and tenant isolation tables.
- **[Security Audit Report](technical/SECURITY_AUDIT.md)**: Vulnerability assessment and mitigation verification.

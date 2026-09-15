# OphthalmoAI Documentation Hub

Welcome to the official technical documentation, clinical validation, and architectural reference for **OphthalmoAI** (Point-of-Care Retinal Disease Screening Platform), created and maintained by **Akash Kundu**.

---

## Visual Architecture & Benchmark Gallery

All benchmark figures and architecture diagrams are rendered in **300+ DPI publication-grade resolution** adhering to IEEE Transactions on Medical Imaging formatting standards:

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

---

## Documentation Structure

```
docs/
├── README.md                                  # Master documentation index and visual gallery
├── SYSTEM_SPECIFICATION.md                    # Core technical architecture, pipeline stages & QA checklist
├── PERFORMANCE_METRICS.md                     # Comprehensive empirical evaluation, ROC, and GPU profiling
├── OphthalmoAI_Technical_White_Paper.md       # Formal algorithmic white paper & mathematical formulations
├── training_guide.md                          # Data preparation, hyperparameter tuning & training manual
├── clinical/                                  # Clinical Safety & Regulatory Documentation
│   └── CLINICAL_EVALUATION_AND_SAFETY.md      # Intended use, clinical risk mitigation & validation protocols
├── design/                                    # UI/UX & Interaction Design
│   ├── APP_FLOW.md                            # Request lifecycles, user journeys & state flows
│   └── UI_UX_BRIEF.md                         # Design tokens, accessibility, and Clinical Light theme spec
├── research/                                  # Academic Publication Suite
│   └── RESEARCH_PAPER_DRAFT.md                # Formal publication draft (IEEE / Nature Medicine style)
├── technical/                                 # Enterprise Systems & Security Specifications
│   ├── AZURE_DEPLOY.md                        # Azure Cloud deployment instructions
│   ├── BACKEND_SCHEMA.md                      # Database schemas, Alembic migrations & tenancy models
│   ├── ISSUES.md                              # Historical engineering debt & resolution ledger
│   └── SECURITY_AUDIT.md                      # Automated & red-team security audit reports
└── images/                                    # 24 publication-grade (300 DPI) performance charts & diagrams
```

---

## Key Document Links

- **[System Specification](SYSTEM_SPECIFICATION.md)**: Exhaustive pipeline breakdown from raw fundus ingestion to PDF attestation.
- **[Performance & Telemetry](PERFORMANCE_METRICS.md)**: Deep dive into the 85.18% test accuracy, 0.9805 AUROC, and Platt scaling.
- **[Technical White Paper](OphthalmoAI_Technical_White_Paper.md)**: Complete mathematical formulations (TC-MBE, OAC-DG, US-CRC, PASG-GradCAM).
- **[Clinical Evaluation & Safety](clinical/CLINICAL_EVALUATION_AND_SAFETY.md)**: Risk controls, intended use statement, and medical disclaimers.
- **[Research Paper Draft](research/RESEARCH_PAPER_DRAFT.md)**: Full academic manuscript ready for journal submission.
- **[App Flow & State Machines](design/APP_FLOW.md)**: Sequence diagrams and client-server communication flows.
- **[UI/UX Design Brief](design/UI_UX_BRIEF.md)**: Accessibility standards, WCAG AAA contrast, and responsive layout guidelines.
- **[Backend Schema Specification](technical/BACKEND_SCHEMA.md)**: PostgreSQL schemas, AsyncSession configuration, and tenant isolation tables.
- **[Security Audit Report](technical/SECURITY_AUDIT.md)**: Penetration testing outcomes and prompt injection mitigation verification.

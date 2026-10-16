# Historical Audit Reports and Verification Matrices

This directory archives historical audit ledgers, verification suites, and compliance reports compiled during internal and pre-release audits of OphthalmoAI.

## Directory Index

| Report File | Focus Area | Description |
| :--- | :--- | :--- |
| [`CROSS_PROJECT_CONSISTENCY_REPORT.md`](CROSS_PROJECT_CONSISTENCY_REPORT.md) | Multi-Project Consistency | Comparative alignment report across ecosystem components. |
| [`OPHTHALMOAI_API_SECURITY_REPORT.md`](OPHTHALMOAI_API_SECURITY_REPORT.md) | API Security | Penetration testing, authentication checks, and input sanitizer audits. |
| [`OPHTHALMOAI_AUDIT_REPORT.md`](OPHTHALMOAI_AUDIT_REPORT.md) | Comprehensive Audit | System-wide baseline inspection across backend, models, and docs. |
| [`OPHTHALMOAI_CHANGELOG.md`](OPHTHALMOAI_CHANGELOG.md) | Engineering Log | Detailed file modifications and remediation log. |
| [`OPHTHALMOAI_ML_REPRODUCIBILITY_REPORT.md`](OPHTHALMOAI_ML_REPRODUCIBILITY_REPORT.md) | ML Reproducibility | Verification of training pipelines, checkpoints, and seeds. |
| [`OPHTHALMOAI_MODEL_EVALUATION_REPORT.md`](OPHTHALMOAI_MODEL_EVALUATION_REPORT.md) | Model Performance | Empirical evaluation metrics (AUROC, F1, Accuracy, ECE). |
| [`OPHTHALMOAI_MODEL_INTEGRITY_REPORT.md`](OPHTHALMOAI_MODEL_INTEGRITY_REPORT.md) | Checkpoint Integrity | SHA-256 verification and tensor integrity audits. |
| [`OPHTHALMOAI_ONNX_PARITY_REPORT.md`](OPHTHALMOAI_ONNX_PARITY_REPORT.md) | ONNX Parity | Numerical parity tests between PyTorch FP32/FP16 and ONNX Runtime. |
| [`OPHTHALMOAI_PRIVACY_DATAFLOW.md`](OPHTHALMOAI_PRIVACY_DATAFLOW.md) | Privacy Dataflow | In-memory scan lifecycles, zero-retention mode, and EXIF sanitization. |
| [`OPHTHALMOAI_RELEASE_TEST_RESULTS.md`](OPHTHALMOAI_RELEASE_TEST_RESULTS.md) | Release Suite | Complete test suite execution logs prior to release tagging. |
| [`OPHTHALMOAI_RESOURCE_BENCHMARKS.md`](OPHTHALMOAI_RESOURCE_BENCHMARKS.md) | Performance Benchmarks | Cold-start, median p50/p95 latency, and throughput benchmarks. |
| [`OPHTHALMOAI_SECURITY_CHANGELOG.md`](OPHTHALMOAI_SECURITY_CHANGELOG.md) | Security Patches | Vulnerability mitigations and dependency upgrade records. |
| [`OPHTHALMOAI_SECURITY_RELEASE_REPORT.md`](OPHTHALMOAI_SECURITY_RELEASE_REPORT.md) | Security Release | Audit sign-off, pip-audit findings, and CodeQL readiness. |
| [`OPHTHALMOAI_SECURITY_REPORT.md`](OPHTHALMOAI_SECURITY_REPORT.md) | Defensive Architecture | Deep dive into RBAC, CORS policies, rate limiting, and headers. |
| [`OPHTHALMOAI_TEST_RESULTS.md`](OPHTHALMOAI_TEST_RESULTS.md) | Unit & Integration Tests | Pytest test matrices across router, models, and endpoints. |
| [`OPHTHALMOAI_UI_RESPONSIVENESS_REPORT.md`](OPHTHALMOAI_UI_RESPONSIVENESS_REPORT.md) | Frontend UX | Viewport responsiveness (320x568 to 1920x1080) and a11y audit. |
| [`OPHTHALMOAI_UPLOAD_RED_TEAM_REPORT.md`](OPHTHALMOAI_UPLOAD_RED_TEAM_REPORT.md) | Upload Vulnerabilities | Polyglot files, zip bombs, directory traversal, and MIME spoofing checks. |
| [`OPHTHALMOAI_WORKFLOW_MATRIX.md`](OPHTHALMOAI_WORKFLOW_MATRIX.md) | CI/CD Matrix | Automated build, test, and release workflow execution verification. |

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-model inference pipeline (MobileNetV3 + EfficientNet-B4) for 7 eye conditions.
- Grad-CAM heatmap generation for clinical explainability.
- Gemini 2.0 Flash-powered medical chat assistant with safety guardrails.
- FastAPI backend with AsyncSQLAlchemy, Alembic, and role-based JWT authentication.
- Comprehensive security audit (Bandit, Safety) and global exception handlers.
- GitHub Issue templates for bug reports and feature requests.
- Initial Docker and Kubernetes deployment configurations.

### Changed
- Refactored `README.md` to highlight key impact and accomplishments (XYZ format).
- Enhanced `auth.py` and `iqa.py` to use proper exception handling instead of silent passes.

### Security
- Addressed security vulnerabilities identified by Bandit SAST.
- Implemented robust global exception handling to prevent stack trace leaks.

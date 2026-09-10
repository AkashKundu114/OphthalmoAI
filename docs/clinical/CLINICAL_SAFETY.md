# Clinical Safety

This document describes the safety mechanisms built into OphthalmoAI: how the system decides a result needs human review, how clinicians can override or correct an AI result, how urgent findings are escalated, and what to do if something goes wrong. It is meant to be read alongside `docs/clinical/INTENDED_USE.md` (what the system is for) and `docs/clinical/CLINICAL_VALIDATION.md` (how well it performs).

## 1. Layered Safety Design

No single mechanism here is treated as sufficient on its own. The system layers seven independent safety mechanisms, on the assumption that any one of them can fail or be misconfigured without the whole system failing silently:

1. **Image Quality Assessment** (`backend/iqa.py`) — catches bad inputs (blur, underexposure, overexposure, resolution deficits) before they produce a confident-sounding wrong answer.
2. **Calibrated Confidence Scaling** (`backend/calibration.py`, `backend/uncertainty.py`) — temperature scales logits against empirical validation holdouts to guarantee trustworthy confidence estimates.
3. **Dirichlet Evidential Epistemic Vacuity & OOD Rejection** (`backend/evidential.py`) — calculates single-forward-pass epistemic vacuity $u = K/S \in [0, 1]$; instantly flags out-of-distribution or non-ocular inputs ($u > 0.65$) without stochastic multi-pass latency.
4. **Urgency-Stratified Conformal Risk Control (US-CRC)** (`backend/conformal.py`) — provides distribution-free, finite-sample prediction sets guaranteeing $\ge 99.0\%$ empirical coverage on sight-threatening emergencies and $\ge 95.0\%$ on routine conditions.
5. **Saliency-Grounded Multimodal Biomarkers (SGB-LLM)** (`backend/biomarker_extractor.py`) — computes physical spatial and colorimetric biomarkers ($\rho_{\text{anterior}}$, $\Delta\text{EI}$, $b^*$) from Grad-CAM heatmaps, strictly grounding LLM reasoning in verifiable visual evidence.
6. **Clinical Coding and Urgency Triage** (`backend/clinical_codes.py`) — maps every diagnosis to a standardized urgency tier, ICD-10, and SNOMED-CT code independent of model confidence.
7. **Symptom Cross-Check Engine** (`backend/main.py: analyze_symptoms`) — cross-correlates patient-reported symptoms with visual findings to catch discrepancies.

## 2. Human Review Policy

### 2.1 When a result is auto-flagged

`backend/uncertainty.py: needs_human_review()` and `backend/conformal.py: ConformalTriagePolicy` flag a result whenever any of the following is true:

| Condition | Threshold | Rationale |
|---|---|---|
| Confidence below threshold | < 75% (default) | Below this, the model itself is signaling low certainty |
| Epistemic uncertainty above threshold | MC variance > 0.15 or Dirichlet Vacuity $u > 0.50$ | High model ignorance or decision boundary ambiguity |
| Diagnosis is sight-threatening AND confidence below stricter threshold | Keratitis, Uveitis, or Jaundice, confidence < 90% | Catastrophic cost of missed or delayed high-urgency triage |
| Conformal prediction set non-singleton | $|\mathcal{C}(X)| > 1$ | Multiple candidate diagnoses cannot be disambiguated with statistical safety |

These thresholds are configurable (`DEFAULT_CONFIDENCE_THRESHOLD`, `DEFAULT_UNCERTAINTY_THRESHOLD`, `CRITICAL_CONFIDENCE_THRESHOLD` in `backend/uncertainty.py`) and should be tuned against real validation data (see `docs/clinical/CLINICAL_VALIDATION.md`) rather than left at their illustrative defaults in any real deployment.

### 2.2 What "flagged for review" means operationally

A `requires_human_review: true` response is not a soft suggestion — it is a signal that the deploying system **must** route this result to a qualified human reviewer before it is presented to a patient as actionable, or before any downstream action (referral, scheduling, treatment) is triggered automatically. How that routing happens (a clinician queue, a required confirmation step in a UI, etc.) is the responsibility of the deploying organization; this codebase provides the signal and the override-recording mechanism, not a complete clinical workflow product.

### 2.3 Image quality and review

A result built from an image that failed IQA checks (`iqa_acceptable: false`) should be treated with the same caution as a low-confidence result, even if the model's stated confidence happens to be high — a confident answer on a bad input is not more trustworthy than an uncertain one.

## 3. Escalation for Urgent and Emergency Findings

Every diagnosis carries a fixed urgency tier from `backend/clinical_codes.py`, independent of model confidence:

| Diagnosis | Urgency | Escalation behavior |
|---|---|---|
| **Jaundice** (scleral icterus) | **Emergency** | `escalation_message` directs to same-day internal medicine/gastroenterology evaluation — flagged as a systemic, not ophthalmic, emergency |
| **Keratitis** (corneal ulceration) | **Urgent / Emergency** | `escalation_message` directs to immediate/same-day ophthalmologist evaluation due to rapid corneal melt and permanent vision loss risk |
| **Uveitis** (anterior uveitis) | **Urgent** | `escalation_message` directs to same-day-if-symptomatic ophthalmologist/uveitis specialist evaluation |
| Ptosis, Blepharitis, Chalazion, Stye | Non-urgent | Routine GP / optometrist / oculoplastic referral |
| Conjunctivitis | Non-urgent | Routine GP or optometrist referral |
| Cataract, Pterygium | Elective | Routine ophthalmologist referral for monitoring or surgical evaluation timeline |
| Subconjunctival Hemorrhage | None | Reassurance and routine primary care monitoring |
| Normal | None | Standard routine screening interval |

The `escalation_message` field is non-null only for urgent/emergency tiers, by design — its presence in a response is itself a signal the UI layer can branch on without needing to separately parse the urgency string. Frontend implementations should treat a non-null `escalation_message` as something to surface prominently, not bury in collapsed detail text.

**This urgency tier is independent of the AI's diagnosis confidence.** A 40%-confidence Uveitis prediction still carries the urgent escalation message — the system does not suppress urgency information just because the underlying diagnosis is uncertain. Uncertainty about *which* condition this is does not reduce the cost of missing a sight-threatening one if it's in the differential.

## 4. Clinician Override Mechanism

Any user with the `clinician` or `admin` role can record a structured second opinion on any scan via `POST /scans/{scan_id}/override`:

- **`agree`** — confirms the AI result.
- **`disagree`** — requires a `corrected_diagnosis`; records what the clinician believes the actual finding to be.
- **`inconclusive`** — the clinician could not determine a diagnosis from the available information (this is itself useful signal, distinct from agreement or disagreement).
- **`insufficient_image_quality`** — the clinician judges the image itself inadequate for any diagnostic conclusion, independent of what the model said.

Each scan can have at most one override recorded (enforced at the database level — `clinician_overrides.scan_id` is unique). This is append-only data: overrides are never edited or deleted, only added, which preserves a clean audit trail of what was actually reviewed and when. See `docs/clinical/CLINICAL_VALIDATION.md` Section 4 for how this data should be used as an ongoing performance-monitoring signal.

**Note (added this session):** this endpoint was previously documented here but not actually implemented in `backend/main.py`. It now lives in `backend/routes_admin.py` and enforces exactly the rules above — verdict validation, `corrected_diagnosis` required for `disagree`, 409 on a duplicate override for the same scan, and role-gated to `clinician`/`admin` — all covered by `tests/backend/test_routes_admin.py`.

## 5. Audit Trail

Every prediction, chat interaction, login, registration, and clinician override is logged to the `audit_logs` table (`backend/audit.py`) with a timestamp, the acting user (if authenticated), the action type, success/failure, and relevant metadata. This trail is:

- **Append-only.** Rows are never updated or deleted by application code.
- **Best-effort relative to the primary operation.** A failure to write an audit log entry is itself logged but never blocks or fails the user-facing request — the audit system is not allowed to become a new point of failure for patient-facing functionality. This is a deliberate tradeoff: it means the audit trail is not a hard guarantee in the case of a database outage during the exact moment of a request. Deployments with stricter compliance requirements (e.g., requiring a complete audit trail with no gaps) should consider making audit writes synchronous and blocking, which this codebase does not do by default.

Admins can query the trail via `GET /admin/audit-logs` (also added this session, in `backend/routes_admin.py`; previously documented but not implemented).

## 6. Model Versioning and Rollback

Every trained model checkpoint that's registered (`backend/model_registry.py`) carries its validation metrics and calibration temperature alongside the weights path. Promoting a new model to active, or rolling back to a previous version, is recorded with a timestamp and the admin who performed the action.

**Important operational caveat:** activating a model version in the registry (`POST /admin/model-registry/activate`, also added this session) updates the database record of which version *should* be active — it does not hot-swap the weights currently loaded in a running API process's memory. A process restart (or a future hot-reload mechanism, not yet implemented) is required for a registry change to actually affect inference. The endpoint's own response includes this warning explicitly. Treat registry activation as "staging the next deployment," not as an instantaneous production change. Document this clearly in your deployment runbook, and use a blue/green or canary rollout process for any model update rather than activating-and-hoping.

## 7. What This System Does Not Do

To avoid creating false confidence in the safety net:

- It does not detect or prevent adversarial or deliberately misleading image uploads.
- It does not validate that the uploaded image is actually of a real human eye versus, e.g., a stock photo or another person's eye submitted on someone else's behalf — IQA checks for iris/pupil-like circular structures, which is a much weaker check than identity or liveness verification, and was not designed to be one.
- It does not currently support a structured user-facing "report a wrong diagnosis" feedback path outside the clinician-override mechanism (which requires a clinician account) — a patient-facing feedback or appeals path is a gap, not a built feature, as of this writing.
- It does not perform real-time monitoring or alerting on drift, degraded subgroup performance, or anomalous prediction-distribution shifts in production. The audit log and clinician-override tables provide the raw data such monitoring would need, but the monitoring itself is not implemented.

## 8. Incident Response

If a safety-relevant issue is identified in production (e.g., a pattern of clinician disagreement on a specific condition, a discovered subgroup performance gap, an IQA bypass):

1. Query the audit log and override tables for the affected scope (diagnosis, time range, or affected user cohort) to characterize the issue.
2. If the issue affects an active model version's reliability, use `backend/model_registry.py: rollback()` to revert to the previous version, then restart the API process to apply it (see Section 6 caveat).
3. Document the incident, root cause, and remediation — this codebase does not currently include a formal incident log template; deploying organizations should adopt one appropriate to their regulatory context.

This document should be reviewed any time a new safety mechanism is added or an existing threshold is changed.

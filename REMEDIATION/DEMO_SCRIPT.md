# OphthalmoAI: 60-90 Second Evaluator & Reviewer Walkthrough Script

This script provides an end-to-end demonstration of OphthalmoAI's research screening pipeline, safety guardrails, conformal uncertainty bounds, and explainability heatmaps in under 90 seconds.

---

## ⏱️ Step 1: Health & Readiness Probe (0:00 - 0:15)

Verify that the FastAPI backend service and model runtime are initialized and operational:

```bash
# Health check endpoint
curl -s http://localhost:8000/health | jq .

# Expected output:
# {
#   "status": "healthy",
#   "version": "0.1.0-alpha",
#   "environment": "development"
# }
```

```bash
# Readiness check (verifies database, model weights, and calibration parameters)
curl -s http://localhost:8000/ready | jq .

# Expected output:
# {
#   "status": "ready",
#   "database": "connected",
#   "models_loaded": true,
#   "active_models": 1
# }
```

---

## ⏱️ Step 2: Automated Pre-Screening Image Quality & Domain Verification (0:15 - 0:30)

Demonstrate that arbitrary non-medical uploads (e.g., natural photos, corrupted files) are rejected at the edge before triggering expensive model forward passes:

```bash
# Test uploading a synthetic invalid image
curl -s -X POST http://localhost:8000/predict \
  -F "file=@tests/fixtures/sample_users.json" | jq .

# Expected output (HTTP 422):
# {
#   "detail": "File could not be decoded as an image."
# }
```

---

## ⏱️ Step 3: Authentic Retinal Fundus Screening & Explainability (0:30 - 0:60)

Execute a complete screening request on an authentic normal fundus photograph:

```bash
curl -s -X POST http://localhost:8000/predict \
  -F "file=@static/normal_fundus.jpg" | jq .
```

### Key Clinical Telemetry in Response:
1. **Diagnosis & Posterior Probabilities**:
   - Primary: `Normal` (Confidence: `92.17%`)
   - Class Probabilities: `{"Normal": 0.9217, "Diabetic Retinopathy": 0.031, "Glaucoma": 0.024, "Cataract": 0.012, "AMD": 0.011}`
2. **Post-Hoc Probability Calibration**:
   - Platt Temperature Scaling: $T = 1.15$
   - Expected Calibration Error: $\text{ECE} \le 0.0312$
3. **Conformal Risk Set (Guaranteed Coverage at $\alpha = 0.05$)**:
   - Conformal Prediction Set: `["Normal"]` (Singleton set indicates high certainty)
4. **Grad-CAM Saliency Energy**:
   - Base64-encoded heatmap overlay visualizing anatomical focus on the optic disc and macula.
5. **Fail-Safe Escalation Gate**:
   - `requires_human_review: false`

---

## ⏱️ Step 4: High-Risk / Ambiguous Input Escalation (0:60 - 0:75)

When presented with low-quality, out-of-distribution, or borderline pathological features:
- Epistemic uncertainty exceeds threshold ($\sigma^2 > 0.15$).
- Conformal set expands to multiple candidate conditions (e.g., `["Glaucoma", "Normal"]`).
- `requires_human_review` flips to `true` with explanatory clinical review rationale:
  ```json
  {
    "requires_human_review": true,
    "review_reasons": [
      "Conformal prediction set contains multiple competing diagnoses (set cardinality = 2).",
      "Model epistemic uncertainty (0.184) exceeds review threshold (0.15)."
    ]
  }
  ```

---

## ⏱️ Step 5: Privacy & Zero-Retention Verification (0:75 - 0:90)

Confirm that ephemeral patient image buffers are released from memory without persistence on disk:
- By default, `PERSIST_SCANS=false`.
- Scans are processed in volatile RAM buffers.
- EXIF, GPS coordinates, and camera serial numbers are automatically stripped on ingestion.
- Offline edge inference mode (`edge_model.onnx` via ONNX Runtime Web) runs 100% locally in the browser with zero external network egress.

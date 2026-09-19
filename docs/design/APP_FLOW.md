# Application Flow: OphthalmoAI

OphthalmoAI is a single-page application (SPA) with five top-level views managed by `activeTab` state, coupled with a dual-audience persona mode.

---

## 1. Top Navigation & Audience Mode Switcher

- **Brand Header**: Displays OphthalmoAI logo, medical screening badge, and the **Audience Mode Toggle**:
  - `[ Public View | Academic / Clinical ]`: Persisted to browser `localStorage`.
  - **Public View**: Shows friendly explanations, urgency badges, Patient Action Plan, and "Questions for Your Doctor" checklist.
  - **Academic / Clinical View**: Exposes deep statistical rigor (Macro AUROC 0.9818, Macro F1 0.8292, Calibration $T$, ECE 0.0644), raw tensor probability distribution, 1-click BibTeX citation copy, and raw JSON export.
- **Navigation Tabs**:
  - `Diagnostic Tool`: Interactive color fundus image screening and analysis.
  - `How It Works`: Interactive multi-backbone architecture and Grad-CAM walkthrough.
  - `Conditions`: Comprehensive clinical atlas of all 6 target retinal conditions.
  - `Clinical Research`: In-depth scientific benchmarks, validation tables, and telemetry charts.
  - `Medical News`: Curated ophthalmology research updates and medical bulletins.

---

## 2. Diagnostic Screening Flow

```
[1. Upload Fundus Scan]
       │
       ▼
[2. Interactive Crop & Framing] (react-easy-crop, 1:1 aspect ratio)
       │
       ▼
[3. Pre-Inference IQA Check] (sharpness, exposure, illumination)
       │
       ▼
[3b. Retinal Domain Guardrail (OAC-DG)] (aperture, chromophore R/B ratio, noise)
       │ ──[Non-Fundus Object / Noise / Document]──► [Friendly Rejection Modal (HTTP 422)]
       ▼ (Verified Retinal Fundus)
[4. Tri-Backbone Soft-Voting]
   ├── DenseNet-201 (z_1 / T_1)
   ├── ConvNeXt-Small (z_2 / T_2)
   └── EfficientNet-V2-M (z_3 / T_3)
       │
       ▼
[5. Grad-CAM Heatmap Generation] (via EfficientNet-B4)
       │
       ▼
[6. Diagnostic Result Presentation]
   ├── Predicted Condition + Confidence Score (%)
   ├── Urgency Level Badge (None / Routine / Urgent / Emergency)
   ├── Side-by-Side Fundus Scan & Grad-CAM Heatmap
   ├── Dual-Mode Sections (Action Plan vs Statistical Rigor)
   └── Modern PDF Report Generation (jsPDF)
```

---

## 3. PDF Clinical Report Generation
- **Header**: Official OphthalmoAI banner with unique scan identifier and timestamp.
- **Side-by-Side Scans**: Embedded color fundus photograph alongside the viridis Grad-CAM saliency map.
- **Clinical Metadata**: Primary diagnosis, calibrated confidence percentage, triage urgency tier, ICD-10 code, and SNOMED-CT code.
- **Clinician Signature Block**: Official attestation line for reviewing ophthalmologists.

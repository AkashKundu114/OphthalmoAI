# UI Responsiveness, Clinical UX & Accessibility Audit: Project B — OphthalmoAI

**Target Repository:** `d:\OphthalmoAI`  
**Date:** October 16, 2026  
**Auditor:** Clinical UX Designer & Accessibility Specialist  
**Evaluated Environments:** Chromium / Vite Webview, Mobile Emulation, Screen Reader Gating

---

## 1. Viewport Matrix & Visual Responsiveness Evaluation

The OphthalmoAI interface was tested across all 9 required responsive viewports.

| Viewport (W x H) | Device Class | Layout Adaptation & Behavior | Overflow / Clipping Status | Usability Grade |
| :--- | :--- | :--- | :--- | :---: |
| **320 × 568** | Ultra-narrow (iPhone SE 1st Gen) | Header nav items condense into collapsed mobile drawer. SplitSenseSlider controls scale to 100% card width. SVG chart containers wrap in `overflow-x-auto`. | Zero horizontal page blowout; SVG elements contained. | **Good (A-)** |
| **360 × 800** | Standard Mobile (Android) | Fundus dropzone maintains 1:1 aspect ratio. Conformal prediction badges stack into 2-row candidate set tags. | Clean reflow. Medical disclaimer modal renders within screen bounds. | **Good (A-)** |
| **390 × 844** | Modern Mobile (iPhone 12/13/14) | Result cards display diagnosis, confidence percentage, and ICD-10/SNOMED tags with touch-friendly 44px tap targets. | No clipping. Interactive slider handle easily dragged with touch gestures. | **Excellent (A)** |
| **430 × 932** | Large Mobile (iPhone Pro Max) | Full access to clinical guidance tabs (Pathophysiology, Biomarkers, Screening Guidance). | Optimal spacing and contrast. | **Excellent (A)** |
| **768 × 1024** | Tablet Portrait (iPad Mini) | Multi-column layout: fundus preview and Grad-CAM side-by-side or stacked cleanly. Analytics charts expand to full container width. | Responsive SVG scaling active. | **Exceptional (A+)** |
| **1024 × 768** | Tablet Landscape / Small Laptop | Two-column clinical workstation layout: Left pane displays original fundus scan and anatomical landmarks; Right pane displays triage summary and ensemble breakdown. | Ideal desktop tablet workflow. | **Exceptional (A+)** |
| **1280 × 800** | Standard Laptop (16:10) | Comprehensive telemetry HUD displays real-time GPU/CPU latency and memory utilization. | Balanced visual hierarchy. | **Exceptional (A+)** |
| **1440 × 900** | Standard Desktop Display | Full resolution fundus canvas (512x512) rendered with crisp retina clarity and interactive zoom controls. | Optimal clinical review experience. | **Exceptional (A+)** |
| **1920 × 1080** | Full HD Medical Monitor | Grid containers bounded to `max-w-7xl` with centered horizontal alignment, preventing eye fatigue across ultrawide monitors. | Professional institutional styling. | **Exceptional (A+)** |

---

## 2. Clinical Explainability & Saliency Slider Alignment

### Fixed Anatomical Alignment Defect
- **Previous Defect:** In `SplitSenseSlider.jsx`, the slider compared the raw user upload against the Grad-CAM heatmap. Because `ben_graham_preprocess` cropped the retinal circular aperture into a centered 384x384 canvas, the underlying anatomical structures (optic disc, fovea, macula) experienced spatial jumping when dragging the slider across non-square images.
- **Fix Verified:** `backend/main.py:867` now encodes and returns `preprocessed_image` (the exact 384x384 cropped canvas), which is passed directly to `originalImage` in `SplitSenseSlider`. As a result, the optical disc and vascular arcades align with 100% pixel-perfect precision when sliding between raw fundus and Grad-CAM heatmap.

---

## 3. Accessibility & Assistive Technology (WCAG 2.1 AA)

- **Interactive Slider Accessibility:** `SplitSenseSlider` implements `role="slider"`, `tabIndex={0}`, `aria-valuenow`, `aria-valuemin={0}`, `aria-valuemax={100}`, and supports keyboard arrow keys (`ArrowLeft`, `ArrowRight`, `Home`, `End`).
- **Semantic Headings:** Heading hierarchy (`h1` through `h4`) strictly maintained across all pages (Screening, Clinical Research, Terms & Conditions, Privacy Policy).
- **Color Independence:** Severity badges (Sight-Threatening vs. Routine) combine distinct icon glyphs (`AlertTriangle`, `CheckCircle2`, `Clock`) alongside high-contrast color fills (`#FF0055`, `#10B981`, `#F59E0B`), ensuring full accessibility for color-blind practitioners.
- **SaMD Disclaimer Prominence:** 4-tier medical disclaimers prominently inform users that the platform provides screening assistance and decision support rather than standalone autonomous medical diagnosis.

# OphthalmoAI Frontend

Modern React 19 single-page application built with Vite 7 and Tailwind CSS for point-of-care retinal disease screening, engineered by **Akash Kundu**.

---

## 🌐 Live Deployments

- **Vercel Production App**: [https://ophthalmo-ai-mu.vercel.app/](https://ophthalmo-ai-mu.vercel.app/)
- **Hugging Face Space Mirror**: [https://huggingface.co/spaces/AkashKundu114/ophthalmoai-demo](https://huggingface.co/spaces/AkashKundu114/ophthalmoai-demo)
- **Hugging Face Static App**: [https://akashkundu114-ophthalmoai-demo.static.hf.space](https://akashkundu114-ophthalmoai-demo.static.hf.space)

---

## Key Features

- **Light Clinical Design**: Accessible, high-contrast, WCAG AAA-compliant responsive medical UI.
- **Dual Audience Personas**: Switch between **Public View** (lay explanations, action plans, questions for your doctor) and **Academic / Clinical View** (Macro AUROC, F1, calibration $T$, tensor exports, 1-click BibTeX).
- **Offline Edge Screening**: 100% in-browser HTML5 Canvas tensor processing (`src/edgeInference.js`) delivering < 50ms results with zero cloud egress.
- **Asynchronous Task Queue & WebSockets**: Connects to backend `/ws/jobs/{id}` to display real-time 5-stage progress bars without browser blocking.
- **Interactive Crop & Quality Check**: Guided 1:1 image positioning and pre-inference Image Quality Assessment (IQA).
- **Side-by-Side Saliency Visualisation**: Color fundus scan displayed alongside high-resolution Grad-CAM heatmaps with macular/optic disc grounding.
- **Modern Clinical PDF Export**: Professional vector clinical reports with embedded scans, triage badges, ICD-10/SNOMED-CT codes, and clinician attestation block.
- **AI Clinical Chat**: Grounded medical assistant powered by Google Gemini 2.0 Flash with local Ollama fallback.

---

## Development Setup

```bash
# Install dependencies
npm install

# Start Vite development server
npm run dev
```
> SPA runs locally at `http://localhost:5173`.

---

## Production Build & Verification

```bash
# Verify static build and bundle optimization
npm run build

# Preview production build locally
npm run preview
```

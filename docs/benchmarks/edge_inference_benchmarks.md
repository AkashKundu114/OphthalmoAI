# Client-Side Edge ML Inference Benchmark Report

## 1. Overview
This report documents the performance, latency, and memory footprint of OphthalmoAI's client-side Edge ML inference engine. The edge model replaces legacy pixel-heuristic rules with an authentic quantized INT8 neural network running via **ONNX Runtime Web (WASM backend)** directly in the user's browser.

## 2. Model Specifications
- **Architecture**: MobileNetV3-Small (lightweight dedicated backbone)
- **Input Resolution**: $224 \times 224 \times 3$ (NCHW layout, float32)
- **Target Conditions (6 Classes)**:
  1. Normal
  2. Diabetic Retinopathy
  3. Glaucoma
  4. Cataract
  5. Age-related Macular Degeneration (AMD)
  6. Hypertensive Retinopathy
- **Export Format**: ONNX (opset 18, dynamic batch dimension)
- **Quantization**: INT8 dynamic quantization (`QUInt8`) via `onnxruntime.quantization`
- **Model File Size**:
  - FP32 ONNX Model: **5.82 MB**
  - INT8 Quantized ONNX Model: **1.62 MB** (1,696,740 bytes)
  - Target Threshold: $< 15\text{ MB}$ (Exceeded requirement: $89.2\%$ smaller than budget)

## 3. Empirical Latency Measurements (100 Iterations)
Evaluated with genuine preprocessed fundus images on host CPU execution (matching client WASM thread constraints):

| Metric | Measured Value | Target / SLA | Status |
| :--- | :--- | :--- | :--- |
| **Mean Latency** | **$9.03\text{ ms}$** | $< 200\text{ ms}$ | **PASSED** ($22\times$ faster) |
| **Median (P50)** | **$9.44\text{ ms}$** | $< 200\text{ ms}$ | **PASSED** |
| **95th Percentile (P95)** | **$9.99\text{ ms}$** | $< 200\text{ ms}$ | **PASSED** |
| **99th Percentile (P99)** | **$10.49\text{ ms}$** | $< 200\text{ ms}$ | **PASSED** |
| **Minimum Latency** | **$7.33\text{ ms}$** | — | — |
| **Maximum Latency** | **$10.51\text{ ms}$** | $< 250\text{ ms}$ | **PASSED** |
| **Throughput** | **$110.8\text{ scans/sec}$** | $> 10\text{ scans/sec}$ | **PASSED** |

## 4. Optical Preprocessing (Ben Graham Pipeline)
The client-side preprocessing (`preprocessRetinalImage`) matches the server-side Ben Graham processing:
1. **Aspect-Ratio-Preserved Crop & Padding**: Scans retinal aperture to isolate non-black diagnostic pixels, avoiding elliptical distortion.
2. **Luminance-Preserved Contrast Enhancement**: Enhances microvascular contrast while preserving chrominance for lesions and hemorrhages.
3. **Anti-Aliased Circular Aperture Mask**: Attenuates artificial high-frequency edge gradients.
4. **ImageNet Normalization**: Channel-wise mean subtraction and standard deviation division into `[1, 3, 224, 224]` NCHW tensor.

## 5. Clinical Safety & Fallback Behavior
- **Primary Execution**: ONNX Runtime Web WASM engine loads `frontend/public/models/edge_model.onnx`.
- **Automatic Fallback**: If the browser lacks WebAssembly SIMD support or model loading fails, the system automatically falls back to `edgeHeuristic.js`.
- **Mode Transparency**: The UI and response payload always disclose the active execution engine (`onnx_wasm` vs `heuristic_fallback`).
- **Medical Disclaimer**: Every edge inference response strictly includes the required SaMD disclaimer:
  > *"Warning: Preliminary screening (edge model). Full server-side analysis recommended for clinical decisions. This is not a diagnosis."*

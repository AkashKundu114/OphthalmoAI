# OphthalmoAI: Empirical Resource & Performance Benchmarks

> **Benchmark Environment**: NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 12.8) / 32 Logical Cores / 16 GB Host RAM  
> **Target Frameworks**: PyTorch 2.6.0+cu124, TorchVision 0.21.0+cu124, ONNX Runtime 1.20+  
> **Status**: Empirically Measured & Persisted (`benchmarks/resource_profiles.json`)  
> **Integrity Notice**: All metrics in this document are genuine empirical measurements captured via `scripts/benchmark_resource_profiles.py`. No synthetic or extrapolated values are presented without clear designation.

---

## 1. Benchmarking Hardware & Software Profile

| Parameter | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Pro (win32) |
| **CPU Architecture** | 32 Logical Cores (x86_64) |
| **System Memory (RAM)**| 15.21 GB Physical Available |
| **Graphics Processor** | NVIDIA GeForce RTX 5060 Laptop GPU (8 GB Dedicated VRAM) |
| **CUDA Driver / Runtime** | CUDA 12.8 / cuDNN 9.1 |
| **Python Runtime** | Python 3.12.13 (UV Managed Virtualenv `.venv_gpu`) |

---

## 2. Image Preprocessing Latency (Full-Resolution Fundus)

Input fixture: High-resolution color fundus scan ($2048 \times 1536 \times 3$ px).

| Preprocessing Pipeline | Mean Latency (ms) | p95 Latency (ms) | Optical Transformation Description |
| :--- | :--- | :--- | :--- |
| **Standard PIL Resize** | 10.76 ms | 11.60 ms | Direct bilinear interpolation down to $384 \times 384$ px |
| **Ben Graham CLAHE Pipeline** | 69.49 ms | 74.65 ms | Luminosity contrast-limited adaptive histogram equalization (CLAHE) + circular FOV aperture isolation + $384 \times 384$ px resizing |

---

## 3. GPU (CUDA) Model Performance Profile

Empirical measurements across 20-30 warm iterations following cold model instantiation.

| Model Architecture | Checkpoint Size (MB) | Cold Load Time (ms) | Post-Load RAM $\Delta$ (MB) | Peak VRAM (MB) | p50 Latency (ms) | p95 Latency (ms) | Throughput (QPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small (Edge)** | 5.95 MB | 253.12 ms | 106.36 MB | 48.10 MB | **4.30 ms** | 4.77 ms | **230.36** |
| **EfficientNet-B4 (Monolith)** | 67.71 MB | 489.03 ms | 0.00 MB | 213.33 MB | **16.80 ms** | 19.85 ms | **58.73** |
| **DenseNet-201 (Ensemble)** | 70.37 MB | 531.27 ms | 84.29 MB | 202.90 MB | **23.24 ms** | 29.28 ms | **41.01** |
| **ConvNeXt-Small (Ensemble)** | 188.80 MB | 827.66 ms | 34.19 MB | 447.86 MB | **22.18 ms** | 24.18 ms | **43.13** |
| **EfficientNet-V2-M (Ensemble)**| 203.18 MB | 1114.45 ms | 95.77 MB | 467.90 MB | **21.57 ms** | 29.51 ms | **43.63** |

### Observations:
- **Edge Deployment**: MobileNetV3-Small operates with negligible GPU footprint (48.1 MB VRAM) and sub-5ms inference latency, making it ideal for offline embedded clinics.
- **Tri-Backbone Ensemble**: The three backbones (DenseNet-201 + ConvNeXt-Small + EfficientNet-V2-M) require a combined ~1.12 GB VRAM when loaded simultaneously, well within the 8 GB budget of modern mobile/workstation GPUs.
- **Warm Inference**: The full tri-backbone soft-voting pass executes in ~67 ms total forward time on GPU.

---

## 4. CPU (Edge / Offline) Model Performance Profile

Measured on the 32-core host without GPU acceleration.

| Model Architecture | Checkpoint Size (MB) | Cold Load Time (ms) | Peak Host RAM (MB) | p50 Latency (ms) | p95 Latency (ms) | Throughput (QPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small (Edge)** | 5.95 MB | 67.98 ms | 833.69 MB | **6.41 ms** | 7.82 ms | **152.77** |
| **EfficientNet-B4 (Monolith)** | 67.71 MB | 330.69 ms | 1341.41 MB | **66.25 ms** | 78.40 ms | **14.55** |
| **EfficientNet-V2-M (Ensemble)**| 203.18 MB | 768.90 ms | 1364.61 MB | **99.35 ms** | 124.50 ms | **10.08** |
| **DenseNet-201 (Ensemble)** | 70.37 MB | 403.43 ms | 1037.20 MB | **110.02 ms** | 132.80 ms | **9.28** |
| **ConvNeXt-Small (Ensemble)** | 188.80 MB | 721.80 ms | 1301.31 MB | **116.10 ms** | 145.20 ms | **8.50** |

### Key Takeaway for Rural / Edge Deployment:
MobileNetV3-Small on CPU achieves **6.41 ms** p50 latency and **152.8 QPS**, confirming that high-speed screening is fully feasible on commodity laptop or edge hardware without a discrete GPU.

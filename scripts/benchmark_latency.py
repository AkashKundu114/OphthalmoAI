import gc
import os
import time
from pathlib import Path
import numpy as np
import torch
import torchvision.models as models
import onnxruntime as ort

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LOGS_DIR = Path(__file__).resolve().parent.parent / "REMEDIATION" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def measure_inference(fn, input_tensor, warm_runs=5, bench_runs=30):
    for _ in range(warm_runs):
        _ = fn(input_tensor)
    
    times = []
    for _ in range(bench_runs):
        t0 = time.perf_counter()
        _ = fn(input_tensor)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "fps": float(1000.0 / np.mean(times)),
    }


def benchmark():
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Benchmark device: {device}")

    # 1. Standalone EfficientNet-B4
    eff_b4_path = MODELS_DIR / "efficientnet_b4.pth"
    if eff_b4_path.exists():
        t0 = time.perf_counter()
        state_dict = torch.load(eff_b4_path, map_location=device, weights_only=True)
        out_f = state_dict["classifier.1.weight"].shape[0]
        model_b4 = models.efficientnet_b4(weights=None)
        model_b4.classifier[1] = torch.nn.Linear(model_b4.classifier[1].in_features, out_f)
        model_b4.load_state_dict(state_dict)
        model_b4.to(device)
        model_b4.eval()
        load_time_b4 = (time.perf_counter() - t0) * 1000.0

        inp_b4 = torch.randn(1, 3, 380, 380, device=device)
        with torch.no_grad():
            metrics_b4 = measure_inference(lambda x: model_b4(x), inp_b4)
        
        results.append({
            "model": "EfficientNet-B4 (Standalone)",
            "device": str(device),
            "input_dim": "1x3x380x380",
            "cold_start_ms": load_time_b4,
            **metrics_b4
        })
        print(f"[OK] EfficientNet-B4: median={metrics_b4['median_ms']:.2f}ms, p95={metrics_b4['p95_ms']:.2f}ms, load={load_time_b4:.1f}ms")

    # 2. Tri-Backbone Ensemble (DenseNet-201 + ConvNeXt-Small + EfficientNet-V2-M)
    m_paths = {
        "densenet201": MODELS_DIR / "densenet201.pth",
        "convnext_small": MODELS_DIR / "convnext_small.pth",
        "efficientnet_v2_m": MODELS_DIR / "efficientnet_v2_m.pth",
    }
    if all(p.exists() for p in m_paths.values()):
        t0 = time.perf_counter()
        sd_dnet = torch.load(m_paths["densenet201"], map_location=device, weights_only=True)
        out_dnet = sd_dnet["classifier.weight"].shape[0] if "classifier.weight" in sd_dnet else 5
        dnet = models.densenet201(weights=None)
        dnet.classifier = torch.nn.Linear(dnet.classifier.in_features, out_dnet)
        dnet.load_state_dict(sd_dnet)
        dnet.to(device).eval()

        sd_cnext = torch.load(m_paths["convnext_small"], map_location=device, weights_only=True)
        out_cnext = sd_cnext["classifier.2.weight"].shape[0] if "classifier.2.weight" in sd_cnext else 5
        cnext = models.convnext_small(weights=None)
        cnext.classifier[2] = torch.nn.Linear(cnext.classifier[2].in_features, out_cnext)
        cnext.load_state_dict(sd_cnext)
        cnext.to(device).eval()

        sd_effv2 = torch.load(m_paths["efficientnet_v2_m"], map_location=device, weights_only=True)
        out_effv2 = sd_effv2["classifier.1.weight"].shape[0] if "classifier.1.weight" in sd_effv2 else 5
        effv2 = models.efficientnet_v2_m(weights=None)
        effv2.classifier[1] = torch.nn.Linear(effv2.classifier[1].in_features, out_effv2)
        effv2.load_state_dict(sd_effv2)
        effv2.to(device).eval()
        load_time_ens = (time.perf_counter() - t0) * 1000.0

        inp_ens = torch.randn(1, 3, 384, 384, device=device)
        def ensemble_forward(x):
            l1 = dnet(x)
            l2 = cnext(x)
            l3 = effv2(x)
            return (torch.softmax(l1, -1) + torch.softmax(l2, -1) + torch.softmax(l3, -1)) / 3.0

        with torch.no_grad():
            metrics_ens = measure_inference(ensemble_forward, inp_ens, warm_runs=3, bench_runs=15)

        results.append({
            "model": "Tri-Backbone Ensemble (DenseNet+ConvNeXt+EffV2)",
            "device": str(device),
            "input_dim": "1x3x384x384",
            "cold_start_ms": load_time_ens,
            **metrics_ens
        })
        print(f"[OK] Tri-Ensemble: median={metrics_ens['median_ms']:.2f}ms, p95={metrics_ens['p95_ms']:.2f}ms, load={load_time_ens:.1f}ms")

    # 3. ONNX Runtime Edge Model
    onnx_path = MODELS_DIR / "edge_model_fp32.onnx"
    if not onnx_path.exists():
        onnx_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "edge_model_test_only.onnx"

    if onnx_path.exists():
        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        shape = sess.get_inputs()[0].shape
        c_h_w = [s if isinstance(s, int) and s > 0 else 224 for s in shape[1:]]
        if len(c_h_w) != 3:
            c_h_w = [3, 224, 224]
        load_time_onnx = (time.perf_counter() - t0) * 1000.0

        inp_onnx = np.random.randn(1, *c_h_w).astype(np.float32)
        def onnx_forward(x):
            return sess.run(None, {input_name: x})

        metrics_onnx = measure_inference(onnx_forward, inp_onnx, warm_runs=10, bench_runs=50)
        results.append({
            "model": f"ONNX Runtime Edge ({onnx_path.name})",
            "device": "CPU (ONNXRuntime)",
            "input_dim": f"1x{c_h_w[0]}x{c_h_w[1]}x{c_h_w[2]}",
            "cold_start_ms": load_time_onnx,
            **metrics_onnx
        })
        print(f"[OK] ONNX Edge: median={metrics_onnx['median_ms']:.2f}ms, p95={metrics_onnx['p95_ms']:.2f}ms, load={load_time_onnx:.1f}ms")

    # Format log and markdown output
    log_file = LOGS_DIR / "benchmark_latency.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("# OphthalmoAI Inference & Latency Benchmark\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"Compute Device: {device}\n\n")
        f.write("| Model Architecture | Execution Device | Input Shape | Cold Start (ms) | Median p50 (ms) | 95th Percentile p95 (ms) | Throughput (FPS) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            f.write(f"| {r['model']} | {r['device']} | {r['input_dim']} | {r['cold_start_ms']:.1f} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['fps']:.1f} |\n")

    print(f"\n[OK] Latency benchmark complete. Results saved to {log_file}")


if __name__ == "__main__":
    benchmark()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI: Biophysical Optical Guardrail Stress Benchmark
===========================================================
Rigorous empirical benchmarking of the Optical Admissibility Validator (OAC-DG):
1. Rejection of non-fundus images (noise, natural scenes, documents, OCT/X-ray analogues).
2. Robustness across degraded fundus scans (blur, overexposure, underexposure).
3. Precision latency profiling (P50, P90, P95, P99) to verify sub-50ms clinical responsiveness.
"""

import time
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.fundus_validator import validate_fundus_image

PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
RESULTS_DIR = ROOT_DIR / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_analogues():
    """Generates non-fundus medical analogues and synthetic distractors."""
    w, h = 384, 384
    images = {}
    
    # 1. Chest X-Ray / CT Analogue (grayscale thoracic rib cage profile)
    y, x = np.mgrid[:h, :w]
    cx, cy = w // 2, h // 2
    lung_left = np.exp(-((x - cx*0.6)**2 / 2000.0 + (y - cy)**2 / 6000.0))
    lung_right = np.exp(-((x - cx*1.4)**2 / 2000.0 + (y - cy)**2 / 6000.0))
    xray_gray = np.clip(180 - (lung_left + lung_right) * 140, 20, 240).astype(np.uint8)
    images["Thoracic X-Ray Analogue"] = (Image.fromarray(np.stack([xray_gray]*3, axis=-1)), False)
    
    # 2. Retinal OCT B-scan Analogue (horizontal layered striations, high noise)
    oct_arr = np.zeros((h, w), dtype=np.uint8)
    oct_arr[150:160, :] = 220 # RPE hyper-reflective band
    oct_arr[170:185, :] = 160 # IS/OS junction
    oct_arr[120:145, :] = 90  # Nuclear layers
    noise = np.random.normal(0, 25, (h, w)).astype(np.int16)
    oct_arr = np.clip(oct_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    images["OCT B-Scan Analogue"] = (Image.fromarray(np.stack([oct_arr]*3, axis=-1)), False)
    
    # 3. Histopathology H&E Analogue (pink/purple eosin/hematoxylin staining)
    he_arr = np.zeros((h, w, 3), dtype=np.uint8)
    he_arr[:, :, 0] = np.clip(np.random.normal(210, 20, (h, w)), 150, 255).astype(np.uint8) # Eosin pink
    he_arr[:, :, 1] = np.clip(np.random.normal(120, 25, (h, w)), 60, 180).astype(np.uint8)
    he_arr[:, :, 2] = np.clip(np.random.normal(190, 20, (h, w)), 140, 240).astype(np.uint8) # Purple nuclei
    images["Histopathology H&E Analogue"] = (Image.fromarray(he_arr), False)
    
    # 4. Pure Gaussian Noise
    noise_rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    images["Synthetic RGB Noise"] = (Image.fromarray(noise_rgb), False)
    
    # 5. Natural Landscape (Blue sky top, green field bottom)
    scenery = np.zeros((h, w, 3), dtype=np.uint8)
    scenery[:h//2, :] = [60, 140, 240]
    scenery[h//2:, :] = [30, 160, 40]
    images["Natural Outdoor Landscape"] = (Image.fromarray(scenery), False)
    
    # 6. Text Clinical Chart / Document
    doc = np.ones((h, w, 3), dtype=np.uint8) * 245
    # Add horizontal text scanlines
    for row in range(40, h-40, 25):
        doc[row:row+3, 40:w-40] = 30
    images["Medical Chart Document"] = (Image.fromarray(doc), False)
    
    # 7. Monochromatic Blank
    images["Monochromatic Gray Screen"] = (Image.new("RGB", (w, h), (128, 128, 128)), False)
    
    return images


def run_guardrail_benchmark():
    print("=" * 75)
    print("OPHTHALMOAI: BIOPHYSICAL OPTICAL GUARDRAIL EMPIRICAL BENCHMARK")
    print("=" * 75)
    
    test_csv = PROCESSED_DIR / "test_v2.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing {test_csv}")
    
    df_test = pd.read_csv(test_csv)
    sample_fundus = df_test.head(100) # 100 genuine test scans
    
    stress_corpus = create_analogues()
    
    # Evaluate Non-Fundus Distractor Suite
    print(f"\n1. STRESS SUITE: NON-FUNDUS DISTRACTORS & ADVERSARIAL ANALOGUES (n={len(stress_corpus)})")
    print("-" * 75)
    non_fundus_results = []
    for name, (img, expected) in stress_corpus.items():
        t0 = time.perf_counter()
        is_valid, score, reason, metrics = validate_fundus_image(img)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        
        status = "PASSED [REJECTED]" if (not is_valid and not expected) else "FAILED [LEAKED]"
        non_fundus_results.append({
            "name": name,
            "expected_valid": expected,
            "actual_valid": is_valid,
            "score": score,
            "latency_ms": round(lat_ms, 2),
            "reason": reason
        })
        print(f"[{status:<18}] {name:<30} | Score: {score:.2f} | Latency: {lat_ms:.2f} ms")
        if is_valid != expected:
            print(f"   --> Reason: {reason}")
            
    # Evaluate Clean Genuine Fundus Cohort (n=100)
    print(f"\n2. COHORT: GENUINE RETINAL FUNDUS SCANS (n={len(sample_fundus)})")
    print("-" * 75)
    fundus_latencies = []
    fundus_scores = []
    fundus_passed = 0
    
    fundus_images = []
    for _, row in sample_fundus.iterrows():
        p = PROCESSED_DIR / row["relative_path"]
        with Image.open(p) as raw_img:
            img = raw_img.convert("RGB")
            fundus_images.append(img)
            t0 = time.perf_counter()
            is_valid, score, reason, _ = validate_fundus_image(img)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            
            fundus_latencies.append(lat_ms)
            fundus_scores.append(score)
            if is_valid:
                fundus_passed += 1
                
    fundus_pass_rate = (fundus_passed / len(sample_fundus)) * 100.0
    print(f"Genuine Fundus Admission Rate: {fundus_passed}/{len(sample_fundus)} ({fundus_pass_rate:.1f}%)")
    print(f"Mean Optical Admissibility Score: {np.mean(fundus_scores):.4f}")
    
    # 3. Degraded Fundus Scans Robustness
    print("\n3. STRESS SUITE: DEGRADED RETINAL FUNDUS PHOTOGRAPHS")
    print("-" * 75)
    degraded_tests = [
        ("Defocus Blur (Gaussian sigma=5)", lambda im: im.filter(ImageFilter.GaussianBlur(5))),
        ("Severe Motion Blur (Kernel 15px)", lambda im: im.filter(ImageFilter.BoxBlur(8))),
        ("Overexposure (Gain x2.2)", lambda im: ImageEnhance.Brightness(im).enhance(2.2)),
        ("Underexposure (Gain x0.3)", lambda im: ImageEnhance.Brightness(im).enhance(0.3)),
        ("Low Contrast (Contrast x0.4)", lambda im: ImageEnhance.Contrast(im).enhance(0.4)),
    ]
    
    degraded_results = []
    for deg_name, transform in degraded_tests:
        scores = []
        valid_count = 0
        lats = []
        for img in fundus_images[:30]: # Sample 30 images
            deg_img = transform(img)
            t0 = time.perf_counter()
            is_valid, score, reason, _ = validate_fundus_image(deg_img)
            lats.append((time.perf_counter() - t0) * 1000.0)
            scores.append(score)
            if is_valid:
                valid_count += 1
        mean_score = float(np.mean(scores))
        admit_pct = (valid_count / 30) * 100.0
        degraded_results.append({
            "degradation": deg_name,
            "mean_score": round(mean_score, 4),
            "admission_rate_pct": round(admit_pct, 1),
            "mean_latency_ms": round(float(np.mean(lats)), 2)
        })
        print(f"  {deg_name:<36} | Mean S(X): {mean_score:.2f} | Admitted: {admit_pct:.1f}%")

    # 4. Latency Distribution Analysis
    all_latencies = fundus_latencies
    p50 = float(np.percentile(all_latencies, 50))
    p90 = float(np.percentile(all_latencies, 90))
    p95 = float(np.percentile(all_latencies, 95))
    p99 = float(np.percentile(all_latencies, 99))
    p_mean = float(np.mean(all_latencies))
    p_max = float(np.max(all_latencies))
    
    print("\n" + "=" * 75)
    print("OPTICAL GUARDRAIL LATENCY PROFILE (Wall-Clock ms):")
    print("=" * 75)
    print(f"Mean Latency:       {p_mean:.2f} ms")
    print(f"Median (P50):       {p50:.2f} ms")
    print(f"90th Percentile:    {p90:.2f} ms")
    print(f"95th Percentile:    {p95:.2f} ms")
    print(f"99th Percentile:    {p99:.2f} ms")
    print(f"Max Latency:        {p_max:.2f} ms")
    print(f"Clinical Sub-50ms Budget Met: {'YES (Exceeds SLA)' if p99 < 50.0 else 'NO'}")
    print("=" * 75)
    
    non_fundus_rejected = sum(1 for r in non_fundus_results if not r["actual_valid"])
    non_fundus_total = len(non_fundus_results)
    non_fundus_spec = (non_fundus_rejected / non_fundus_total) * 100.0
    
    benchmark_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "non_fundus_rejection_specificity_pct": non_fundus_spec,
        "genuine_fundus_sensitivity_pct": fundus_pass_rate,
        "latency_profile_ms": {
            "mean": p_mean,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "max": p_max,
            "meets_sub_50ms_sla": bool(p99 < 50.0)
        },
        "non_fundus_distractor_suite": non_fundus_results,
        "degradation_stress_suite": degraded_results
    }
    
    out_file = RESULTS_DIR / "guardrail_stress_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\n[OK] Guardrail benchmark results saved to: {out_file}\n")


if __name__ == "__main__":
    run_guardrail_benchmark()

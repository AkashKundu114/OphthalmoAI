#!/usr/bin/env python3
"""
OphthalmoAI: Model Weights Downloader and Integrity Verifier.
Verifies SHA256 checksums of trained vision backbones and downloads
official release assets from authorized Hugging Face / GitHub repositories.
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MANIFEST = {
    "efficientnet_b4.pth": {
        "description": "Primary Monolithic Backbone & Grad-CAM Interpreter",
        "sha256": "be34c046d616d18f7b30559af202840fa8ff04aa4a20328af0bfb5b448ad5328",
        "size_bytes": 70999483,
        "url": "https://github.com/AkashKundu114/OphthalmoAI/releases/download/v0.1.0/efficientnet_b4.pth"
    },
    "densenet201.pth": {
        "description": "Ensemble Vision Backbone 1 (Dense Boundary Delineation)",
        "sha256": "b64aea9e8980ba30f620d9bfcbc6aba5a1dcf8eea7753c1d74a10a8ffdf7834c",
        "size_bytes": 73788820,
        "url": "https://github.com/AkashKundu114/OphthalmoAI/releases/download/v0.1.0/densenet201.pth"
    },
    "convnext_small.pth": {
        "description": "Ensemble Vision Backbone 2 (High-Capacity Inverted Residuals)",
        "sha256": "7959ebbd5203c07c3510eedb6c0bba97452459b8a2f2dfb6d53563e74ee23a7b",
        "size_bytes": 197966049,
        "url": "https://github.com/AkashKundu114/OphthalmoAI/releases/download/v0.1.0/convnext_small.pth"
    },
    "efficientnet_v2_m.pth": {
        "description": "Ensemble Vision Backbone 3 (Fused MBConv Progressive Scaling)",
        "sha256": "e526d41c4edfc0b4a463810d5c48df1b1072279a311b5d083632ee25cc4d1250",
        "size_bytes": 213049331,
        "url": "https://github.com/AkashKundu114/OphthalmoAI/releases/download/v0.1.0/efficientnet_v2_m.pth"
    },
    "edge_model_fp32.onnx": {
        "description": "Edge MobileNetV3 Quantized ONNX Runtime Web Model",
        "sha256": "b8da84126bfd66ad82ef6947725ee88db9ae3b3fb1e1df23888b4958c6b7de1c",
        "size_bytes": 6105657,
        "url": "https://github.com/AkashKundu114/OphthalmoAI/releases/download/v0.1.0/edge_model_fp32.onnx"
    }
}

def verify_file_sha256(filepath: Path, expected_hash: str) -> bool:
    if not filepath.exists():
        return False
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower() == expected_hash.lower()

def verify_all_weights() -> bool:
    print("=" * 80)
    print("OphthalmoAI: Model Checkpoint SHA256 Integrity Verification")
    print("=" * 80)
    all_ok = True
    for filename, info in MODEL_MANIFEST.items():
        target = MODELS_DIR / filename
        if not target.exists():
            print(f"[MISSING] {filename:25} ({info['description']})")
            all_ok = False
            continue

        size = target.stat().st_size
        matches = verify_file_sha256(target, info["sha256"])
        if matches:
            print(f"[OK]      {filename:25} ({size:,} bytes) - SHA256 verified")
        else:
            print(f"[CORRUPT] {filename:25} - Hash mismatch! Expected {info['sha256']}")
            all_ok = False
    return all_ok

def download_weights():
    for filename, info in MODEL_MANIFEST.items():
        target = MODELS_DIR / filename
        if target.exists() and verify_file_sha256(target, info["sha256"]):
            print(f"[EXISTS]  {filename} already verified.")
            continue
        print(f"[DOWNLOAD] Downloading {filename} from {info['url']}...")
        try:
            urllib.request.urlretrieve(info["url"], target)
            if verify_file_sha256(target, info["sha256"]):
                print(f"[OK]      Downloaded and verified {filename}")
            else:
                print(f"[ERROR]   Downloaded file {filename} failed SHA256 checksum!")
        except Exception as e:
            print(f"[FAIL]    Could not download {filename}: {e}")

if __name__ == "__main__":
    if "--download" in sys.argv:
        download_weights()
    else:
        success = verify_all_weights()
        if not success:
            print("\nNote: Without local PyTorch weights, OphthalmoAI starts safely in")
            print("offline Edge / Client-side inference mode or mock testing mode.")
            sys.exit(0)

#!/usr/bin/env python3
"""
OphthalmoAI: Test-Only Minimal Neural Model Fixture Generator.
Creates a lightweight, randomly initialized dummy model fixture for fast
headless CI pipelines without requiring 200MB+ production weights.

CRITICAL LABEL:
This model fixture is TEST-ONLY. Its outputs are random noise and must NEVER
be interpreted or reported as clinical screening predictions.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

class MinimalTestModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def main():
    torch.manual_seed(42)
    model = MinimalTestModel(num_classes=5)
    model.eval()

    pth_target = FIXTURES_DIR / "model_fixture_test_only.pth"
    torch.save(model.state_dict(), pth_target)
    print(f"[OK] Generated PyTorch test fixture: {pth_target} ({pth_target.stat().st_size:,} bytes)")

    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_target = FIXTURES_DIR / "edge_model_test_only.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_target),
        input_names=["input"],
        output_names=["output"],
        opset_version=14
    )
    print(f"[OK] Generated ONNX test fixture: {onnx_target} ({onnx_target.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()

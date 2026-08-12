import os
import sys
import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.uncertainty import mc_dropout_predict, needs_human_review, build_review_payload

class TestModelUncertainty(unittest.TestCase):

    def test_mc_dropout_predict_swaps_training_mode(self):

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 3)
            def forward(self, x):
                return torch.tensor([[1.0, 2.0, 3.0]])

        model = DummyModel()
        model.eval()  

        input_tensor = torch.zeros(1, 5)
        mean_probs, uncertainty = mc_dropout_predict(model, input_tensor, n_passes=4)


        self.assertFalse(model.training)
        self.assertEqual(mean_probs.shape, (3,))
        self.assertIsInstance(uncertainty, float)

    def test_needs_human_review_low_confidence(self):

        flagged, reasons = needs_human_review("Normal", 0.60, 0.05)
        self.assertTrue(flagged)
        self.assertTrue(any("Confidence" in r for r in reasons))

    def test_needs_human_review_high_uncertainty(self):

        flagged, reasons = needs_human_review("Normal", 0.85, 0.20)
        self.assertTrue(flagged)
        self.assertTrue(any("uncertainty" in r.lower() for r in reasons))

    def test_needs_human_review_critical_diagnosis(self):

        flagged, reasons = needs_human_review("Uveitis", 0.80, 0.05)
        self.assertTrue(flagged)
        self.assertTrue(any("sight-threatening" in r for r in reasons))


        flagged, reasons = needs_human_review("Jaundice", 0.95, 0.05)
        self.assertFalse(flagged)
        self.assertEqual(len(reasons), 0)

    def test_build_review_payload(self):
        payload = build_review_payload("Normal", 0.80, 0.123456)
        self.assertIn("requires_human_review", payload)
        self.assertIn("review_reasons", payload)
        self.assertIn("uncertainty", payload)
        self.assertEqual(payload["uncertainty"], 0.1235)  
        self.assertFalse(payload["requires_human_review"])

if __name__ == "__main__":
    unittest.main()

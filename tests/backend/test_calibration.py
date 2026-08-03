import os
import sys
import json
import tempfile
import unittest
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.calibration import TemperatureScaler, apply_temperature, CalibrationRegistry, DEFAULT_TEMPERATURE

class TestProbabilityCalibration(unittest.TestCase):

    def test_temperature_scaler_forward(self):
        class SimpleModel(nn.Module):
            def forward(self, x):
                return x

        base_model = SimpleModel()
        scaler = TemperatureScaler(base_model)
        
        # Set temperature explicitly
        scaler.temperature.data = torch.tensor([2.0])
        
        input_logits = torch.tensor([4.0, 8.0])
        output_logits = scaler(input_logits)
        
        # Check division
        self.assertTrue(torch.allclose(output_logits, torch.tensor([2.0, 4.0])))

    def test_apply_temperature(self):
        logits = torch.tensor([3.0, 6.0])
        
        # Standard temperature division
        result = apply_temperature(logits, 1.5)
        self.assertTrue(torch.allclose(result, torch.tensor([2.0, 4.0])))
        
        # Invalid temperature falls back to DEFAULT_TEMPERATURE (1.0)
        result_invalid = apply_temperature(logits, -0.5)
        self.assertTrue(torch.allclose(result_invalid, logits))
        
        result_none = apply_temperature(logits, None)
        self.assertTrue(torch.allclose(result_none, logits))

    def test_calibration_registry(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Prepare dummy calibration data
            dummy_data = {"eyelid": 1.2, "anterior": 0.8}
            with open(tmp_path, "w") as f:
                json.dump(dummy_data, f)
                
            registry = CalibrationRegistry(tmp_path)
            
            # Check loaded temperatures
            self.assertEqual(registry.get("eyelid"), 1.2)
            self.assertEqual(registry.get("anterior"), 0.8)
            self.assertEqual(registry.get("surface"), DEFAULT_TEMPERATURE)  # Fallback
            
            # Check is_calibrated
            self.assertTrue(registry.is_calibrated("eyelid"))
            self.assertFalse(registry.is_calibrated("surface"))
            
            # Check all
            self.assertEqual(registry.all(), dummy_data)
            
            # Save new values
            new_data = {"eyelid": 1.5, "surface": 1.1}
            CalibrationRegistry.save(tmp_path, new_data)
            registry.reload()
            
            self.assertEqual(registry.get("eyelid"), 1.5)
            self.assertEqual(registry.get("surface"), 1.1)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()

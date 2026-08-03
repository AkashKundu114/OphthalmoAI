import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.iqa import assess_image_quality, _check_brightness

class TestImageQualityAssessment(unittest.TestCase):

    def _make_image(self, brightness=128, size=(380, 380)):
        arr = np.full((*size, 3), brightness, dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def test_acceptable_image(self):
        img = self._make_image(120)
        with patch('cv2.HoughCircles', return_value=np.array([[[190, 190, 50]]])), \
             patch('cv2.Laplacian', return_value=MagicMock(var=lambda: 150.0)):
            acceptable, issues = assess_image_quality(img)
            self.assertTrue(acceptable)
            self.assertEqual(len(issues), 0)

    def test_dark_image(self):
        img = self._make_image(10)
        with patch('cv2.HoughCircles', return_value=np.array([[[190, 190, 50]]])), \
             patch('cv2.Laplacian', return_value=MagicMock(var=lambda: 150.0)):
            acceptable, issues = assess_image_quality(img)
            self.assertFalse(acceptable)
            self.assertTrue(any("too dark" in i for i in issues))

    def test_overexposed_image(self):
        img = self._make_image(245)
        with patch('cv2.HoughCircles', return_value=np.array([[[190, 190, 50]]])), \
             patch('cv2.Laplacian', return_value=MagicMock(var=lambda: 150.0)):
            acceptable, issues = assess_image_quality(img)
            self.assertFalse(acceptable)
            self.assertTrue(any("overexposed" in i for i in issues))

    def test_blurry_image(self):
        img = self._make_image(120)
        with patch('cv2.HoughCircles', return_value=np.array([[[190, 190, 50]]])), \
             patch('cv2.Laplacian', return_value=MagicMock(var=lambda: 20.0)):
            acceptable, issues = assess_image_quality(img)
            self.assertFalse(acceptable)
            self.assertTrue(any("blurry" in i for i in issues))

    def test_no_iris_detected(self):
        img = self._make_image(120)
        with patch('cv2.HoughCircles', return_value=None), \
             patch('cv2.Laplacian', return_value=MagicMock(var=lambda: 150.0)):
            acceptable, issues = assess_image_quality(img)
            self.assertFalse(acceptable)
            self.assertTrue(any("iris/pupil" in i for i in issues))

    def test_numpy_fallback_on_laplacian_exception(self):
        img = self._make_image(120)
        with patch('cv2.HoughCircles', return_value=np.array([[[190, 190, 50]]])), \
             patch('cv2.Laplacian', side_effect=Exception("OpenCV Error")):
            acceptable, issues = assess_image_quality(img)
            self.assertIsInstance(acceptable, bool)

    def test_opencv_not_installed_behavior(self):
        img = self._make_image(120)
        with patch('backend.iqa.CV2_AVAILABLE', False):
            acceptable, issues = assess_image_quality(img)
            self.assertFalse(acceptable)
            self.assertTrue(any("OpenCV not installed" in i for i in issues))

    def test_check_brightness_helper(self):
        issues = []
        _check_brightness(15.0, issues)
        self.assertEqual(len(issues), 1)
        self.assertIn("too dark", issues[0])

        issues = []
        _check_brightness(245.0, issues)
        self.assertEqual(len(issues), 1)
        self.assertIn("overexposed", issues[0])

        issues = []
        _check_brightness(120.0, issues)
        self.assertEqual(len(issues), 0)

if __name__ == "__main__":
    unittest.main()

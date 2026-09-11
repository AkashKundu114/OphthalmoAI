from __future__ import annotations
import unittest
import numpy as np
from PIL import Image
from backend.biomarker_extractor import extract_visual_biomarkers, format_biomarkers_for_llm


class TestBiomarkerExtractor(unittest.TestCase):

    def test_extract_biomarkers_with_valid_image_and_gradcam(self):
        # Create a simple 100x100 RGB image
        img = Image.new("RGB", (100, 100), color=(200, 150, 100))
        # Create gradcam with a central hotspot
        gradcam = np.zeros((100, 100), dtype=np.float32)
        gradcam[40:60, 40:60] = 1.0

        biomarkers = extract_visual_biomarkers(img, gradcam, "Cataract")
        self.assertIn("corneal_involvement_pct", biomarkers)
        self.assertIn("vascular_erythema_index", biomarkers)
        self.assertIn("scleral_icterus_index", biomarkers)
        self.assertIn("saliency_focus_profile", biomarkers)
        self.assertIn("active_lesion_area_pct", biomarkers)
        self.assertIsInstance(biomarkers["active_lesion_area_pct"], float)

    def test_extract_biomarkers_with_none_gradcam(self):
        img = Image.new("RGB", (50, 50), color=(128, 128, 128))
        biomarkers = extract_visual_biomarkers(img, None, "Normal")
        self.assertEqual(biomarkers["active_lesion_area_pct"], 0.0)
        self.assertEqual(biomarkers["corneal_involvement_pct"], 0.0)

    def test_extract_biomarkers_resizes_mismatched_gradcam(self):
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        gradcam_small = np.ones((50, 50), dtype=np.float32)
        biomarkers = extract_visual_biomarkers(img, gradcam_small, "Conjunctivitis")
        self.assertIsNotNone(biomarkers["saliency_focus_profile"])

    def test_format_biomarkers_for_llm(self):
        biomarkers = {
            "corneal_involvement_pct": 25.5,
            "vascular_erythema_index": 3.2,
            "scleral_icterus_index": 0.4,
            "saliency_focus_profile": "Focal / Well-Circumscribed",
            "active_lesion_area_pct": 12.0,
        }
        prompt = format_biomarkers_for_llm(
            biomarkers=biomarkers,
            diagnosis="Keratitis",
            conformal_set=["Keratitis", "Corneal Ulcer"],
            coverage_guarantee=95.0,
            epistemic_vacuity=0.125,
        )
        self.assertIn("Keratitis", prompt)
        self.assertIn("25.5%", prompt)
        self.assertIn("Focal / Well-Circumscribed", prompt)
        self.assertIn("95.0%", prompt)


if __name__ == "__main__":
    unittest.main()

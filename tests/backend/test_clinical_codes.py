import os
import sys
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.clinical_codes import get_clinical_code, is_critical, sort_by_urgency

class TestClinicalCodeMappings(unittest.TestCase):

    def test_cataract_mappings(self):
        code = get_clinical_code("Cataract")
        self.assertEqual(code["icd10"], "H26.9")
        self.assertEqual(code["snomed_ct"], "193570009")
        self.assertEqual(code["urgency"], "elective")
        self.assertEqual(code["urgency_rank"], 1)

    def test_uveitis_mappings_critical(self):
        code = get_clinical_code("Uveitis")
        self.assertEqual(code["icd10"], "H20.9")
        self.assertEqual(code["snomed_ct"], "128473001")
        self.assertEqual(code["urgency"], "urgent")
        self.assertEqual(code["urgency_rank"], 3)
        self.assertTrue(is_critical("Uveitis"))
        self.assertIsNotNone(code["escalation_message"])

    def test_jaundice_mappings_emergency(self):
        code = get_clinical_code("Jaundice")
        self.assertEqual(code["icd10"], "R17")
        self.assertEqual(code["snomed_ct"], "65142007")
        self.assertEqual(code["urgency"], "emergency")
        self.assertEqual(code["urgency_rank"], 4)
        self.assertTrue(is_critical("Jaundice"))
        self.assertIsNotNone(code["escalation_message"])

    def test_normal_mappings(self):
        code = get_clinical_code("Normal")
        self.assertEqual(code["icd10"], "Z01.00")
        self.assertEqual(code["snomed_ct"], "165070006")
        self.assertEqual(code["urgency"], "none")
        self.assertEqual(code["urgency_rank"], 0)
        self.assertFalse(is_critical("Normal"))

    def test_unrecognized_diagnosis_fallback(self):
        code = get_clinical_code("Glaucoma")  # Not in registry yet
        self.assertEqual(code["icd10"], "Z01.00")
        self.assertEqual(code["urgency_rank"], 2)
        self.assertIn("not recognised by clinical code table", code["referral"])

    def test_sort_by_urgency(self):
        diagnoses = ["Normal", "Uveitis", "Cataract", "Jaundice"]
        # urgency ranks: Jaundice (4), Uveitis (3), Cataract (1), Normal (0)
        sorted_list = sort_by_urgency(diagnoses)
        self.assertEqual(sorted_list, ["Jaundice", "Uveitis", "Cataract", "Normal"])

if __name__ == "__main__":
    unittest.main()

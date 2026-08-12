import os
import sys
import io
import unittest
from fastapi.testclient import TestClient
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import backend.main as bm
from backend.db import create_tables

class TestStartupInnovationSuite(unittest.TestCase):

    def setUp(self):
        create_tables()
        self.client = TestClient(bm.app)

    def test_multimodal_clinical_fusion(self):
        payload = {
            "hba1c": 8.0,
            "systolic_bp": 140,
            "diastolic_bp": 90,
            "age": 62,
            "is_smoker": True,
            "diagnosis": "Uveitis",
            "confidence": 85.0
        }
        res = self.client.post("/predict/multimodal", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["cardiovascular_risk_level"], "High")
        self.assertEqual(data["ophthalmic_progression_risk_level"], "High")
        self.assertIn("cardiovascular_risk_score", data)

    def test_federated_learning_sync(self):
        res = self.client.post("/admin/federated/sync")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("global_round", data)
        self.assertIn("aggregated_accuracy", data)
        self.assertEqual(data["status"], "success")

    def test_synthetic_case_generator(self):
        res = self.client.post("/admin/synthetic/generate?condition=Cataract&severity=severe")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["condition"], "Cataract")
        self.assertIn("synthetic_image_url", data)
        self.assertIn("synthetic_gradcam_url", data)

    def test_low_bandwidth_compression(self):
        img = Image.new("RGB", (512, 512), color="red")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
        res = self.client.post("/scans/compress", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("original_size_bytes", data)
        self.assertIn("compressed_size_bytes", data)
        self.assertIn("bandwidth_saved_percent", data)

    def test_dicom_pacs_import(self):
        res = self.client.post("/admin/pacs-import")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "imported")
        self.assertIn("dicom_metadata", data)

    def test_triage_queue_and_sign_off(self):
        res_import = self.client.post("/admin/pacs-import")
        scan_id = res_import.json()["scan_id"]

        from backend.db import SessionLocal, ScanResult
        session = SessionLocal()
        scan = session.query(ScanResult).filter(ScanResult.id == scan_id).first()
        scan.sign_off_status = "pending"
        session.commit()
        session.close()

        res_queue = self.client.get("/admin/triage-queue")
        self.assertEqual(res_queue.status_code, 200)
        queue_data = res_queue.json()
        self.assertGreaterEqual(queue_data["queue_length"], 1)

        res_signoff = self.client.post(f"/scans/{scan_id}/sign-off", data={"verified_diagnosis": "Uveitis", "notes": "Confirmed uveitis"})
        self.assertEqual(res_signoff.status_code, 200)
        signoff_data = res_signoff.json()
        self.assertEqual(signoff_data["status"], "overridden")
        self.assertTrue(signoff_data["sync_successful"])


        res_details = self.client.get(f"/scans/{scan_id}/details")
        self.assertEqual(res_details.status_code, 200)
        details_data = res_details.json()
        self.assertEqual(details_data["scan_id"], scan_id)
        self.assertIn("probabilities", details_data)
        self.assertIn("symptoms_reported", details_data)

if __name__ == "__main__":
    unittest.main()

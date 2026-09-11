from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch
import backend.storage as storage


class TestStorage(unittest.TestCase):

    def setUp(self):
        storage._client = None

    def tearDown(self):
        storage._client = None

    def test_is_configured_false_when_no_bucket(self):
        with patch.object(storage, "BUCKET", ""):
            self.assertFalse(storage.is_configured())

    def test_is_configured_true_when_bucket_present(self):
        with patch.object(storage, "BUCKET", "test-bucket"):
            self.assertTrue(storage.is_configured())

    def test_store_returns_none_when_unconfigured(self):
        with patch.object(storage, "BUCKET", ""):
            self.assertIsNone(storage.store(b"data", "test-key"))

    def test_store_success(self):
        mock_client = MagicMock()
        with patch.object(storage, "BUCKET", "test-bucket"), \
             patch.object(storage, "KMS_KEY_ID", "key-123"), \
             patch.object(storage, "_get_client", return_value=mock_client):
            result = storage.store(b"data", "scan-1.jpg", content_type="image/jpeg")
            self.assertEqual(result, "scan-1.jpg")
            mock_client.put_object.assert_called_once_with(
                Bucket="test-bucket",
                Key="scan-1.jpg",
                Body=b"data",
                ContentType="image/jpeg",
                ServerSideEncryption="aws:kms",
                SSEKMSKeyId="key-123",
            )

    def test_store_handles_exception(self):
        mock_client = MagicMock()
        mock_client.put_object.side_effect = RuntimeError("S3 Put error")
        with patch.object(storage, "BUCKET", "test-bucket"), \
             patch.object(storage, "_get_client", return_value=mock_client):
            result = storage.store(b"data", "scan-1.jpg")
            self.assertIsNone(result)

    def test_store_when_client_is_none(self):
        with patch.object(storage, "BUCKET", "test-bucket"), \
             patch.object(storage, "_get_client", return_value=None):
            self.assertIsNone(storage.store(b"data", "scan-1.jpg"))

    def test_fetch_returns_none_when_unconfigured_or_empty_key(self):
        with patch.object(storage, "BUCKET", ""):
            self.assertIsNone(storage.fetch("key"))
        with patch.object(storage, "BUCKET", "test-bucket"):
            self.assertIsNone(storage.fetch(""))

    def test_fetch_success(self):
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"retrieved-bytes"
        mock_client.get_object.return_value = {"Body": mock_body}
        with patch.object(storage, "BUCKET", "test-bucket"), \
             patch.object(storage, "_get_client", return_value=mock_client):
            data = storage.fetch("scan-1.jpg")
            self.assertEqual(data, b"retrieved-bytes")

    def test_fetch_handles_exception(self):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = RuntimeError("S3 Get error")
        with patch.object(storage, "BUCKET", "test-bucket"), \
             patch.object(storage, "_get_client", return_value=mock_client):
            self.assertIsNone(storage.fetch("scan-1.jpg"))

    def test_fetch_when_client_is_none(self):
        with patch.object(storage, "BUCKET", "test-bucket"), \
             patch.object(storage, "_get_client", return_value=None):
            self.assertIsNone(storage.fetch("scan-1.jpg"))

    def test_presigned_url_returns_none_when_unconfigured_or_empty_key(self):
        with patch.object(storage, "BUCKET", ""):
            self.assertIsNone(storage.presigned_url("key"))
        with patch.object(storage, "BUCKET", "test-bucket"):
            self.assertIsNone(storage.presigned_url(""))

    def test_presigned_url_success(self):
        mock_client = MagicMock()
        mock_client.generate_presigned_url.return_value = "https://s3.amazonaws.com/presigned"
        with patch.object(storage, "BUCKET", "test-bucket"), \
             patch.object(storage, "_get_client", return_value=mock_client):
            url = storage.presigned_url("scan-1.jpg", expires_seconds=600)
            self.assertEqual(url, "https://s3.amazonaws.com/presigned")
            mock_client.generate_presigned_url.assert_called_once_with(
                "get_object",
                Params={"Bucket": "test-bucket", "Key": "scan-1.jpg"},
                ExpiresIn=600,
            )

    def test_presigned_url_handles_exception(self):
        mock_client = MagicMock()
        mock_client.generate_presigned_url.side_effect = RuntimeError("Presign error")
        with patch.object(storage, "BUCKET", "test-bucket"), \
             patch.object(storage, "_get_client", return_value=mock_client):
            self.assertIsNone(storage.presigned_url("scan-1.jpg"))

    def test_presigned_url_when_client_is_none(self):
        with patch.object(storage, "BUCKET", "test-bucket"), \
             patch.object(storage, "_get_client", return_value=None):
            self.assertIsNone(storage.presigned_url("scan-1.jpg"))


if __name__ == "__main__":
    unittest.main()

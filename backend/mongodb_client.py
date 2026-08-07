from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional

try:
    import pymongo
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

class OphthalmoMongoDocumentStore:
    def __init__(self):
        self.url = os.getenv("MONGODB_URL", "").strip()
        self.use_real_mongo = MONGO_AVAILABLE and bool(self.url)
        self.db = None
        self.collection = None
        
        if self.use_real_mongo:
            try:
                self.client = pymongo.MongoClient(self.url, serverSelectionTimeoutMS=2000)
                self.client.server_info()
                self.db = self.client.get_database("ophthalmoai")
                self.collection = self.db.get_collection("scan_details")
            except Exception as e:
                print(f"Failed to connect to real MongoDB at {self.url}, falling back to Mock File Store: {e}")
                self.use_real_mongo = False
                
        if not self.use_real_mongo:
            self.mock_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mongodb_mock")
            os.makedirs(self.mock_dir, exist_ok=True)

    def insert_document(self, doc_id: str, document: Dict[str, Any]) -> str:
        """
        Inserts a diagnostic document. Maps to a MongoDB insert or Mock File write.
        """
        payload = {"_id": doc_id, **document}
        
        if self.use_real_mongo and self.collection is not None:
            self.collection.replace_one({"_id": doc_id}, payload, upsert=True)
        else:
            file_path = os.path.join(self.mock_dir, f"{doc_id}.json")
            with open(file_path, "w") as f:
                json.dump(payload, f, indent=2)
                
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a document by its ID.
        """
        if self.use_real_mongo and self.collection is not None:
            return self.collection.find_one({"_id": doc_id})
        else:
            file_path = os.path.join(self.mock_dir, f"{doc_id}.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    return json.load(f)
            return None

    def delete_document(self, doc_id: str) -> bool:
        """
        Deletes a document by ID.
        """
        if self.use_real_mongo and self.collection is not None:
            res = self.collection.delete_one({"_id": doc_id})
            return res.deleted_count > 0
        else:
            file_path = os.path.join(self.mock_dir, f"{doc_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False

# Singleton instance
mongo_store = OphthalmoMongoDocumentStore()

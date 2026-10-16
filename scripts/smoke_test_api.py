import io
import json
import os
import sys
import uuid
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from backend.main import app

results = {}

print("--- Running API Smoke Test Suite with Lifespan ---")

with TestClient(app) as client:
    # 1. Health
    resp = client.get("/health")
    results["GET /health"] = {"status": resp.status_code, "body": resp.json()}
    print(f"GET /health: {resp.status_code}")

    # 2. Ready
    resp = client.get("/ready")
    results["GET /ready"] = {"status": resp.status_code, "body": resp.json()}
    print(f"GET /ready: {resp.status_code} (model_loaded: {resp.json().get('model_loaded')})")

    # 3. Conditions
    resp = client.get("/conditions")
    results["GET /conditions"] = {"status": resp.status_code, "count": len(resp.json()) if resp.status_code == 200 else 0}
    print(f"GET /conditions: {resp.status_code} (count: {results['GET /conditions']['count']})")

    # 4. Auth flow: register & login
    unique_id = uuid.uuid4().hex[:6]
    test_email = f"smoke_clinician_{unique_id}@example.com"
    test_pwd = "My$tr0ngP@ssw0rd!2026"
    reg_payload = {"email": test_email, "password": test_pwd, "full_name": "Dr Smoke Tester"}
    resp = client.post("/auth/register", json=reg_payload)
    print(f"POST /auth/register: {resp.status_code}")
    results["POST /auth/register"] = {"status": resp.status_code}

    # Form data for /auth/token
    resp = client.post("/auth/token", data={"email": test_email, "password": test_pwd})
    token = ""
    if resp.status_code == 200:
        token = resp.json().get("access_token", "")
        results["POST /auth/token"] = {"status": resp.status_code, "token_received": bool(token)}
        print(f"POST /auth/token: {resp.status_code} (token received: {bool(token)})")
    else:
        results["POST /auth/token"] = {"status": resp.status_code, "error": resp.text}
        print(f"POST /auth/token: {resp.status_code} {resp.text}")

    auth_headers = {"Authorization": f"Bearer {token}"} if token else {}

    # 5. /auth/me
    resp = client.get("/auth/me", headers=auth_headers)
    results["GET /auth/me"] = {"status": resp.status_code, "email": resp.json().get("email") if resp.status_code == 200 else None}
    print(f"GET /auth/me: {resp.status_code} (user: {results['GET /auth/me']['email']})")

    # 6. /predict with authentic fundus sample image
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "public", "samples", "fundus_normal.jpg")
    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("fundus_normal.jpg", img_bytes, "image/jpeg")}
    data = {
        "pain": "None",
        "vision": "No",
        "itch": "No",
        "halos": "No",
        "discharge": "None",
        "light_sens": "No",
        "floaters": "No",
        "duration": "Not Sure",
    }
    resp = client.post("/predict", files=files, data=data)
    results["POST /predict"] = {
        "status": resp.status_code,
        "diagnosis": resp.json().get("diagnosis") if resp.status_code == 200 else None,
        "confidence": resp.json().get("confidence") if resp.status_code == 200 else None,
    }
    print(f"POST /predict: {resp.status_code} -> Diagnosis: {results['POST /predict']['diagnosis']}")

    # 7. /chat
    chat_payload = {"message": "What is diabetic retinopathy?"}
    resp = client.post("/chat", json=chat_payload)
    results["POST /chat"] = {"status": resp.status_code}
    print(f"POST /chat: {resp.status_code}")

    # 8. Clinician cases
    resp = client.get("/clinician/cases", headers=auth_headers)
    results["GET /clinician/cases"] = {"status": resp.status_code, "count": len(resp.json()) if resp.status_code == 200 else 0}
    print(f"GET /clinician/cases: {resp.status_code}")

print("\n--- Smoke Summary ---")
print(json.dumps(results, indent=2))

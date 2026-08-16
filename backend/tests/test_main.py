def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_check(client):
    response = client.get("/ready")
    # It might return 200 or 503 depending on if the model is loaded in the test env
    assert response.status_code in [200, 503]

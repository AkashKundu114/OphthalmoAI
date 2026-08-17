def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True

def test_ready_check(client):
    response = client.get("/ready")
    assert response.status_code in [200, 503]

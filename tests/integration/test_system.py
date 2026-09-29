from fastapi.testclient import TestClient

from insurance_copilot.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_endpoint() -> None:
    response = client.get("/ready")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ready"
    assert body["checks"]["database"] is True
    assert body["checks"]["guidance_index"] is True

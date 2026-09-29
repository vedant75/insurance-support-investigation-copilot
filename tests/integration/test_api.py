from fastapi.testclient import TestClient

from insurance_copilot.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_complaint_and_guidance() -> None:
    response = client.post(
        "/analyze",
        json={
            "question": (
                "Show the recorded details for "
                "complaint 467758 and retrieve "
                "relevant TDI guidance about "
                "total loss disputes."
            ),
            "complaint_number": "467758",
            "guidance_top_k": 3,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["workflow"] == "deterministic"

    assert "get_complaint" in body["tools_used"]

    assert "search_insurance_guidance" in body["tools_used"]

    evidence_ids = {item["evidence_id"] for item in body["report"]["evidence"]}

    assert "SQL-COMPLAINT-467758" in evidence_ids

from uuid import uuid4

import insurance_copilot.workflows.graph_nodes as graph_nodes
from insurance_copilot.domain.models import (
    GraphAnalyzeRequest,
    GuidanceSearchResult,
    WorkflowStatus,
)
from insurance_copilot.services.graph_service import (
    run_graph_analysis,
)


def test_missing_complaint_is_reported() -> None:
    response = run_graph_analysis(
        GraphAnalyzeRequest(
            question=("Show the recorded details for complaint 9999999."),
            complaint_number="9999999",
            thread_id=str(uuid4()),
        )
    )

    assert response.status == WorkflowStatus.COMPLETED

    assert response.report is not None

    unresolved = " ".join(response.report.unresolved_questions)

    assert "No complaint record was found" in unresolved


def test_database_tool_failure_requests_review(
    monkeypatch,
) -> None:
    def injected_failure(
        complaint_number: str,
    ):
        raise RuntimeError("Injected database failure")

    monkeypatch.setattr(
        graph_nodes,
        "get_complaint",
        injected_failure,
    )

    response = run_graph_analysis(
        GraphAnalyzeRequest(
            question=("Show complaint 467758."),
            complaint_number="467758",
            thread_id=str(uuid4()),
        )
    )

    assert response.status == WorkflowStatus.PAUSED_FOR_REVIEW

    assert len(response.tool_failures) == 1

    assert response.tool_failures[0].tool_name == "get_complaint"

    assert response.tool_failures[0].error_type == "RuntimeError"

    assert response.review_request is not None

    assert "tool" in response.review_request.reason.lower()


def test_empty_retrieval_requests_review(
    monkeypatch,
) -> None:
    def empty_guidance(
        query: str,
        top_k: int = 5,
    ) -> GuidanceSearchResult:
        return GuidanceSearchResult(
            query=query,
            total_hits=0,
            hits=[],
        )

    monkeypatch.setattr(
        graph_nodes,
        "search_insurance_guidance",
        empty_guidance,
    )

    response = run_graph_analysis(
        GraphAnalyzeRequest(
            question=("What does TDI say about collision coverage?"),
            thread_id=str(uuid4()),
        )
    )

    assert response.status == WorkflowStatus.PAUSED_FOR_REVIEW

    assert response.review_request is not None

    assert "insufficient" in response.review_request.reason.lower()

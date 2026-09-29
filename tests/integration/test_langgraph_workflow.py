from uuid import uuid4

from insurance_copilot.domain.models import (
    GraphAnalyzeRequest,
    HumanReviewDecision,
    ReviewDecision,
    WorkflowStatus,
)
from insurance_copilot.services.graph_service import (
    get_graph_state,
    resume_graph_analysis,
    run_graph_analysis,
)


def test_graph_sql_only_routing() -> None:
    response = run_graph_analysis(
        GraphAnalyzeRequest(
            question=("What are the most common recorded automobile complaint issue tags?"),
            thread_id=str(uuid4()),
        )
    )

    assert response.status == WorkflowStatus.COMPLETED

    assert response.tools_used == ["get_complaint_statistics"]


def test_graph_pause_and_resume() -> None:
    thread_id = str(uuid4())

    paused = run_graph_analysis(
        GraphAnalyzeRequest(
            question=(
                "Show the recorded details "
                "for complaint 467758 and "
                "retrieve relevant TDI "
                "guidance about total loss "
                "disputes."
            ),
            complaint_number="467758",
            require_human_review=True,
            thread_id=thread_id,
        )
    )

    assert paused.status == WorkflowStatus.PAUSED_FOR_REVIEW

    assert paused.review_request is not None

    snapshot = get_graph_state(thread_id)

    assert len(snapshot["pending_interrupts"]) == 1

    resumed = resume_graph_analysis(
        thread_id,
        HumanReviewDecision(
            decision=(ReviewDecision.APPROVE),
            reviewer_note=("Evidence checked."),
        ),
    )

    assert resumed.status == WorkflowStatus.COMPLETED

    assert resumed.report is not None

    assert resumed.review_decision is not None


def test_human_can_edit_summary() -> None:
    thread_id = str(uuid4())

    paused = run_graph_analysis(
        GraphAnalyzeRequest(
            question=("What does TDI say about collision coverage?"),
            require_human_review=True,
            thread_id=thread_id,
        )
    )

    assert paused.status == WorkflowStatus.PAUSED_FOR_REVIEW

    edited_summary = "Human-reviewed summary for collision coverage guidance."

    resumed = resume_graph_analysis(
        thread_id,
        HumanReviewDecision(
            decision=(ReviewDecision.EDIT),
            reviewer_note=("Clarified wording."),
            edited_summary=(edited_summary),
        ),
    )

    assert resumed.report.summary == edited_summary

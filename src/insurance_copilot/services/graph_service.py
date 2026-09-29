from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from langgraph.types import Command

from insurance_copilot.domain.models import (
    ComplaintIntelligenceReport,
    GraphAnalysisResponse,
    GraphAnalyzeRequest,
    HumanReviewDecision,
    ReviewRequest,
    ToolFailure,
    WorkflowStatus,
)
from insurance_copilot.workflows.langgraph_workflow import (
    get_graph_runtime,
)


def _config(
    thread_id: str,
) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def _response_from_state(
    *,
    thread_id: str,
    state: dict,
    latency_ms: float,
    review_request: dict | None = None,
) -> GraphAnalysisResponse:
    status_value = state.get(
        "status",
        "completed",
    )

    if review_request is not None:
        status = (
            WorkflowStatus
            .PAUSED_FOR_REVIEW
        )

    elif status_value == "rejected":
        status = (
            WorkflowStatus.REJECTED
        )

    else:
        status = (
            WorkflowStatus.COMPLETED
        )

    report_data = state.get(
        "report"
    )

    report = (
        ComplaintIntelligenceReport(
            **report_data
        )
        if report_data
        else None
    )

    failures = [
        ToolFailure(
            **failure
        )
        for failure in state.get(
            "tool_failures",
            [],
        )
    ]

    human_review_data = (
        state.get(
            "human_review"
        )
    )

    human_review = (
        HumanReviewDecision(
            **human_review_data
        )
        if human_review_data
        else None
    )

    review_model = (
        ReviewRequest(
            **review_request
        )
        if review_request
        else None
    )

    return GraphAnalysisResponse(
        thread_id=thread_id,
        status=status,
        tools_used=state.get(
            "tools_used",
            [],
        ),
        latency_ms=round(
            latency_ms,
            2,
        ),
        report=report,
        review_request=(
            review_model
        ),
        review_decision=(
            human_review
        ),
        tool_failures=failures,
    )


def run_graph_analysis(
    request: GraphAnalyzeRequest,
) -> GraphAnalysisResponse:
    started = perf_counter()

    runtime = (
        get_graph_runtime()
    )

    thread_id = (
        request.thread_id
        or str(uuid4())
    )

    config = _config(
        thread_id
    )

    initial_state = {
        "question": (
            request.question
        ),
        "complaint_number": (
            request.complaint_number
        ),
        "guidance_top_k": (
            request.guidance_top_k
        ),
        "require_human_review": (
            request
            .require_human_review
        ),
    }

    result = runtime.graph.invoke(
        initial_state,
        config=config,
        durability="sync",
        version="v2",
    )

    latency_ms = (
        perf_counter()
        - started
    ) * 1000

    state = dict(
        result.value
    )

    if result.interrupts:
        review_payload = (
            result.interrupts[0]
            .value
        )

        return _response_from_state(
            thread_id=thread_id,
            state=state,
            latency_ms=latency_ms,
            review_request=(
                review_payload
            ),
        )

    return _response_from_state(
        thread_id=thread_id,
        state=state,
        latency_ms=latency_ms,
    )


def resume_graph_analysis(
    thread_id: str,
    decision: HumanReviewDecision,
) -> GraphAnalysisResponse:
    started = perf_counter()

    runtime = (
        get_graph_runtime()
    )

    config = _config(
        thread_id
    )

    snapshot = (
        runtime.graph.get_state(
            config
        )
    )

    if not snapshot.values:
        raise ValueError(
            "No checkpoint exists for "
            f"thread '{thread_id}'."
        )

    if not snapshot.interrupts:
        raise ValueError(
            "This workflow is not "
            "waiting for human review."
        )

    result = runtime.graph.invoke(
        Command(
            resume=(
                decision.model_dump(
                    mode="json"
                )
            )
        ),
        config=config,
        durability="sync",
        version="v2",
    )

    latency_ms = (
        perf_counter()
        - started
    ) * 1000

    state = dict(
        result.value
    )

    if result.interrupts:
        review_payload = (
            result.interrupts[0]
            .value
        )

        return _response_from_state(
            thread_id=thread_id,
            state=state,
            latency_ms=latency_ms,
            review_request=(
                review_payload
            ),
        )

    return _response_from_state(
        thread_id=thread_id,
        state=state,
        latency_ms=latency_ms,
    )


def get_graph_state(
    thread_id: str,
) -> dict:
    runtime = (
        get_graph_runtime()
    )

    snapshot = (
        runtime.graph.get_state(
            _config(thread_id)
        )
    )

    if not snapshot.values:
        raise ValueError(
            "No checkpoint exists for "
            f"thread '{thread_id}'."
        )

    return {
        "thread_id": thread_id,
        "next_nodes": list(
            snapshot.next
        ),
        "pending_interrupts": [
            {
                "id": interrupt.id,
                "value": interrupt.value,
            }
            for interrupt
            in snapshot.interrupts
        ],
        "state": snapshot.values,
        "created_at": (
            snapshot.created_at
        ),
    }

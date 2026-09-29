from fastapi import (
    APIRouter,
    HTTPException,
)

from insurance_copilot.domain.models import (
    GraphAnalysisResponse,
    GraphAnalyzeRequest,
    HumanReviewDecision,
)
from insurance_copilot.services.graph_service import (
    get_graph_state,
)
from insurance_copilot.services.traced_workflows import (
    resume_graph_analysis,
    run_graph_analysis,
)

router = APIRouter(
    prefix="/graph",
    tags=["langgraph"],
)


@router.post(
    "/analyze",
    response_model=GraphAnalysisResponse,
)
def analyze_with_graph(
    request: GraphAnalyzeRequest,
) -> GraphAnalysisResponse:
    return run_graph_analysis(request)


@router.post(
    "/{thread_id}/resume",
    response_model=GraphAnalysisResponse,
)
def resume_graph(
    thread_id: str,
    decision: HumanReviewDecision,
) -> GraphAnalysisResponse:
    try:
        return resume_graph_analysis(
            thread_id,
            decision,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@router.get(
    "/{thread_id}/state",
)
def graph_state(
    thread_id: str,
) -> dict:
    try:
        return get_graph_state(thread_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

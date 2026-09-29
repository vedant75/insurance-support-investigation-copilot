from fastapi import (
    APIRouter,
    HTTPException,
)

from insurance_copilot.domain.models import (
    AgentAnalysisResponse,
    GraphAnalyzeRequest,
    HumanReviewDecision,
)
from insurance_copilot.services.agent_service import (
    get_agent_state,
)
from insurance_copilot.services.traced_workflows import (
    resume_agent_analysis,
    run_agent_analysis,
)

router = APIRouter(
    prefix="/agent",
    tags=["agentic"],
)


@router.post(
    "/analyze",
    response_model=AgentAnalysisResponse,
)
def analyze_with_agent(
    request: GraphAnalyzeRequest,
) -> AgentAnalysisResponse:
    return run_agent_analysis(request)


@router.post(
    "/{thread_id}/resume",
    response_model=AgentAnalysisResponse,
)
def resume_agent(
    thread_id: str,
    decision: HumanReviewDecision,
) -> AgentAnalysisResponse:
    try:
        return resume_agent_analysis(
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
def agent_state(
    thread_id: str,
) -> dict:
    try:
        return get_agent_state(thread_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

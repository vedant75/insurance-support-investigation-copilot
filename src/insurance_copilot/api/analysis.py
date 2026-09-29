from fastapi import APIRouter

from insurance_copilot.domain.models import (
    AnalysisResponse,
    AnalyzeRequest,
)
from insurance_copilot.services.traced_workflows import (
    run_deterministic_analysis,
)

router = APIRouter(
    prefix="/analyze",
    tags=["analysis"],
)


@router.post(
    "",
    response_model=AnalysisResponse,
)
def analyze(
    request: AnalyzeRequest,
) -> AnalysisResponse:
    return run_deterministic_analysis(request)

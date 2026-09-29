from __future__ import annotations

import mlflow
from mlflow.entities import SpanType

from insurance_copilot.domain.models import (
    AnalysisResponse,
    AnalyzeRequest,
    GraphAnalysisResponse,
    GraphAnalyzeRequest,
    HumanReviewDecision,
)
from insurance_copilot.observability.mlflow_config import (
    configure_mlflow,
)
from insurance_copilot.services.graph_service import (
    resume_graph_analysis as _resume_graph_analysis,
)
from insurance_copilot.services.graph_service import (
    run_graph_analysis as _run_graph_analysis,
)
from insurance_copilot.workflows.deterministic import (
    run_deterministic_analysis as _run_deterministic_analysis,
)

configure_mlflow()


@mlflow.trace(
    name="deterministic_analysis",
    span_type=SpanType.CHAIN,
)
def run_deterministic_analysis(
    request: AnalyzeRequest,
) -> AnalysisResponse:
    return _run_deterministic_analysis(request)


@mlflow.trace(
    name="langgraph_analysis",
    span_type=SpanType.CHAIN,
)
def run_graph_analysis(
    request: GraphAnalyzeRequest,
) -> GraphAnalysisResponse:
    return _run_graph_analysis(request)


@mlflow.trace(
    name="langgraph_resume",
    span_type=SpanType.CHAIN,
)
def resume_graph_analysis(
    thread_id: str,
    decision: HumanReviewDecision,
) -> GraphAnalysisResponse:
    return _resume_graph_analysis(
        thread_id,
        decision,
    )

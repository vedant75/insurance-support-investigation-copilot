from __future__ import annotations

from typing import Any, TypedDict


class InvestigationState(TypedDict, total=False):
    # Input
    question: str
    complaint_number: str | None
    guidance_top_k: int
    require_human_review: bool

    # Routing decisions
    statistics_group: str | None
    guidance_needed: bool

    # Evidence/result accumulation
    evidence: list[dict[str, Any]]
    insights: list[dict[str, Any]]

    limitations: list[str]
    unresolved_questions: list[str]

    tools_used: list[str]
    tool_failures: list[dict[str, str]]

    # Reliability signals
    missing_evidence: bool
    insufficient_retrieval: bool

    # HITL
    requires_review: bool
    review_reason: str | None
    human_review: dict[str, Any] | None

    # Final result
    report: dict[str, Any] | None
    status: str

    # Agent state
    tool_history: list[dict[str, Any]]
    pending_tool_call: dict[str, Any] | None

    agent_iterations: int
    max_agent_iterations: int

    model_input_tokens: int
    model_output_tokens: int
    model_total_tokens: int

    synthesis: dict[str, Any] | None

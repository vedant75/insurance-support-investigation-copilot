from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from insurance_copilot.domain.models import Insight


class AgentDecisionType(str, Enum):
    TOOL_CALL = "tool_call"
    FINALIZE = "finalize"


class AgentToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class AgentToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    action: AgentDecisionType
    tool_call: AgentToolCall | None = None
    decision_summary: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "AgentDecision":
        if self.action == AgentDecisionType.TOOL_CALL and self.tool_call is None:
            raise ValueError("tool_call is required when action='tool_call'")

        if self.action == AgentDecisionType.FINALIZE and self.tool_call is not None:
            raise ValueError("tool_call must be omitted when action='finalize'")

        return self


class ToolTrace(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    succeeded: bool

    evidence_ids: list[str] = Field(default_factory=list)

    error_type: str | None = None
    error_message: str | None = None


class ModelUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    cost_usd: float = 0.0


class AgentModelResult(BaseModel):
    model: str
    decision: AgentDecision
    usage: ModelUsage = Field(default_factory=ModelUsage)

    request_id: str | None = None


class ReportSynthesis(BaseModel):
    summary: str

    insights: list[Insight] = Field(default_factory=list)

    limitations: list[str] = Field(default_factory=list)

    unresolved_questions: list[str] = Field(default_factory=list)


class SynthesisModelResult(BaseModel):
    model: str
    synthesis: ReportSynthesis

    usage: ModelUsage = Field(default_factory=ModelUsage)

    request_id: str | None = None

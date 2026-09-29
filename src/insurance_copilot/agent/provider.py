from __future__ import annotations

from typing import Protocol

from insurance_copilot.agent.models import (
    AgentModelResult,
    AgentToolSpec,
    SynthesisModelResult,
    ToolTrace,
)
from insurance_copilot.domain.models import EvidenceItem


class AgentModelProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def choose_next_action(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        tool_history: list[ToolTrace],
        available_tools: list[AgentToolSpec],
    ) -> AgentModelResult: ...

    def synthesize_report(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        tool_history: list[ToolTrace],
        limitations: list[str],
        unresolved_questions: list[str],
    ) -> SynthesisModelResult: ...

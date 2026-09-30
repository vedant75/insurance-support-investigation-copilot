from __future__ import annotations

import json
import time

from google import genai

from insurance_copilot.agent.models import (
    AgentDecision,
    AgentDecisionType,
    AgentModelResult,
    AgentToolCall,
    AgentToolSpec,
    ModelUsage,
    ReportSynthesis,
    SynthesisModelResult,
    ToolTrace,
)
from insurance_copilot.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)
from insurance_copilot.agent.provider import (
    AgentModelProvider,
)
from insurance_copilot.domain.models import (
    EvidenceItem,
)


class GeminiProvider(AgentModelProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        max_retries: int = 3,
    ) -> None:
        self.client = genai.Client(
            api_key=api_key,
        )

        self._model_name = model_name
        self.max_retries = max_retries

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def _usage(interaction) -> ModelUsage:
        usage = interaction.usage

        if usage is None:
            return ModelUsage()

        return ModelUsage(
            input_tokens=(
                getattr(
                    usage,
                    "total_input_tokens",
                    0,
                )
                or 0
            ),
            output_tokens=(
                getattr(
                    usage,
                    "total_output_tokens",
                    0,
                )
                or 0
            ),
            total_tokens=(
                getattr(
                    usage,
                    "total_tokens",
                    0,
                )
                or 0
            ),
        )

    def _create_interaction(
        self,
        **kwargs,
    ):
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return self.client.interactions.create(**kwargs)

            except Exception as exc:
                last_error = exc

                text = str(exc).casefold()

                transient = "503" in text or "service_unavailable" in text or "high demand" in text

                if not transient:
                    raise

                if attempt + 1 == self.max_retries:
                    break

                time.sleep(2**attempt)

        if last_error is not None:
            raise last_error

        raise RuntimeError("Gemini request failed.")

    @staticmethod
    def _tool_payload(
        tools: list[AgentToolSpec],
    ) -> list[dict]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": (tool.description),
                "parameters": (tool.input_schema),
            }
            for tool in tools
        ]

    def choose_next_action(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        tool_history: list[ToolTrace],
        available_tools: list[AgentToolSpec],
    ) -> AgentModelResult:
        prompt = (
            f"{AGENT_SYSTEM_PROMPT}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "CURRENT EVIDENCE:\n"
            f"{json.dumps([item.model_dump(mode='json') for item in evidence])}\n\n"
            "TOOL HISTORY:\n"
            f"{json.dumps([item.model_dump(mode='json') for item in tool_history])}\n\n"
            "Choose the next required evidence tool. "
            "If no further evidence is necessary, "
            "respond normally without calling a tool."
        )

        interaction = self._create_interaction(
            model=self.model_name,
            input=prompt,
            tools=self._tool_payload(available_tools),
        )

        for step in interaction.steps:
            if step.type != "function_call":
                continue

            return AgentModelResult(
                model=self.model_name,
                decision=AgentDecision(
                    action=(AgentDecisionType.TOOL_CALL),
                    tool_call=AgentToolCall(
                        tool_name=step.name,
                        arguments=dict(step.arguments or {}),
                    ),
                    decision_summary=(f"Gemini requested {step.name}."),
                ),
                usage=self._usage(interaction),
                request_id=interaction.id,
            )

        return AgentModelResult(
            model=self.model_name,
            decision=AgentDecision(
                action=(AgentDecisionType.FINALIZE),
                decision_summary=(interaction.output_text or "Evidence collection complete."),
            ),
            usage=self._usage(interaction),
            request_id=interaction.id,
        )

    def synthesize_report(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        tool_history: list[ToolTrace],
        limitations: list[str],
        unresolved_questions: list[str],
    ) -> SynthesisModelResult:
        prompt = (
            f"{SYNTHESIS_SYSTEM_PROMPT}\n\n"
            f"QUESTION:\n{question}\n\n"
            "EVIDENCE LEDGER:\n"
            f"{json.dumps([item.model_dump(mode='json') for item in evidence])}\n\n"
            "TOOLS EXECUTED:\n"
            f"{json.dumps([item.model_dump(mode='json') for item in tool_history])}\n\n"
            "KNOWN LIMITATIONS:\n"
            f"{json.dumps(limitations)}\n\n"
            "UNRESOLVED QUESTIONS:\n"
            f"{json.dumps(unresolved_questions)}"
        )

        interaction = self._create_interaction(
            model=self.model_name,
            input=prompt,
            response_format=[
                {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": ReportSynthesis.model_json_schema(),
                }
            ],
        )

        synthesis = ReportSynthesis.model_validate_json(interaction.output_text)

        return SynthesisModelResult(
            model=self.model_name,
            synthesis=synthesis,
            usage=self._usage(interaction),
            request_id=interaction.id,
        )

from __future__ import annotations

from typing import Any

from insurance_copilot.agent.evidence import (
    evidence_from_tool_result,
)
from insurance_copilot.agent.models import (
    AgentDecisionType,
    ToolTrace,
)
from insurance_copilot.agent.provider import (
    AgentModelProvider,
)
from insurance_copilot.agent.tool_registry import (
    execute_agent_tool,
    get_agent_tool_specs,
)
from insurance_copilot.domain.models import (
    ComplaintIntelligenceReport,
    EvidenceItem,
    ReviewDecision,
)
from insurance_copilot.workflows.graph_nodes import (
    BASE_LIMITATIONS,
)
from insurance_copilot.workflows.state import (
    InvestigationState,
)


def initialize_agent_node(
    state: InvestigationState,
) -> dict[str, Any]:
    return {
        "evidence": [],
        "insights": [],
        "limitations": (BASE_LIMITATIONS.copy()),
        "unresolved_questions": [],
        "tools_used": [],
        "tool_failures": [],
        "missing_evidence": False,
        "insufficient_retrieval": False,
        "requires_review": False,
        "review_reason": None,
        "human_review": None,
        "report": None,
        "status": "running",
        "tool_history": [],
        "pending_tool_call": None,
        "agent_iterations": 0,
        "max_agent_iterations": 6,
        "model_input_tokens": 0,
        "model_output_tokens": 0,
        "model_total_tokens": 0,
        "synthesis": None,
    }


def make_agent_decide_node(
    provider: AgentModelProvider,
):
    def decide(
        state: InvestigationState,
    ) -> dict[str, Any]:
        if state.get(
            "agent_iterations",
            0,
        ) >= state.get(
            "max_agent_iterations",
            6,
        ):
            return {
                "pending_tool_call": None,
                "unresolved_questions": [
                    *state.get(
                        "unresolved_questions",
                        [],
                    ),
                    ("The agent reached its maximum tool-call limit."),
                ],
            }

        evidence = [
            EvidenceItem.model_validate(item)
            for item in state.get(
                "evidence",
                [],
            )
        ]

        history = [
            ToolTrace.model_validate(item)
            for item in state.get(
                "tool_history",
                [],
            )
        ]

        result = provider.choose_next_action(
            question=state["question"],
            evidence=evidence,
            tool_history=history,
            available_tools=(get_agent_tool_specs()),
        )

        usage = result.usage

        updates = {
            "model_input_tokens": (
                state.get(
                    "model_input_tokens",
                    0,
                )
                + usage.input_tokens
            ),
            "model_output_tokens": (
                state.get(
                    "model_output_tokens",
                    0,
                )
                + usage.output_tokens
            ),
            "model_total_tokens": (
                state.get(
                    "model_total_tokens",
                    0,
                )
                + usage.total_tokens
            ),
        }

        if result.decision.action == AgentDecisionType.FINALIZE:
            return {
                **updates,
                "pending_tool_call": None,
            }

        return {
            **updates,
            "pending_tool_call": (result.decision.tool_call.model_dump(mode="json")),
        }

    return decide


def execute_agent_tool_node(
    state: InvestigationState,
) -> dict[str, Any]:
    call = state.get("pending_tool_call")

    if not call:
        return {}

    tool_name = call["tool_name"]
    arguments = dict(
        call.get(
            "arguments",
            {},
        )
    )

    if tool_name == "search_insurance_guidance":
        requested_top_k = state.get(
            "guidance_top_k",
            3,
        )

        model_top_k = arguments.get(
            "top_k",
            requested_top_k,
        )

        arguments["top_k"] = max(
            1,
            min(
                int(model_top_k),
                int(requested_top_k),
            ),
        )

    history = list(
        state.get(
            "tool_history",
            [],
        )
    )

    tools_used = list(
        state.get(
            "tools_used",
            [],
        )
    )

    failures = list(
        state.get(
            "tool_failures",
            [],
        )
    )

    evidence = list(
        state.get(
            "evidence",
            [],
        )
    )

    try:
        result = execute_agent_tool(
            tool_name,
            arguments,
        )

        new_evidence = evidence_from_tool_result(
            tool_name,
            result,
        )

        existing_ids = {item["evidence_id"] for item in evidence}

        for item in new_evidence:
            if item.evidence_id not in existing_ids:
                evidence.append(item.model_dump(mode="json"))

        history.append(
            ToolTrace(
                tool_name=tool_name,
                arguments=arguments,
                succeeded=True,
                evidence_ids=[item.evidence_id for item in new_evidence],
            ).model_dump(mode="json")
        )

    except Exception as exc:
        failures.append(
            {
                "tool_name": tool_name,
                "error_type": (type(exc).__name__),
                "message": str(exc),
            }
        )

        history.append(
            ToolTrace(
                tool_name=tool_name,
                arguments=arguments,
                succeeded=False,
                error_type=(type(exc).__name__),
                error_message=str(exc),
            ).model_dump(mode="json")
        )

    tools_used.append(tool_name)

    return {
        "pending_tool_call": None,
        "evidence": evidence,
        "tool_history": history,
        "tools_used": tools_used,
        "tool_failures": failures,
        "agent_iterations": (
            state.get(
                "agent_iterations",
                0,
            )
            + 1
        ),
    }


def make_agent_synthesis_node(
    provider: AgentModelProvider,
):
    def synthesize(
        state: InvestigationState,
    ) -> dict[str, Any]:
        evidence = [
            EvidenceItem.model_validate(item)
            for item in state.get(
                "evidence",
                [],
            )
        ]

        history = [
            ToolTrace.model_validate(item)
            for item in state.get(
                "tool_history",
                [],
            )
        ]

        result = provider.synthesize_report(
            question=state["question"],
            evidence=evidence,
            tool_history=history,
            limitations=state.get(
                "limitations",
                [],
            ),
            unresolved_questions=(
                state.get(
                    "unresolved_questions",
                    [],
                )
            ),
        )

        return {
            "synthesis": (result.synthesis.model_dump(mode="json")),
            "model_input_tokens": (
                state.get(
                    "model_input_tokens",
                    0,
                )
                + result.usage.input_tokens
            ),
            "model_output_tokens": (
                state.get(
                    "model_output_tokens",
                    0,
                )
                + result.usage.output_tokens
            ),
            "model_total_tokens": (
                state.get(
                    "model_total_tokens",
                    0,
                )
                + result.usage.total_tokens
            ),
        }

    return synthesize


def assess_agent_node(
    state: InvestigationState,
) -> dict[str, Any]:
    reasons: list[str] = []

    if state.get(
        "require_human_review",
        False,
    ):
        reasons.append("Human review was explicitly requested.")

    if state.get(
        "tool_failures",
        [],
    ):
        reasons.append("One or more evidence tools failed.")

    requires_review = bool(reasons)

    return {
        "requires_review": (requires_review),
        "review_reason": (" ".join(reasons) if reasons else None),
    }


def finalize_agent_node(
    state: InvestigationState,
) -> dict[str, Any]:
    synthesis = state.get("synthesis")

    if synthesis is None:
        raise RuntimeError("Agent synthesis is missing.")

    review = state.get("human_review")

    summary = synthesis["summary"]

    status = "completed"

    if review:
        decision = review.get("decision")

        if decision == ReviewDecision.REJECT.value:
            status = "rejected"

        elif decision == ReviewDecision.EDIT.value and review.get("edited_summary"):
            summary = review["edited_summary"]

    report = ComplaintIntelligenceReport(
        question=state["question"],
        summary=summary,
        insights=(
            synthesis.get(
                "insights",
                [],
            )
        ),
        limitations=(
            synthesis.get(
                "limitations",
                [],
            )
        ),
        unresolved_questions=(
            synthesis.get(
                "unresolved_questions",
                [],
            )
        ),
        evidence=[
            EvidenceItem.model_validate(item)
            for item in state.get(
                "evidence",
                [],
            )
        ],
    )

    return {
        "report": report.model_dump(mode="json"),
        "status": status,
    }

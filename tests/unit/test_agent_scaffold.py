import json

import pytest
from pydantic import ValidationError

from insurance_copilot.agent.models import (
    AgentDecision,
    AgentDecisionType,
    AgentToolCall,
)
from insurance_copilot.agent.tool_registry import (
    get_agent_tool_specs,
    validate_tool_arguments,
)


def test_agent_tool_registry_contains_expected_tools() -> None:
    tools = get_agent_tool_specs()

    names = {tool.name for tool in tools}

    assert names == {
        "get_complaint",
        "search_complaints",
        "get_complaint_statistics",
        "search_insurance_guidance",
    }


def test_agent_tool_schemas_are_json_serializable() -> None:
    tools = get_agent_tool_specs()

    payload = [
        tool.model_dump(
            mode="json",
        )
        for tool in tools
    ]

    json.dumps(payload)


def test_tool_call_decision_requires_tool_call() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(
            action=AgentDecisionType.TOOL_CALL,
        )


def test_finalize_decision_rejects_tool_call() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(
            action=AgentDecisionType.FINALIZE,
            tool_call=AgentToolCall(
                tool_name="get_complaint",
                arguments={
                    "complaint_number": "467758",
                },
            ),
        )


def test_statistics_tool_arguments_validate() -> None:
    validated = validate_tool_arguments(
        "get_complaint_statistics",
        {
            "filters": {
                "finding_type": "Confirmed",
            },
            "group_by": "keyword",
            "limit": 10,
        },
    )

    assert validated.group_by.value == "keyword"
    assert validated.filters.finding_type == "Confirmed"
    assert validated.limit == 10


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown agent tool",
    ):
        validate_tool_arguments(
            "execute_arbitrary_sql",
            {},
        )

from insurance_copilot.workflows import (
    agent_nodes,
)


def test_agent_guidance_respects_request_top_k(
    monkeypatch,
):
    captured: dict = {}

    def fake_execute(
        tool_name,
        arguments,
    ):
        captured["tool_name"] = tool_name

        captured["arguments"] = arguments

        return None

    monkeypatch.setattr(
        agent_nodes,
        "execute_agent_tool",
        fake_execute,
    )

    monkeypatch.setattr(
        agent_nodes,
        "evidence_from_tool_result",
        lambda tool_name, result: [],
    )

    state = {
        "pending_tool_call": {
            "tool_name": ("search_insurance_guidance"),
            "arguments": {
                "query": ("total loss dispute"),
                "top_k": 8,
            },
        },
        "guidance_top_k": 3,
        "evidence": [],
        "tool_history": [],
        "tools_used": [],
        "tool_failures": [],
        "agent_iterations": 0,
    }

    result = agent_nodes.execute_agent_tool_node(state)

    assert captured["arguments"]["top_k"] == 3

    assert result["tool_history"][0]["arguments"]["top_k"] == 3

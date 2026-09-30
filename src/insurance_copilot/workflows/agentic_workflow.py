from __future__ import annotations

import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from insurance_copilot.agent.gemini_provider import (
    GeminiProvider,
)
from insurance_copilot.config import (
    get_settings,
)
from insurance_copilot.workflows.agent_nodes import (
    assess_agent_node,
    execute_agent_tool_node,
    finalize_agent_node,
    initialize_agent_node,
    make_agent_decide_node,
    make_agent_synthesis_node,
)
from insurance_copilot.workflows.graph_nodes import (
    human_review_node,
)
from insurance_copilot.workflows.state import (
    InvestigationState,
)


def route_after_decision(
    state: InvestigationState,
) -> str:
    if state.get("pending_tool_call"):
        return "execute_tool"

    return "synthesize"


def route_after_assessment(
    state: InvestigationState,
) -> str:
    if state.get(
        "requires_review",
        False,
    ):
        return "human_review"

    return "finalize"


def build_agent_graph(
    *,
    checkpointer: SqliteSaver,
    provider: GeminiProvider,
):
    builder = StateGraph(InvestigationState)

    builder.add_node(
        "initialize",
        initialize_agent_node,
    )

    builder.add_node(
        "decide",
        make_agent_decide_node(provider),
    )

    builder.add_node(
        "execute_tool",
        execute_agent_tool_node,
    )

    builder.add_node(
        "synthesize",
        make_agent_synthesis_node(provider),
    )

    builder.add_node(
        "assess",
        assess_agent_node,
    )

    builder.add_node(
        "human_review",
        human_review_node,
    )

    builder.add_node(
        "finalize",
        finalize_agent_node,
    )

    builder.add_edge(
        START,
        "initialize",
    )

    builder.add_edge(
        "initialize",
        "decide",
    )

    builder.add_conditional_edges(
        "decide",
        route_after_decision,
    )

    builder.add_edge(
        "execute_tool",
        "decide",
    )

    builder.add_edge(
        "synthesize",
        "assess",
    )

    builder.add_conditional_edges(
        "assess",
        route_after_assessment,
    )

    builder.add_edge(
        "human_review",
        "finalize",
    )

    builder.add_edge(
        "finalize",
        END,
    )

    return builder.compile(checkpointer=checkpointer)


class AgentGraphRuntime:
    def __init__(self) -> None:
        settings = get_settings()

        if not settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY is not configured.")

        if not settings.llm_model:
            raise RuntimeError("LLM_MODEL is not configured.")

        checkpoint_path = settings.checkpoint_database_path

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            checkpoint_path,
            check_same_thread=False,
        )

        self.checkpointer = SqliteSaver(self.connection)

        self.provider = GeminiProvider(
            api_key=(settings.llm_api_key),
            model_name=(settings.llm_model),
        )

        self.graph = build_agent_graph(
            checkpointer=(self.checkpointer),
            provider=self.provider,
        )


@lru_cache(maxsize=1)
def get_agent_graph_runtime() -> AgentGraphRuntime:
    return AgentGraphRuntime()

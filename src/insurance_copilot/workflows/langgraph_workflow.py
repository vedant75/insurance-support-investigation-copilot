from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from insurance_copilot.config import (
    get_settings,
)
from insurance_copilot.workflows.graph_nodes import (
    assess_evidence_node,
    collect_complaint_node,
    collect_guidance_node,
    collect_statistics_node,
    finalize_node,
    human_review_node,
    initialize_node,
)
from insurance_copilot.workflows.state import (
    InvestigationState,
)


def route_after_initialize(
    state: InvestigationState,
) -> str:
    if state.get(
        "complaint_number"
    ):
        return "collect_complaint"

    if state.get(
        "statistics_group"
    ):
        return "collect_statistics"

    if state.get(
        "guidance_needed"
    ):
        return "collect_guidance"

    return "assess_evidence"


def route_after_complaint(
    state: InvestigationState,
) -> str:
    if state.get(
        "statistics_group"
    ):
        return "collect_statistics"

    if state.get(
        "guidance_needed"
    ):
        return "collect_guidance"

    return "assess_evidence"


def route_after_statistics(
    state: InvestigationState,
) -> str:
    if state.get(
        "guidance_needed"
    ):
        return "collect_guidance"

    return "assess_evidence"


def route_after_assessment(
    state: InvestigationState,
) -> str:
    if state.get(
        "requires_review",
        False,
    ):
        return "human_review"

    return "finalize"


def build_graph(
    checkpointer: SqliteSaver,
):
    builder = StateGraph(
        InvestigationState
    )

    builder.add_node(
        "initialize",
        initialize_node,
    )

    builder.add_node(
        "collect_complaint",
        collect_complaint_node,
    )

    builder.add_node(
        "collect_statistics",
        collect_statistics_node,
    )

    builder.add_node(
        "collect_guidance",
        collect_guidance_node,
    )

    builder.add_node(
        "assess_evidence",
        assess_evidence_node,
    )

    builder.add_node(
        "human_review",
        human_review_node,
    )

    builder.add_node(
        "finalize",
        finalize_node,
    )

    builder.add_edge(
        START,
        "initialize",
    )

    builder.add_conditional_edges(
        "initialize",
        route_after_initialize,
    )

    builder.add_conditional_edges(
        "collect_complaint",
        route_after_complaint,
    )

    builder.add_conditional_edges(
        "collect_statistics",
        route_after_statistics,
    )

    builder.add_edge(
        "collect_guidance",
        "assess_evidence",
    )

    builder.add_conditional_edges(
        "assess_evidence",
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

    return builder.compile(
        checkpointer=checkpointer
    )


class GraphRuntime:
    def __init__(
        self,
        database_path: Path,
    ) -> None:
        database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            database_path,
            check_same_thread=False,
        )

        self.checkpointer = (
            SqliteSaver(
                self.connection
            )
        )

        self.graph = build_graph(
            self.checkpointer
        )


@lru_cache(maxsize=1)
def get_graph_runtime() -> GraphRuntime:
    settings = get_settings()

    return GraphRuntime(
        settings
        .checkpoint_database_path
    )

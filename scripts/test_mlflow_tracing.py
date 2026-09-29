from uuid import uuid4

from insurance_copilot.domain.models import (
    GraphAnalyzeRequest,
)
from insurance_copilot.services.traced_workflows import (
    run_graph_analysis,
)


def main() -> None:
    response = run_graph_analysis(
        GraphAnalyzeRequest(
            question=("What are the most common recorded automobile complaint issue tags?"),
            thread_id=str(uuid4()),
        )
    )

    print(
        "status:",
        response.status.value,
    )

    print(
        "tools:",
        response.tools_used,
    )

    print(
        "latency_ms:",
        response.latency_ms,
    )


if __name__ == "__main__":
    main()

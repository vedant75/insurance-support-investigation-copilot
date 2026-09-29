from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from insurance_copilot.domain.models import (
    AnalyzeRequest,
)
from insurance_copilot.workflows.deterministic import (
    run_deterministic_analysis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES_PATH = PROJECT_ROOT / "data" / "evals" / "deterministic_cases.json"

RESULTS_DIR = PROJECT_ROOT / "data" / "evals" / "results"


def load_cases() -> list[dict]:
    with CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def citation_valid(
    response,
) -> bool:
    available_ids = {item.evidence_id for item in response.report.evidence}

    referenced_ids = {
        evidence_id for insight in response.report.insights for evidence_id in insight.evidence_ids
    }

    return referenced_ids.issubset(available_ids)


def evidence_prefixes_present(
    response,
    prefixes: list[str],
) -> bool:
    evidence_ids = [item.evidence_id for item in response.report.evidence]

    return all(
        any(evidence_id.startswith(prefix) for evidence_id in evidence_ids) for prefix in prefixes
    )


def unresolved_expectation_met(
    response,
    expected_text: str | None,
) -> bool:
    if expected_text is None:
        return True

    combined = " ".join(response.report.unresolved_questions).casefold()

    return expected_text.casefold() in combined


def retrieval_expectation_met(
    response,
    expected_terms: list[str] | None,
) -> bool | None:
    if not expected_terms:
        return None

    guidance_evidence = [
        item for item in response.report.evidence if item.evidence_type.value == "guidance_document"
    ]

    searchable_text = " ".join(
        (
            item.content
            + " "
            + str(
                item.metadata.get(
                    "section",
                    "",
                )
            )
        ).casefold()
        for item in guidance_evidence
    )

    return any(term.casefold() in searchable_text for term in expected_terms)


def main() -> None:
    cases = load_cases()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_results = []

    latencies = []

    for case in cases:
        request = AnalyzeRequest(
            question=case["question"],
            complaint_number=case.get("complaint_number"),
            guidance_top_k=3,
        )

        response = run_deterministic_analysis(request)

        latencies.append(response.latency_ms)

        actual_tools = set(response.tools_used)

        required_tools = set(
            case.get(
                "required_tools",
                [],
            )
        )

        tool_recall_ok = required_tools.issubset(actual_tools)

        exact_tool_selection = actual_tools == required_tools

        prefixes_ok = evidence_prefixes_present(
            response,
            case.get(
                "expected_evidence_prefixes",
                [],
            ),
        )

        unresolved_ok = unresolved_expectation_met(
            response,
            case.get("expected_unresolved_contains"),
        )

        citations_ok = citation_valid(response)

        retrieval_ok = retrieval_expectation_met(
            response,
            case.get("retrieval_any_of"),
        )

        task_complete = all(
            [
                tool_recall_ok,
                prefixes_ok,
                unresolved_ok,
            ]
        )

        case_results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "task_complete": (task_complete),
                "citation_valid": (citations_ok),
                "tool_recall_ok": (tool_recall_ok),
                "exact_tool_selection": (exact_tool_selection),
                "retrieval_success": (retrieval_ok),
                "latency_ms": (response.latency_ms),
                "expected_tools": sorted(required_tools),
                "actual_tools": (response.tools_used),
            }
        )

    total = len(case_results)

    task_completion_rate = sum(result["task_complete"] for result in case_results) / total

    citation_validity_rate = sum(result["citation_valid"] for result in case_results) / total

    tool_recall_rate = sum(result["tool_recall_ok"] for result in case_results) / total

    exact_tool_selection_rate = (
        sum(result["exact_tool_selection"] for result in case_results) / total
    )

    retrieval_cases = [result for result in case_results if result["retrieval_success"] is not None]

    retrieval_success_rate = (
        (sum(result["retrieval_success"] for result in retrieval_cases) / len(retrieval_cases))
        if retrieval_cases
        else None
    )

    summary = {
        "workflow": ("deterministic_no_llm"),
        "evaluated_at_utc": (datetime.now(timezone.utc).isoformat()),
        "case_count": total,
        "task_completion_rate": round(
            task_completion_rate,
            4,
        ),
        "citation_validity_rate": round(
            citation_validity_rate,
            4,
        ),
        "tool_recall_rate": round(
            tool_recall_rate,
            4,
        ),
        "exact_tool_selection_rate": round(
            exact_tool_selection_rate,
            4,
        ),
        "retrieval_success_rate": (
            round(
                retrieval_success_rate,
                4,
            )
            if retrieval_success_rate is not None
            else None
        ),
        "latency_ms": {
            "p50": round(
                float(
                    np.percentile(
                        latencies,
                        50,
                    )
                ),
                2,
            ),
            "p95": round(
                float(
                    np.percentile(
                        latencies,
                        95,
                    )
                ),
                2,
            ),
            "mean": round(
                float(np.mean(latencies)),
                2,
            ),
        },
        "model_calls": 0,
        "model_cost_usd": 0.0,
    }

    results = {
        "summary": summary,
        "cases": case_results,
    }

    output_path = RESULTS_DIR / "deterministic_baseline.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("DETERMINISTIC BASELINE")
    print("=" * 70)

    print(f"Cases:                  {total}")

    print(f"Task completion:        {task_completion_rate:.1%}")

    print(f"Citation validity:      {citation_validity_rate:.1%}")

    print(f"Tool recall:            {tool_recall_rate:.1%}")

    print(f"Exact tool selection:   {exact_tool_selection_rate:.1%}")

    if retrieval_success_rate is not None:
        print(f"Retrieval success:      {retrieval_success_rate:.1%}")

    print(f"p50 latency:            {summary['latency_ms']['p50']} ms")

    print(f"p95 latency:            {summary['latency_ms']['p95']} ms")

    print("Model cost/task:        $0.0000")

    print()
    print(f"Saved → {output_path}")

    print("\nCases with non-exact tool selection:")

    for result in case_results:
        if not result["exact_tool_selection"]:
            print(
                f"  {result['id']}: "
                f"expected="
                f"{result['expected_tools']} "
                f"actual="
                f"{result['actual_tools']}"
            )


if __name__ == "__main__":
    main()

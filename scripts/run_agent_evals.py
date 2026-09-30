from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from insurance_copilot.config import get_settings
from insurance_copilot.domain.models import (
    GraphAnalyzeRequest,
    WorkflowStatus,
)
from insurance_copilot.services.agent_service import (
    run_agent_analysis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES_PATH = PROJECT_ROOT / "data" / "evals" / "agent_cases.json"

RESULTS_DIR = PROJECT_ROOT / "data" / "evals" / "results"

PROGRESS_PATH = RESULTS_DIR / "agent_progress.json"

FINAL_RESULTS_PATH = RESULTS_DIR / "agent_benchmark.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cases() -> list[dict]:
    with CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return {
            "benchmark_model": None,
            "cases": [],
        }

    with PROGRESS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json_atomic(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    temp_path.replace(path)


def citation_valid(
    response,
) -> bool:
    if response.report is None:
        return False

    available_ids = {item.evidence_id for item in response.report.evidence}

    referenced_ids = {
        evidence_id for insight in response.report.insights for evidence_id in insight.evidence_ids
    }

    return referenced_ids.issubset(available_ids)


def evidence_prefixes_present(
    response,
    prefixes: list[str],
) -> bool:
    if not prefixes:
        return True

    if response.report is None:
        return False

    evidence_ids = [item.evidence_id for item in response.report.evidence]

    return all(
        any(evidence_id.startswith(prefix) for evidence_id in evidence_ids) for prefix in prefixes
    )


def response_text(
    response,
) -> str:
    if response.report is None:
        return ""

    report = response.report

    parts = [
        report.summary,
        *[insight.statement for insight in report.insights],
        *report.limitations,
        *report.unresolved_questions,
    ]

    return " ".join(parts).casefold()


def literal_forbidden_claims_absent(
    response,
    forbidden_claims: list[str],
) -> bool | None:
    """
    Literal phrase check only.

    This is intentionally NOT called a semantic
    grounding score. Adversarial semantic cases
    are preserved for manual review.
    """
    if not forbidden_claims:
        return None

    text = response_text(response)

    return all(claim.casefold() not in text for claim in forbidden_claims)


def is_transient_provider_error(
    exc: Exception,
) -> bool:
    text = str(exc).casefold()

    signals = (
        "429",
        "too_many_requests",
        "rate limit",
        "rate_limit",
        "quota",
        "resource_exhausted",
        "503",
        "service_unavailable",
        "high demand",
    )

    return any(signal in text for signal in signals)


def calculate_rate(
    rows: list[dict],
    field: str,
) -> float | None:
    if not rows:
        return None

    return round(
        sum(bool(row[field]) for row in rows) / len(rows),
        4,
    )


def build_summary(
    case_results: list[dict],
) -> dict:
    evaluated = [
        result
        for result in case_results
        if result.get(
            "evaluated",
            False,
        )
    ]

    if not evaluated:
        return {
            "evaluated_cases": 0,
        }

    latencies = [result["latency_ms"] for result in evaluated]

    iterations = [result["agent_iterations"] for result in evaluated]

    forbidden_cases = [
        result
        for result in evaluated
        if (result.get("literal_forbidden_claims_absent") is not None)
    ]

    status_counts = Counter(result["status"] for result in evaluated)

    model_names = sorted({result["model"] for result in evaluated})

    return {
        "workflow": ("langgraph_agentic"),
        "evaluated_at_utc": utc_now(),
        "evaluated_cases": len(evaluated),
        "models": model_names,
        "status_counts": dict(status_counts),
        "task_completion_rate": (
            calculate_rate(
                evaluated,
                "task_complete",
            )
        ),
        "citation_validity_rate": (
            calculate_rate(
                evaluated,
                "citation_valid",
            )
        ),
        "tool_recall_rate": (
            calculate_rate(
                evaluated,
                "tool_recall_ok",
            )
        ),
        "exact_tool_selection_rate": (
            calculate_rate(
                evaluated,
                "exact_tool_selection",
            )
        ),
        "no_duplicate_tool_call_rate": (
            calculate_rate(
                evaluated,
                "no_duplicate_tool_calls",
            )
        ),
        "tool_failure_free_rate": (
            calculate_rate(
                evaluated,
                "tool_failure_free",
            )
        ),
        "literal_forbidden_phrase_absence_rate": (
            calculate_rate(
                forbidden_cases,
                "literal_forbidden_claims_absent",
            )
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
        "agent_iterations": {
            "mean": round(
                float(np.mean(iterations)),
                2,
            ),
            "max": max(iterations),
        },
        "tokens": {
            "total_input": sum(result["model_input_tokens"] for result in evaluated),
            "total_output": sum(result["model_output_tokens"] for result in evaluated),
            "total_model": sum(result["model_total_tokens"] for result in evaluated),
            "mean_model_tokens_per_case": round(
                float(np.mean([result["model_total_tokens"] for result in evaluated])),
                2,
            ),
        },
        "manual_semantic_review_cases": [
            result["id"]
            for result in evaluated
            if result.get(
                "manual_semantic_review_required",
                False,
            )
        ],
    }


def ordered_results(
    cases: list[dict],
    results_by_id: dict[str, dict],
) -> list[dict]:
    return [results_by_id[case["id"]] for case in cases if case["id"] in results_by_id]


def save_progress(
    *,
    benchmark_model: str,
    cases: list[dict],
    results_by_id: dict[str, dict],
    last_provider_error: dict | None,
) -> dict:
    results = ordered_results(
        cases,
        results_by_id,
    )

    payload = {
        "benchmark_model": (benchmark_model),
        "updated_at_utc": utc_now(),
        "last_provider_error": (last_provider_error),
        "summary": build_summary(results),
        "cases": results,
    }

    save_json_atomic(
        PROGRESS_PATH,
        payload,
    )

    return payload


def evaluate_case(
    case: dict,
):
    request = GraphAnalyzeRequest(
        question=case["question"],
        complaint_number=case.get("complaint_number"),
        guidance_top_k=3,
        require_human_review=False,
    )

    return run_agent_analysis(request)


def build_case_result(
    *,
    case: dict,
    response,
) -> dict:
    required_tools = set(
        case.get(
            "required_tools",
            [],
        )
    )

    actual_tools = list(response.tools_used)

    actual_tool_set = set(actual_tools)

    tool_recall_ok = required_tools.issubset(actual_tool_set)

    exact_tool_selection = actual_tool_set == required_tools

    no_duplicate_tool_calls = len(actual_tools) == len(actual_tool_set)

    prefixes_ok = evidence_prefixes_present(
        response,
        case.get(
            "expected_evidence_prefixes",
            [],
        ),
    )

    citations_ok = citation_valid(response)

    tool_failure_free = len(response.tool_failures) == 0

    task_complete = all(
        [
            (response.status == WorkflowStatus.COMPLETED),
            response.report is not None,
            tool_recall_ok,
            prefixes_ok,
            tool_failure_free,
        ]
    )

    expected_semantic_points = case.get(
        "expected_semantic_points",
        [],
    )

    forbidden_claims = case.get(
        "forbidden_claims",
        [],
    )

    return {
        "id": case["id"],
        "question": (case["question"]),
        "evaluated": True,
        "thread_id": (response.thread_id),
        "model": response.model,
        "status": (response.status.value),
        "task_complete": (task_complete),
        "expected_tools": sorted(required_tools),
        "actual_tools": (actual_tools),
        "tool_recall_ok": (tool_recall_ok),
        "exact_tool_selection": (exact_tool_selection),
        "no_duplicate_tool_calls": (no_duplicate_tool_calls),
        "tool_failure_free": (tool_failure_free),
        "evidence_prefixes_ok": (prefixes_ok),
        "citation_valid": (citations_ok),
        "literal_forbidden_claims_absent": (
            literal_forbidden_claims_absent(
                response,
                forbidden_claims,
            )
        ),
        "manual_semantic_review_required": bool(expected_semantic_points),
        "expected_semantic_points": (expected_semantic_points),
        "forbidden_claims": (forbidden_claims),
        "agent_iterations": (response.agent_iterations),
        "latency_ms": (response.latency_ms),
        "model_input_tokens": (response.model_input_tokens),
        "model_output_tokens": (response.model_output_tokens),
        "model_total_tokens": (response.model_total_tokens),
        "tool_failures": [failure.model_dump(mode="json") for failure in response.tool_failures],
        "report": (response.report.model_dump(mode="json") if response.report else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=("Run resumable InsureAssist agent evaluations."))

    parser.add_argument(
        "--max-cases",
        type=int,
        default=2,
        help=("Maximum number of new cases to execute in this run."),
    )

    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help=("Run only specific case IDs. Can be supplied multiple times."),
    )

    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help=("Optional delay between cases."),
    )

    args = parser.parse_args()

    if args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")

    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds must be >= 0")

    settings = get_settings()

    if not settings.llm_model:
        raise RuntimeError("LLM_MODEL is not configured.")

    benchmark_model = settings.llm_model

    cases = load_cases()

    all_case_ids = {case["id"] for case in cases}

    requested_case_ids = set(args.case_id)

    unknown_case_ids = requested_case_ids - all_case_ids

    if unknown_case_ids:
        raise ValueError(f"Unknown case IDs: {sorted(unknown_case_ids)}")

    selected_cases = [
        case for case in cases if (not requested_case_ids or case["id"] in requested_case_ids)
    ]

    progress = load_progress()

    previous_model = progress.get("benchmark_model")

    previous_results = progress.get("cases", [])

    if previous_model and previous_results and previous_model != benchmark_model:
        raise RuntimeError(
            "Existing agent benchmark "
            f"uses model '{previous_model}', "
            "but LLM_MODEL is currently "
            f"'{benchmark_model}'. "
            "Use one model for the complete "
            "benchmark. Change LLM_MODEL back "
            "or delete agent_progress.json "
            "to intentionally start over."
        )

    results_by_id = {
        result["id"]: result
        for result in previous_results
        if result.get(
            "evaluated",
            False,
        )
    }

    new_cases_run = 0

    provider_error = None

    for case in selected_cases:
        case_id = case["id"]

        if case_id in results_by_id:
            print(f"SKIP {case_id} (already evaluated)")
            continue

        if new_cases_run >= args.max_cases:
            break

        print()
        print("=" * 72)
        print(f"RUNNING {case_id}")
        print(case["question"])
        print("=" * 72)

        try:
            response = evaluate_case(case)

        except Exception as exc:
            if is_transient_provider_error(exc):
                provider_error = {
                    "occurred_at_utc": (utc_now()),
                    "error_type": (type(exc).__name__),
                    "message": str(exc),
                }

                save_progress(
                    benchmark_model=(benchmark_model),
                    cases=cases,
                    results_by_id=(results_by_id),
                    last_provider_error=(provider_error),
                )

                print()
                print("Gemini quota/capacity limit reached.")
                print("Completed cases were already saved.")
                print("Run this script again when quota is available.")
                print(f"Error: {exc}")

                break

            raise

        case_result = build_case_result(
            case=case,
            response=response,
        )

        results_by_id[case_id] = case_result

        save_progress(
            benchmark_model=(benchmark_model),
            cases=cases,
            results_by_id=(results_by_id),
            last_provider_error=None,
        )

        new_cases_run += 1

        print(f"Status:      {response.status.value}")

        print(f"Tools:       {response.tools_used}")

        print(f"Iterations:  {response.agent_iterations}")

        print(f"Latency:     {response.latency_ms} ms")

        print(f"Tokens:      {response.model_total_tokens}")

        print(f"Task OK:     {case_result['task_complete']}")

        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    final_payload = save_progress(
        benchmark_model=(benchmark_model),
        cases=cases,
        results_by_id=(results_by_id),
        last_provider_error=(provider_error),
    )

    evaluated_count = len(results_by_id)

    total_count = len(cases)

    print()
    print("=" * 72)
    print("AGENT EVALUATION PROGRESS")
    print("=" * 72)

    print(f"Model:      {benchmark_model}")

    print(f"Completed:  {evaluated_count}/{total_count}")

    print(f"Saved:      {PROGRESS_PATH}")

    if evaluated_count == total_count:
        save_json_atomic(
            FINAL_RESULTS_PATH,
            final_payload,
        )

        print()
        print("All evaluation cases completed.")

        print(f"Final benchmark: {FINAL_RESULTS_PATH}")

    manual_cases = final_payload.get(
        "summary",
        {},
    ).get(
        "manual_semantic_review_cases",
        [],
    )

    if manual_cases:
        print()
        print("Manual semantic review required for:")

        for case_id in manual_cases:
            print(f"  - {case_id}")


if __name__ == "__main__":
    main()

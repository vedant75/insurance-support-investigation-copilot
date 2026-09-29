from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from insurance_copilot.domain.models import (
    AnalyzeRequest,
    ComplaintFilters,
    HumanReviewDecision,
    ReviewDecision,
    StatisticsGroupBy,
)
from insurance_copilot.tools.complaints import (
    get_complaint,
    get_complaint_statistics,
)
from insurance_copilot.tools.guidance import (
    search_insurance_guidance,
)
from insurance_copilot.workflows.deterministic import (
    extract_complaint_number,
    infer_filters,
    infer_statistics_group,
)
from insurance_copilot.workflows.state import (
    InvestigationState,
)


BASE_LIMITATIONS = [
    (
        "The TDI public complaint dataset contains "
        "structured complaint classifications, not "
        "the full complaint narrative or underlying "
        "claim file."
    ),
    (
        "TDI consumer guidance is general regulatory "
        "guidance and does not replace the terms of "
        "an individual insurance policy."
    ),
]


def should_search_guidance(
    question: str,
) -> bool:
    text = question.casefold()

    guidance_signals = [
        "guidance",
        "tdi say",
        "what does tdi",
        "what happens if",
        "how can",
        "relevant tdi",
        "insurance guidance",
        "collision coverage",
        "total loss dispute",
        "claim decision",
        "file an insurance complaint",
        "disagree",
        "appraisal",
    ]

    return any(
        signal in text
        for signal in guidance_signals
    )


def initialize_node(
    state: InvestigationState,
) -> dict[str, Any]:
    request = AnalyzeRequest(
        question=state["question"],
        complaint_number=state.get(
            "complaint_number"
        ),
        guidance_top_k=state.get(
            "guidance_top_k",
            3,
        ),
    )

    complaint_number = (
        extract_complaint_number(
            request
        )
    )

    statistics_group = (
        infer_statistics_group(
            state["question"]
        )
    )

    return {
        "complaint_number": complaint_number,
        "statistics_group": (
            statistics_group.value
            if statistics_group
            else None
        ),
        "guidance_needed": (
            should_search_guidance(
                state["question"]
            )
        ),
        "evidence": [],
        "insights": [],
        "limitations": (
            BASE_LIMITATIONS.copy()
        ),
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
    }


def collect_complaint_node(
    state: InvestigationState,
) -> dict[str, Any]:
    complaint_number = state.get(
        "complaint_number"
    )

    if not complaint_number:
        return {}

    tools_used = [
        *state.get("tools_used", []),
        "get_complaint",
    ]

    evidence = list(
        state.get("evidence", [])
    )

    insights = list(
        state.get("insights", [])
    )

    unresolved = list(
        state.get(
            "unresolved_questions",
            [],
        )
    )

    failures = list(
        state.get(
            "tool_failures",
            [],
        )
    )

    try:
        complaint = get_complaint(
            complaint_number
        )

    except Exception as exc:
        failures.append(
            {
                "tool_name": "get_complaint",
                "error_type": (
                    type(exc).__name__
                ),
                "message": str(exc),
            }
        )

        unresolved.append(
            (
                "The complaint record could "
                "not be retrieved because the "
                "structured-data tool failed."
            )
        )

        return {
            "tools_used": tools_used,
            "tool_failures": failures,
            "unresolved_questions": (
                unresolved
            ),
            "missing_evidence": True,
        }

    if complaint is None:
        unresolved.append(
            (
                "No complaint record was found "
                f"for complaint "
                f"{complaint_number}."
            )
        )

        return {
            "tools_used": tools_used,
            "unresolved_questions": (
                unresolved
            ),
            "missing_evidence": True,
        }

    evidence_id = (
        "SQL-COMPLAINT-"
        f"{complaint.complaint_number}"
    )

    evidence.append(
        {
            "evidence_id": evidence_id,
            "evidence_type": (
                "complaint_record"
            ),
            "source": (
                "TDI complaint database"
            ),
            "content": (
                complaint.model_dump_json()
            ),
            "metadata": {
                "complaint_number": (
                    complaint
                    .complaint_number
                ),
                "received_date": (
                    complaint
                    .received_date
                    .isoformat()
                ),
            },
        }
    )

    keyword_text = (
        ", ".join(
            complaint.keywords
        )
        if complaint.keywords
        else "No keywords recorded"
    )

    insights.extend(
        [
            {
                "statement": (
                    f"Complaint "
                    f"{complaint.complaint_number} "
                    f"is recorded as "
                    f"'{complaint.finding_type}' "
                    "with issue tags: "
                    f"{keyword_text}."
                ),
                "evidence_ids": [
                    evidence_id
                ],
            },
            {
                "statement": (
                    "The complaint was received "
                    f"on "
                    f"{complaint.received_date} "
                    "and closed on "
                    f"{complaint.closed_date}, "
                    "a recorded complaint "
                    "handling interval of "
                    f"{complaint.closure_days} "
                    "days."
                ),
                "evidence_ids": [
                    evidence_id
                ],
            },
        ]
    )

    unresolved.extend(
        [
            (
                "What was stated in the "
                "original complaint narrative?"
            ),
            (
                "What evidence was contained "
                "in the underlying claim file?"
            ),
            (
                "What policy language applied "
                "to the individual complainant?"
            ),
        ]
    )

    return {
        "tools_used": tools_used,
        "evidence": evidence,
        "insights": insights,
        "unresolved_questions": (
            unresolved
        ),
    }


def collect_statistics_node(
    state: InvestigationState,
) -> dict[str, Any]:
    group_value = state.get(
        "statistics_group"
    )

    if not group_value:
        return {}

    tools_used = [
        *state.get("tools_used", []),
        "get_complaint_statistics",
    ]

    failures = list(
        state.get(
            "tool_failures",
            [],
        )
    )

    evidence = list(
        state.get("evidence", [])
    )

    insights = list(
        state.get("insights", [])
    )

    unresolved = list(
        state.get(
            "unresolved_questions",
            [],
        )
    )

    try:
        group_by = (
            StatisticsGroupBy(
                group_value
            )
        )

        filters: ComplaintFilters = (
            infer_filters(
                state["question"]
            )
        )

        statistics = (
            get_complaint_statistics(
                filters=filters,
                group_by=group_by,
                limit=10,
            )
        )

    except Exception as exc:
        failures.append(
            {
                "tool_name": (
                    "get_complaint_statistics"
                ),
                "error_type": (
                    type(exc).__name__
                ),
                "message": str(exc),
            }
        )

        unresolved.append(
            (
                "Complaint statistics could "
                "not be calculated because "
                "the structured-data tool "
                "failed."
            )
        )

        return {
            "tools_used": tools_used,
            "tool_failures": failures,
            "unresolved_questions": (
                unresolved
            ),
            "missing_evidence": True,
        }

    evidence_id = (
        "SQL-STATS-"
        f"{group_by.value.upper()}"
    )

    evidence.append(
        {
            "evidence_id": evidence_id,
            "evidence_type": (
                "sql_aggregation"
            ),
            "source": (
                "TDI complaint database"
            ),
            "content": (
                statistics.model_dump_json()
            ),
            "metadata": {
                "group_by": (
                    group_by.value
                ),
                "matching_complaints": (
                    statistics
                    .total_matching_complaints
                ),
            },
        }
    )

    if statistics.rows:
        top_rows = (
            statistics.rows[:5]
        )

        formatted = "; ".join(
            (
                f"{row.key}: "
                f"{row.count:,}"
            )
            for row in top_rows
        )

        insights.append(
            {
                "statement": (
                    "The leading recorded "
                    f"{group_by.value} "
                    "groups are: "
                    f"{formatted}."
                ),
                "evidence_ids": [
                    evidence_id
                ],
            }
        )
    else:
        unresolved.append(
            (
                "The statistics query returned "
                "no matching complaint groups."
            )
        )

    return {
        "tools_used": tools_used,
        "evidence": evidence,
        "insights": insights,
        "unresolved_questions": (
            unresolved
        ),
    }


def collect_guidance_node(
    state: InvestigationState,
) -> dict[str, Any]:
    tools_used = [
        *state.get("tools_used", []),
        "search_insurance_guidance",
    ]

    evidence = list(
        state.get("evidence", [])
    )

    insights = list(
        state.get("insights", [])
    )

    failures = list(
        state.get(
            "tool_failures",
            [],
        )
    )

    unresolved = list(
        state.get(
            "unresolved_questions",
            [],
        )
    )

    try:
        guidance = (
            search_insurance_guidance(
                query=state["question"],
                top_k=state.get(
                    "guidance_top_k",
                    3,
                ),
            )
        )

    except Exception as exc:
        failures.append(
            {
                "tool_name": (
                    "search_insurance_guidance"
                ),
                "error_type": (
                    type(exc).__name__
                ),
                "message": str(exc),
            }
        )

        unresolved.append(
            (
                "Official TDI guidance could "
                "not be retrieved because the "
                "retrieval tool failed."
            )
        )

        return {
            "tools_used": tools_used,
            "tool_failures": failures,
            "unresolved_questions": (
                unresolved
            ),
            "missing_evidence": True,
            "insufficient_retrieval": True,
        }

    if not guidance.hits:
        unresolved.append(
            (
                "No sufficiently relevant "
                "TDI guidance was retrieved."
            )
        )

        return {
            "tools_used": tools_used,
            "unresolved_questions": (
                unresolved
            ),
            "insufficient_retrieval": True,
        }

    for hit in guidance.hits:
        evidence.append(
            {
                "evidence_id": (
                    hit.chunk_id
                ),
                "evidence_type": (
                    "guidance_document"
                ),
                "source": (
                    hit.document_title
                ),
                "content": hit.text,
                "metadata": {
                    "section": (
                        hit.section_title
                    ),
                    "source_url": (
                        hit.source_url
                    ),
                    "retrieval_score": (
                        hit.score
                    ),
                },
            }
        )

    strongest_hit = (
        guidance.hits[0]
    )

    insights.append(
        {
            "statement": (
                "The most relevant retrieved "
                "TDI guidance section is "
                f"'{strongest_hit.section_title}' "
                "from "
                f"{strongest_hit.document_title}."
            ),
            "evidence_ids": [
                strongest_hit.chunk_id
            ],
        }
    )

    insufficient = (
        strongest_hit.score < 0.05
    )

    if insufficient:
        unresolved.append(
            (
                "The best guidance retrieval "
                "match had low lexical "
                "similarity, so the evidence "
                "may be insufficient."
            )
        )

    return {
        "tools_used": tools_used,
        "evidence": evidence,
        "insights": insights,
        "unresolved_questions": (
            unresolved
        ),
        "insufficient_retrieval": (
            insufficient
        ),
    }


def assess_evidence_node(
    state: InvestigationState,
) -> dict[str, Any]:
    reasons: list[str] = []

    if state.get(
        "require_human_review",
        False,
    ):
        reasons.append(
            "Human review was explicitly "
            "requested."
        )

    if state.get(
        "tool_failures"
    ):
        reasons.append(
            "One or more tools failed."
        )

    if state.get(
        "insufficient_retrieval",
        False,
    ):
        reasons.append(
            "Guidance retrieval was "
            "insufficient."
        )

    requires_review = bool(
        reasons
    )

    return {
        "requires_review": (
            requires_review
        ),
        "review_reason": (
            " ".join(reasons)
            if reasons
            else None
        ),
    }


def human_review_node(
    state: InvestigationState,
) -> dict[str, Any]:
    payload = {
        "reason": (
            state.get(
                "review_reason"
            )
            or "Human review requested."
        ),
        "question": state["question"],
        "evidence_count": len(
            state.get("evidence", [])
        ),
        "tools_used": (
            state.get(
                "tools_used",
                [],
            )
        ),
        "unresolved_questions": (
            state.get(
                "unresolved_questions",
                [],
            )
        ),
        "tool_failures": (
            state.get(
                "tool_failures",
                [],
            )
        ),
    }

    decision = interrupt(
        payload,
        response_schema=(
            HumanReviewDecision
        ),
    )

    if isinstance(
        decision,
        HumanReviewDecision,
    ):
        decision_data = (
            decision.model_dump(
                mode="json"
            )
        )
    else:
        decision_data = dict(
            decision
        )

    return {
        "human_review": (
            decision_data
        )
    }


def finalize_node(
    state: InvestigationState,
) -> dict[str, Any]:
    review = state.get(
        "human_review"
    )

    if (
        review
        and review.get("decision")
        == ReviewDecision.REJECT.value
    ):
        status = "rejected"

    else:
        status = "completed"

    summary_parts: list[str] = []

    if "get_complaint" in state.get(
        "tools_used",
        [],
    ):
        summary_parts.append(
            (
                "The workflow checked the "
                "requested complaint record."
            )
        )

    if (
        "get_complaint_statistics"
        in state.get(
            "tools_used",
            [],
        )
    ):
        summary_parts.append(
            (
                "It calculated complaint "
                "statistics from the structured "
                "TDI dataset."
            )
        )

    if (
        "search_insurance_guidance"
        in state.get(
            "tools_used",
            [],
        )
    ):
        summary_parts.append(
            (
                "It retrieved relevant official "
                "TDI consumer guidance."
            )
        )

    if state.get(
        "tool_failures"
    ):
        summary_parts.append(
            (
                "One or more evidence tools "
                "failed, and those failures "
                "were preserved in the result."
            )
        )

    summary = " ".join(
        summary_parts
    )

    if not summary:
        summary = (
            "The workflow completed without "
            "collecting external evidence."
        )

    if (
        review
        and review.get("decision")
        == ReviewDecision.EDIT.value
        and review.get(
            "edited_summary"
        )
    ):
        summary = review[
            "edited_summary"
        ]

    report = {
        "question": state["question"],
        "summary": summary,
        "insights": (
            state.get(
                "insights",
                [],
            )
        ),
        "limitations": (
            state.get(
                "limitations",
                [],
            )
        ),
        "unresolved_questions": (
            state.get(
                "unresolved_questions",
                [],
            )
        ),
        "evidence": (
            state.get(
                "evidence",
                [],
            )
        ),
    }

    return {
        "report": report,
        "status": status,
    }

from __future__ import annotations

import re
from time import perf_counter

from insurance_copilot.domain.models import (
    AnalysisResponse,
    AnalyzeRequest,
    ComplaintFilters,
    ComplaintIntelligenceReport,
    EvidenceItem,
    EvidenceType,
    Insight,
    StatisticsGroupBy,
)
from insurance_copilot.tools.complaints import (
    get_complaint,
    get_complaint_statistics,
)
from insurance_copilot.tools.guidance import (
    search_insurance_guidance,
)


COMPLAINT_NUMBER_PATTERN = re.compile(
    r"\b\d{5,}\b"
)


def extract_complaint_number(
    request: AnalyzeRequest,
) -> str | None:
    if request.complaint_number:
        return request.complaint_number

    match = (
        COMPLAINT_NUMBER_PATTERN.search(
            request.question
        )
    )

    if match:
        return match.group(0)

    return None


def infer_filters(
    question: str,
) -> ComplaintFilters:
    normalized = question.casefold()

    finding_type = None

    if "not confirmed" in normalized:
        finding_type = "Not Confirmed"

    elif "confirmed" in normalized:
        finding_type = "Confirmed"

    return ComplaintFilters(
        finding_type=finding_type
    )


def infer_statistics_group(
    question: str,
) -> StatisticsGroupBy | None:
    normalized = question.casefold()

    monthly_terms = [
        "monthly",
        "month by month",
        "by month",
        "over time",
        "trend",
        "volume",
    ]

    if any(
        term in normalized
        for term in monthly_terms
    ):
        return StatisticsGroupBy.MONTH

    keyword_terms = [
        "most common",
        "common issue",
        "top issue",
        "frequent issue",
        "keyword",
        "keywords",
        "issue tag",
        "issue tags",
    ]

    if any(
        term in normalized
        for term in keyword_terms
    ):
        return StatisticsGroupBy.KEYWORD

    if "finding type" in normalized:
        return (
            StatisticsGroupBy.FINDING_TYPE
        )

    if "coverage level" in normalized:
        return (
            StatisticsGroupBy.COVERAGE_LEVEL
        )

    if "filed by" in normalized:
        return (
            StatisticsGroupBy.COMPLAINT_FILED_BY
        )

    if "complainant type" in normalized:
        return (
            StatisticsGroupBy.COMPLAINANT_TYPE
        )

    return None


def run_deterministic_analysis(
    request: AnalyzeRequest,
) -> AnalysisResponse:
    started = perf_counter()

    evidence: list[
        EvidenceItem
    ] = []

    insights: list[
        Insight
    ] = []

    limitations: list[str] = [
        (
            "The TDI public complaint dataset "
            "contains structured complaint "
            "classifications, not the full "
            "complaint narrative or underlying "
            "claim file."
        ),
        (
            "TDI consumer guidance is general "
            "regulatory guidance and does not "
            "replace the terms of an individual "
            "insurance policy."
        ),
    ]

    unresolved_questions: list[
        str
    ] = []

    tools_used: list[str] = []

    complaint_number = (
        extract_complaint_number(
            request
        )
    )

    if complaint_number:
        complaint = get_complaint(
            complaint_number
        )

        tools_used.append(
            "get_complaint"
        )

        if complaint is None:
            unresolved_questions.append(
                (
                    "No complaint record was "
                    f"found for complaint "
                    f"{complaint_number}."
                )
            )

        else:
            evidence_id = (
                "SQL-COMPLAINT-"
                f"{complaint.complaint_number}"
            )

            evidence.append(
                EvidenceItem(
                    evidence_id=(
                        evidence_id
                    ),
                    evidence_type=(
                        EvidenceType
                        .COMPLAINT_RECORD
                    ),
                    source=(
                        "TDI complaint database"
                    ),
                    content=(
                        complaint
                        .model_dump_json()
                    ),
                    metadata={
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
                )
            )

            keyword_text = (
                ", ".join(
                    complaint.keywords
                )
                if complaint.keywords
                else "No keywords recorded"
            )

            insights.append(
                Insight(
                    statement=(
                        "Complaint "
                        f"{complaint.complaint_number} "
                        "is recorded as "
                        f"'{complaint.finding_type}' "
                        "with issue tags: "
                        f"{keyword_text}."
                    ),
                    evidence_ids=[
                        evidence_id
                    ],
                )
            )

            insights.append(
                Insight(
                    statement=(
                        "The complaint was "
                        f"received on "
                        f"{complaint.received_date} "
                        "and closed on "
                        f"{complaint.closed_date}, "
                        "a recorded complaint "
                        "handling interval of "
                        f"{complaint.closure_days} "
                        "days."
                    ),
                    evidence_ids=[
                        evidence_id
                    ],
                )
            )

            unresolved_questions.extend(
                [
                    (
                        "What was stated in the "
                        "original complaint "
                        "narrative?"
                    ),
                    (
                        "What evidence was "
                        "contained in the "
                        "underlying claim file?"
                    ),
                    (
                        "What policy language "
                        "applied to the individual "
                        "complainant?"
                    ),
                ]
            )

    filters = infer_filters(
        request.question
    )

    statistics_group = (
        infer_statistics_group(
            request.question
        )
    )

    if statistics_group is not None:
        statistics = (
            get_complaint_statistics(
                filters=filters,
                group_by=(
                    statistics_group
                ),
                limit=10,
            )
        )

        tools_used.append(
            "get_complaint_statistics"
        )

        evidence_id = (
            "SQL-STATS-"
            f"{statistics_group.value.upper()}"
        )

        evidence.append(
            EvidenceItem(
                evidence_id=evidence_id,
                evidence_type=(
                    EvidenceType
                    .SQL_AGGREGATION
                ),
                source=(
                    "TDI complaint database"
                ),
                content=(
                    statistics
                    .model_dump_json()
                ),
                metadata={
                    "group_by": (
                        statistics_group
                        .value
                    ),
                    "matching_complaints": (
                        statistics
                        .total_matching_complaints
                    ),
                },
            )
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
                Insight(
                    statement=(
                        "The leading recorded "
                        f"{statistics_group.value} "
                        "groups are: "
                        f"{formatted}."
                    ),
                    evidence_ids=[
                        evidence_id
                    ],
                )
            )

    guidance = (
        search_insurance_guidance(
            query=request.question,
            top_k=(
                request.guidance_top_k
            ),
        )
    )

    tools_used.append(
        "search_insurance_guidance"
    )

    for hit in guidance.hits:
        evidence.append(
            EvidenceItem(
                evidence_id=(
                    hit.chunk_id
                ),
                evidence_type=(
                    EvidenceType
                    .GUIDANCE_DOCUMENT
                ),
                source=(
                    hit.document_title
                ),
                content=hit.text,
                metadata={
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
            )
        )

    if guidance.hits:
        strongest_hit = (
            guidance.hits[0]
        )

        insights.append(
            Insight(
                statement=(
                    "The most relevant retrieved "
                    "TDI guidance section is "
                    f"'{strongest_hit.section_title}' "
                    "from "
                    f"{strongest_hit.document_title}."
                ),
                evidence_ids=[
                    strongest_hit.chunk_id
                ],
            )
        )

    else:
        unresolved_questions.append(
            (
                "No sufficiently relevant "
                "TDI guidance was retrieved "
                "for this question."
            )
        )

    summary_parts: list[str] = []

    if complaint_number:
        summary_parts.append(
            (
                "The workflow checked the "
                "requested complaint record."
            )
        )

    if statistics_group is not None:
        summary_parts.append(
            (
                "It calculated complaint "
                "statistics from the structured "
                "TDI dataset."
            )
        )

    summary_parts.append(
        (
            "It retrieved relevant official "
            "TDI consumer guidance."
        )
    )

    summary = " ".join(
        summary_parts
    )

    report = (
        ComplaintIntelligenceReport(
            question=request.question,
            summary=summary,
            insights=insights,
            limitations=limitations,
            unresolved_questions=(
                unresolved_questions
            ),
            evidence=evidence,
        )
    )

    latency_ms = (
        perf_counter()
        - started
    ) * 1000

    return AnalysisResponse(
        workflow="deterministic",
        tools_used=tools_used,
        latency_ms=round(
            latency_ms,
            2,
        ),
        report=report,
    )

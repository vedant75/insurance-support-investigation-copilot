from __future__ import annotations

import hashlib

from pydantic import BaseModel

from insurance_copilot.domain.models import (
    ComplaintRecord,
    ComplaintSearchResult,
    ComplaintStatistics,
    EvidenceItem,
    EvidenceType,
    GuidanceSearchResult,
)


def _digest(
    result: BaseModel,
) -> str:
    payload = result.model_dump_json()

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def evidence_from_tool_result(
    tool_name: str,
    result: BaseModel | None,
) -> list[EvidenceItem]:
    if result is None:
        return []

    if isinstance(
        result,
        ComplaintRecord,
    ):
        return [
            EvidenceItem(
                evidence_id=(f"SQL-COMPLAINT-{result.complaint_number}"),
                evidence_type=(EvidenceType.COMPLAINT_RECORD),
                source=("TDI complaint database"),
                content=(result.model_dump_json()),
                metadata={
                    "complaint_number": (result.complaint_number),
                },
            )
        ]

    if isinstance(
        result,
        ComplaintStatistics,
    ):
        return [
            EvidenceItem(
                evidence_id=(f"SQL-STATS-{result.group_by.value.upper()}"),
                evidence_type=(EvidenceType.SQL_AGGREGATION),
                source=("TDI complaint database"),
                content=(result.model_dump_json()),
                metadata={
                    "group_by": (result.group_by.value),
                    "matching_complaints": (result.total_matching_complaints),
                },
            )
        ]

    if isinstance(
        result,
        ComplaintSearchResult,
    ):
        return [
            EvidenceItem(
                evidence_id=(f"SQL-SEARCH-{_digest(result)}"),
                evidence_type=(EvidenceType.SQL_AGGREGATION),
                source=("TDI complaint database"),
                content=(result.model_dump_json()),
                metadata={
                    "returned": (result.total_returned),
                },
            )
        ]

    if isinstance(
        result,
        GuidanceSearchResult,
    ):
        return [
            EvidenceItem(
                evidence_id=hit.chunk_id,
                evidence_type=(EvidenceType.GUIDANCE_DOCUMENT),
                source=(hit.document_title),
                content=hit.text,
                metadata={
                    "section": (hit.section_title),
                    "source_url": (hit.source_url),
                    "retrieval_score": (hit.score),
                },
            )
            for hit in result.hits
        ]

    raise TypeError(f"Unsupported tool result for {tool_name}: {type(result).__name__}")

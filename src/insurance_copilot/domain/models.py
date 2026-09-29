from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, model_validator

class StatisticsGroupBy(str, Enum):
    KEYWORD = "keyword"
    FINDING_TYPE = "finding_type"
    COVERAGE_LEVEL = "coverage_level"
    COMPLAINT_FILED_BY = "complaint_filed_by"
    COMPLAINANT_TYPE = "complainant_type"
    COMPLAINT_TYPE = "complaint_type"
    MONTH = "month"


class ComplaintFilters(BaseModel):
    date_from: date | None = None
    date_to: date | None = None

    finding_type: str | None = None
    coverage_level: str | None = None
    complaint_filed_by: str | None = None
    complainant_type: str | None = None
    complaint_type: str | None = None
    keyword: str | None = None


class ComplaintRecord(BaseModel):
    complaint_number: str

    complaint_filed_by: str
    received_date: date
    closed_date: date

    complaint_type: str
    coverage_type: str
    coverage_level: str

    others_involved: str | None = None
    complainant_type: str
    finding_type: str

    keywords: list[str] = Field(default_factory=list)

    closure_days: int | None = None


class StatisticRow(BaseModel):
    key: str
    count: int


class ComplaintStatistics(BaseModel):
    group_by: StatisticsGroupBy
    filters: ComplaintFilters

    total_matching_complaints: int
    rows: list[StatisticRow]


class ComplaintSearchResult(BaseModel):
    filters: ComplaintFilters
    total_returned: int
    complaints: list[ComplaintRecord]

class GuidanceChunk(BaseModel):
    chunk_id: str
    document_id: str

    document_title: str
    section_title: str

    source_url: str
    retrieved_at_utc: str

    text: str


class GuidanceHit(BaseModel):
    chunk_id: str
    document_title: str
    section_title: str

    source_url: str

    text: str
    score: float


class GuidanceSearchResult(BaseModel):
    query: str
    total_hits: int
    hits: list[GuidanceHit]

class EvidenceType(str, Enum):
    COMPLAINT_RECORD = "complaint_record"
    SQL_AGGREGATION = "sql_aggregation"
    GUIDANCE_DOCUMENT = "guidance_document"


class EvidenceItem(BaseModel):
    evidence_id: str
    evidence_type: EvidenceType

    source: str
    content: str

    metadata: dict[
        str,
        str | int | float | bool | None,
    ] = Field(default_factory=dict)


class Insight(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(
        default_factory=list
    )


class AnalyzeRequest(BaseModel):
    question: str = Field(
        min_length=3
    )

    complaint_number: str | None = None

    guidance_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )


class ComplaintIntelligenceReport(BaseModel):
    question: str

    summary: str

    insights: list[Insight] = Field(
        default_factory=list
    )

    limitations: list[str] = Field(
        default_factory=list
    )

    unresolved_questions: list[str] = Field(
        default_factory=list
    )

    evidence: list[EvidenceItem] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_evidence_references(
        self,
    ) -> "ComplaintIntelligenceReport":
        available_ids = {
            item.evidence_id
            for item in self.evidence
        }

        referenced_ids = {
            evidence_id
            for insight in self.insights
            for evidence_id
            in insight.evidence_ids
        }

        missing_ids = (
            referenced_ids
            - available_ids
        )

        if missing_ids:
            raise ValueError(
                "Insights reference unknown "
                "evidence IDs: "
                f"{sorted(missing_ids)}"
            )

        return self


class AnalysisResponse(BaseModel):
    workflow: str
    tools_used: list[str]

    latency_ms: float

    report: ComplaintIntelligenceReport

class WorkflowStatus(str, Enum):
    COMPLETED = "completed"
    PAUSED_FOR_REVIEW = "paused_for_review"
    REJECTED = "rejected"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class HumanReviewDecision(BaseModel):
    decision: ReviewDecision

    reviewer_note: str | None = None
    edited_summary: str | None = None

    @model_validator(mode="after")
    def validate_edit(
        self,
    ) -> "HumanReviewDecision":
        if (
            self.decision == ReviewDecision.EDIT
            and not self.edited_summary
        ):
            raise ValueError(
                "edited_summary is required "
                "when decision='edit'"
            )

        return self


class GraphAnalyzeRequest(AnalyzeRequest):
    require_human_review: bool = False
    thread_id: str | None = None


class ToolFailure(BaseModel):
    tool_name: str
    error_type: str
    message: str


class ReviewRequest(BaseModel):
    reason: str

    question: str

    evidence_count: int
    tools_used: list[str]

    unresolved_questions: list[str]
    tool_failures: list[ToolFailure]


class GraphAnalysisResponse(BaseModel):
    thread_id: str

    workflow: str = "langgraph_deterministic"
    status: WorkflowStatus

    tools_used: list[str]
    latency_ms: float

    report: ComplaintIntelligenceReport | None = None

    review_request: ReviewRequest | None = None
    review_decision: HumanReviewDecision | None = None

    tool_failures: list[ToolFailure] = Field(
        default_factory=list
    )
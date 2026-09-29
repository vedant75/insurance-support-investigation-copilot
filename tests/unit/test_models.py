import pytest
from pydantic import ValidationError

from insurance_copilot.domain.models import (
    ComplaintIntelligenceReport,
    Insight,
)


def test_report_rejects_unknown_evidence_reference() -> None:
    with pytest.raises(ValidationError):
        ComplaintIntelligenceReport(
            question="Test question",
            summary="Test summary",
            insights=[
                Insight(
                    statement="Unsupported claim",
                    evidence_ids=["FAKE-001"],
                )
            ],
            evidence=[],
        )

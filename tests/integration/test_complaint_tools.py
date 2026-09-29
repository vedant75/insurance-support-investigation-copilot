import sqlite3

import pytest

from insurance_copilot.config import get_settings
from insurance_copilot.db.connection import connect_read_only
from insurance_copilot.domain.models import (
    ComplaintFilters,
    StatisticsGroupBy,
)
from insurance_copilot.tools.complaints import (
    get_complaint,
    get_complaint_statistics,
)


def test_get_known_complaint() -> None:
    complaint = get_complaint("467758")

    assert complaint is not None
    assert complaint.complaint_number == "467758"
    assert complaint.coverage_type == "Automobile"
    assert "TOTAL LOSS" in complaint.keywords


def test_keyword_statistics() -> None:
    stats = get_complaint_statistics(
        filters=ComplaintFilters(),
        group_by=StatisticsGroupBy.KEYWORD,
        limit=5,
    )

    assert stats.total_matching_complaints > 0
    assert len(stats.rows) > 0

    assert stats.rows[0].key == "ADJUSTER'S HANDLING"


def test_database_is_read_only() -> None:
    connection = connect_read_only(get_settings().database_path)

    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM complaints")
    finally:
        connection.close()

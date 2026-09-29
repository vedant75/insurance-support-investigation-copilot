from __future__ import annotations

from insurance_copilot.config import get_settings
from insurance_copilot.db.connection import (
    connect_read_only,
)
from insurance_copilot.db.repository import (
    ComplaintRepository,
)
from insurance_copilot.domain.models import (
    ComplaintFilters,
    ComplaintRecord,
    ComplaintSearchResult,
    ComplaintStatistics,
    StatisticsGroupBy,
)


def get_complaint(
    complaint_number: str,
) -> ComplaintRecord | None:
    settings = get_settings()

    connection = connect_read_only(settings.database_path)

    try:
        repository = ComplaintRepository(connection)

        return repository.get_complaint(complaint_number)

    finally:
        connection.close()


def search_complaints(
    filters: ComplaintFilters,
    limit: int = 20,
) -> ComplaintSearchResult:
    settings = get_settings()

    connection = connect_read_only(settings.database_path)

    try:
        repository = ComplaintRepository(connection)

        return repository.search_complaints(
            filters=filters,
            limit=limit,
        )

    finally:
        connection.close()


def get_complaint_statistics(
    filters: ComplaintFilters,
    group_by: StatisticsGroupBy,
    limit: int = 20,
) -> ComplaintStatistics:
    settings = get_settings()

    connection = connect_read_only(settings.database_path)

    try:
        repository = ComplaintRepository(connection)

        return repository.get_statistics(
            filters=filters,
            group_by=group_by,
            limit=limit,
        )

    finally:
        connection.close()

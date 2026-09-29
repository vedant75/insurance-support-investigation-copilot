from datetime import date

from insurance_copilot.domain.models import (
    ComplaintFilters,
    StatisticsGroupBy,
)
from insurance_copilot.tools.complaints import (
    get_complaint,
    get_complaint_statistics,
    search_complaints,
)


def main() -> None:
    print("\n=== GET COMPLAINT ===")

    complaint = get_complaint("469059")

    print(complaint.model_dump_json(indent=2) if complaint else "Complaint not found")

    print("\n=== TOP KEYWORDS ===")

    keyword_stats = get_complaint_statistics(
        filters=ComplaintFilters(),
        group_by=(StatisticsGroupBy.KEYWORD),
        limit=10,
    )

    print(keyword_stats.model_dump_json(indent=2))

    print("\n=== MONTHLY VOLUME ===")

    monthly_stats = get_complaint_statistics(
        filters=ComplaintFilters(
            date_from=date(
                2026,
                1,
                1,
            ),
        ),
        group_by=(StatisticsGroupBy.MONTH),
        limit=100,
    )

    print(monthly_stats.model_dump_json(indent=2))

    print("\n=== CONFIRMED ADJUSTER HANDLING COMPLAINTS ===")

    search_result = search_complaints(
        filters=ComplaintFilters(
            finding_type="Confirmed",
            keyword="ADJUSTER'S HANDLING",
        ),
        limit=5,
    )

    print(search_result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import sqlite3

from insurance_copilot.domain.models import (
    ComplaintFilters,
    ComplaintRecord,
    ComplaintSearchResult,
    ComplaintStatistics,
    StatisticRow,
    StatisticsGroupBy,
)


GROUP_BY_EXPRESSIONS = {
    StatisticsGroupBy.FINDING_TYPE: "c.finding_type",
    StatisticsGroupBy.COVERAGE_LEVEL: "c.coverage_level",
    StatisticsGroupBy.COMPLAINT_FILED_BY: "c.complaint_filed_by",
    StatisticsGroupBy.COMPLAINANT_TYPE: "c.complainant_type",
    StatisticsGroupBy.COMPLAINT_TYPE: "c.complaint_type",
    StatisticsGroupBy.MONTH: "substr(c.received_date, 1, 7)",
}


class ComplaintRepository:
    def __init__(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        self.connection = connection

    def _build_filters(
        self,
        filters: ComplaintFilters,
    ) -> tuple[list[str], list[object]]:
        clauses: list[str] = []
        parameters: list[object] = []

        if filters.date_from:
            clauses.append(
                "c.received_date >= ?"
            )
            parameters.append(
                filters.date_from.isoformat()
            )

        if filters.date_to:
            clauses.append(
                "c.received_date <= ?"
            )
            parameters.append(
                filters.date_to.isoformat()
            )

        equality_filters = {
            "c.finding_type": filters.finding_type,
            "c.coverage_level": filters.coverage_level,
            "c.complaint_filed_by": (
                filters.complaint_filed_by
            ),
            "c.complainant_type": (
                filters.complainant_type
            ),
            "c.complaint_type": (
                filters.complaint_type
            ),
        }

        for column, value in equality_filters.items():
            if value is not None:
                clauses.append(
                    f"{column} = ?"
                )
                parameters.append(value)

        if filters.keyword is not None:
            clauses.append(
                """
                EXISTS (
                    SELECT 1
                    FROM complaint_keywords ck_filter
                    WHERE
                        ck_filter.complaint_number =
                            c.complaint_number
                        AND ck_filter.keyword = ?
                )
                """
            )

            parameters.append(
                filters.keyword
            )

        return clauses, parameters

    def _row_to_record(
        self,
        row: sqlite3.Row,
    ) -> ComplaintRecord:
        keyword_rows = self.connection.execute(
            """
            SELECT keyword
            FROM complaint_keywords
            WHERE complaint_number = ?
            ORDER BY keyword
            """,
            (row["complaint_number"],),
        ).fetchall()

        keywords = [
            keyword_row["keyword"]
            for keyword_row in keyword_rows
        ]

        return ComplaintRecord(
            complaint_number=row[
                "complaint_number"
            ],
            complaint_filed_by=row[
                "complaint_filed_by"
            ],
            received_date=row[
                "received_date"
            ],
            closed_date=row[
                "closed_date"
            ],
            complaint_type=row[
                "complaint_type"
            ],
            coverage_type=row[
                "coverage_type"
            ],
            coverage_level=row[
                "coverage_level"
            ],
            others_involved=row[
                "others_involved"
            ],
            complainant_type=row[
                "complainant_type"
            ],
            finding_type=row[
                "finding_type"
            ],
            keywords=keywords,
            closure_days=row[
                "closure_days"
            ],
        )

    def get_complaint(
        self,
        complaint_number: str,
    ) -> ComplaintRecord | None:
        row = self.connection.execute(
            """
            SELECT *
            FROM complaints
            WHERE complaint_number = ?
            """,
            (complaint_number,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def search_complaints(
        self,
        filters: ComplaintFilters,
        limit: int = 20,
    ) -> ComplaintSearchResult:
        limit = max(
            1,
            min(limit, 100),
        )

        clauses, parameters = self._build_filters(
            filters
        )

        where_sql = ""

        if clauses:
            where_sql = (
                "WHERE "
                + " AND ".join(clauses)
            )

        rows = self.connection.execute(
            f"""
            SELECT c.*
            FROM complaints c
            {where_sql}
            ORDER BY c.received_date DESC
            LIMIT ?
            """,
            [*parameters, limit],
        ).fetchall()

        complaints = [
            self._row_to_record(row)
            for row in rows
        ]

        return ComplaintSearchResult(
            filters=filters,
            total_returned=len(complaints),
            complaints=complaints,
        )

    def get_statistics(
        self,
        filters: ComplaintFilters,
        group_by: StatisticsGroupBy,
        limit: int = 20,
    ) -> ComplaintStatistics:
        limit = max(
            1,
            min(limit, 100),
        )

        clauses, parameters = self._build_filters(
            filters
        )

        where_sql = ""

        if clauses:
            where_sql = (
                "WHERE "
                + " AND ".join(clauses)
            )

        total_matching = self.connection.execute(
            f"""
            SELECT COUNT(*)
            FROM complaints c
            {where_sql}
            """,
            parameters,
        ).fetchone()[0]

        if group_by == StatisticsGroupBy.KEYWORD:
            rows = self.connection.execute(
                f"""
                SELECT
                    ck.keyword AS group_key,
                    COUNT(*) AS complaint_count
                FROM complaints c

                JOIN complaint_keywords ck
                    ON ck.complaint_number =
                       c.complaint_number

                {where_sql}

                GROUP BY ck.keyword
                ORDER BY complaint_count DESC,
                         group_key ASC
                LIMIT ?
                """,
                [*parameters, limit],
            ).fetchall()

        else:
            expression = GROUP_BY_EXPRESSIONS[
                group_by
            ]

            ordering = (
                "group_key ASC"
                if group_by
                == StatisticsGroupBy.MONTH
                else
                "complaint_count DESC, group_key ASC"
            )

            rows = self.connection.execute(
                f"""
                SELECT
                    {expression} AS group_key,
                    COUNT(*) AS complaint_count
                FROM complaints c

                {where_sql}

                GROUP BY {expression}

                ORDER BY {ordering}

                LIMIT ?
                """,
                [*parameters, limit],
            ).fetchall()

        statistics = [
            StatisticRow(
                key=str(row["group_key"]),
                count=row["complaint_count"],
            )
            for row in rows
        ]

        return ComplaintStatistics(
            group_by=group_by,
            filters=filters,
            total_matching_complaints=(
                total_matching
            ),
            rows=statistics,
        )

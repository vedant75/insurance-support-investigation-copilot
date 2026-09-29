from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MVP_PATH = PROJECT_ROOT / "data" / "processed" / "auto_complaints_mvp.csv"
FULL_PATH = PROJECT_ROOT / "data" / "processed" / "auto_complaints.csv"

DATABASE_PATH = PROJECT_ROOT / "data" / "runtime" / "complaints.db"


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS complaint_keywords;
DROP TABLE IF EXISTS complaints;
DROP TABLE IF EXISTS dataset_metadata;


CREATE TABLE complaints (
    complaint_number TEXT PRIMARY KEY,

    complaint_filed_by TEXT NOT NULL,

    received_date TEXT NOT NULL,
    closed_date TEXT NOT NULL,

    complaint_type TEXT NOT NULL,
    coverage_type TEXT NOT NULL,
    coverage_level TEXT NOT NULL,

    others_involved TEXT,

    complainant_type TEXT NOT NULL,
    finding_type TEXT NOT NULL,

    keywords_raw TEXT,

    closure_days INTEGER
);


CREATE TABLE complaint_keywords (
    complaint_number TEXT NOT NULL,
    keyword TEXT NOT NULL,

    PRIMARY KEY (complaint_number, keyword),

    FOREIGN KEY (complaint_number)
        REFERENCES complaints (complaint_number)
        ON DELETE CASCADE
);


CREATE TABLE dataset_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);


CREATE INDEX idx_complaints_received_date
    ON complaints(received_date);

CREATE INDEX idx_complaints_closed_date
    ON complaints(closed_date);

CREATE INDEX idx_complaints_finding_type
    ON complaints(finding_type);

CREATE INDEX idx_complaints_coverage_level
    ON complaints(coverage_level);

CREATE INDEX idx_complaints_filed_by
    ON complaints(complaint_filed_by);

CREATE INDEX idx_complaints_complainant_type
    ON complaints(complainant_type);

CREATE INDEX idx_complaints_complaint_type
    ON complaints(complaint_type);

CREATE INDEX idx_keywords_keyword
    ON complaint_keywords(keyword);

CREATE INDEX idx_keywords_complaint
    ON complaint_keywords(complaint_number);
"""


def normalize_optional(value: object) -> str | None:
    if pd.isna(value):
        return None

    value = str(value).strip()

    return value or None


def split_keywords(value: object) -> list[str]:
    if pd.isna(value):
        return []

    keywords = []

    for keyword in str(value).split(";"):
        keyword = keyword.strip()

        if keyword:
            keywords.append(keyword)

    return sorted(set(keywords))


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run scripts/download_data.py first.")

    df = pd.read_csv(
        path,
        dtype={"Complaint number": "string"},
        low_memory=False,
    )

    for column in ["Received date", "Closed date"]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    if df["Complaint number"].duplicated().any():
        raise ValueError("Complaint numbers are not unique in the selected dataset.")

    return df


def build_database(source_path: Path) -> None:
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    df = load_dataframe(source_path)

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        connection.executescript(SCHEMA_SQL)

        complaint_rows = []
        keyword_rows = []

        for row in df.itertuples(index=False, name=None):
            record = dict(zip(df.columns, row, strict=True))

            received_date = record["Received date"]
            closed_date = record["Closed date"]

            closure_days = None

            if pd.notna(received_date) and pd.notna(closed_date):
                closure_days = int((closed_date - received_date).days)

            complaint_number = str(record["Complaint number"])

            keywords = split_keywords(record["Keywords"])

            complaint_rows.append(
                (
                    complaint_number,
                    record["Complaint filed by"],
                    received_date.date().isoformat(),
                    closed_date.date().isoformat(),
                    record["Complaint type"],
                    record["Coverage type"],
                    record["Coverage level"],
                    normalize_optional(record["Others involved"]),
                    record["Complainant type"],
                    record["Finding type"],
                    normalize_optional(record["Keywords"]),
                    closure_days,
                )
            )

            keyword_rows.extend((complaint_number, keyword) for keyword in keywords)

        connection.executemany(
            """
            INSERT INTO complaints (
                complaint_number,
                complaint_filed_by,
                received_date,
                closed_date,
                complaint_type,
                coverage_type,
                coverage_level,
                others_involved,
                complainant_type,
                finding_type,
                keywords_raw,
                closure_days
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            complaint_rows,
        )

        connection.executemany(
            """
            INSERT INTO complaint_keywords (
                complaint_number,
                keyword
            )
            VALUES (?, ?)
            """,
            keyword_rows,
        )

        metadata = {
            "source_file": str(source_path),
            "database_built_at_utc": (datetime.now(timezone.utc).isoformat()),
            "complaint_count": str(len(complaint_rows)),
            "keyword_relationship_count": str(len(keyword_rows)),
        }

        connection.executemany(
            """
            INSERT INTO dataset_metadata(key, value)
            VALUES (?, ?)
            """,
            metadata.items(),
        )

        connection.commit()

        complaint_count = connection.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]

        keyword_count = connection.execute("SELECT COUNT(*) FROM complaint_keywords").fetchone()[0]

        unique_keywords = connection.execute(
            """
            SELECT COUNT(DISTINCT keyword)
            FROM complaint_keywords
            """
        ).fetchone()[0]

        print("=" * 70)
        print("DATABASE BUILD COMPLETE")
        print("=" * 70)
        print(f"Source:                  {source_path}")
        print(f"Database:                {DATABASE_PATH}")
        print(f"Complaints:              {complaint_count:,}")
        print(f"Keyword relationships:   {keyword_count:,}")
        print(f"Unique keywords:         {unique_keywords:,}")

    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
        help="Load the full Automobile dataset instead of the 5,000-row MVP.",
    )

    args = parser.parse_args()

    source_path = FULL_PATH if args.full else MVP_PATH

    build_database(source_path)


if __name__ == "__main__":
    main()

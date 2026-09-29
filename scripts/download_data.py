from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATASET_URL = "https://data.texas.gov/api/views/jjc8-mxkg/rows.csv?accessType=DOWNLOAD"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

RAW_PATH = RAW_DIR / "tdi_complaints.csv"
AUTO_PATH = PROCESSED_DIR / "auto_complaints.csv"
MVP_PATH = PROCESSED_DIR / "auto_complaints_mvp.csv"

PROFILE_PATH = METADATA_DIR / "auto_complaints_profile.json"
DOWNLOAD_METADATA_PATH = METADATA_DIR / "download_metadata.json"

MVP_SIZE = 5_000


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, METADATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_dataset() -> None:
    print(f"Downloading TDI complaint data from:\n{DATASET_URL}\n")
    urlretrieve(DATASET_URL, RAW_PATH)

    print(f"Saved raw dataset → {RAW_PATH}")
    print(f"Size: {RAW_PATH.stat().st_size / (1024 * 1024):.2f} MB")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_PATH,
        dtype={"Complaint number": "string"},
        low_memory=False,
    )

    df.columns = df.columns.str.strip()

    required_columns = {
        "Complaint number",
        "Received date",
        "Closed date",
        "Coverage type",
        "Coverage level",
        "Complaint type",
        "Finding type",
        "Keywords",
        "Complaint filed by",
        "Complainant type",
        "Others involved",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset schema changed. Missing expected columns: {sorted(missing_columns)}"
        )

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in ("Received date", "Closed date"):
        df[column] = pd.to_datetime(
            df[column],
            format="mixed",
            errors="coerce",
        )

    return df


def build_auto_subset(df: pd.DataFrame) -> pd.DataFrame:
    auto = df.loc[
        df["Coverage type"].astype("string").str.strip().str.casefold().eq("automobile")
    ].copy()

    auto = auto.sort_values(
        "Received date",
        ascending=False,
        na_position="last",
    )

    return auto


def top_values(
    series: pd.Series,
    limit: int = 20,
) -> dict[str, int]:
    values = series.astype("string").fillna("<MISSING>").value_counts(dropna=False).head(limit)

    return {str(key): int(value) for key, value in values.items()}


def build_profile(df: pd.DataFrame) -> dict:
    categorical_columns = [
        "Coverage type",
        "Coverage level",
        "Complaint type",
        "Finding type",
        "Keywords",
        "Complaint filed by",
        "Complainant type",
        "Others involved",
    ]

    null_counts = {column: int(df[column].isna().sum()) for column in df.columns}

    null_percentages = {
        column: round(float(df[column].isna().mean() * 100), 2) for column in df.columns
    }

    unique_counts = {column: int(df[column].nunique(dropna=True)) for column in df.columns}

    top_value_profile = {
        column: top_values(df[column]) for column in categorical_columns if column in df.columns
    }

    received_dates = df["Received date"].dropna()
    closed_dates = df["Closed date"].dropna()

    closure_days = (df["Closed date"] - df["Received date"]).dt.days

    valid_closure_days = closure_days[closure_days.ge(0)].dropna()

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "column_count": int(len(df.columns)),
        "duplicate_complaint_numbers": int(df["Complaint number"].duplicated().sum()),
        "date_range": {
            "received_min": (
                received_dates.min().isoformat() if not received_dates.empty else None
            ),
            "received_max": (
                received_dates.max().isoformat() if not received_dates.empty else None
            ),
            "closed_min": (closed_dates.min().isoformat() if not closed_dates.empty else None),
            "closed_max": (closed_dates.max().isoformat() if not closed_dates.empty else None),
        },
        "closure_time_days": {
            "median": (
                round(float(valid_closure_days.median()), 2)
                if not valid_closure_days.empty
                else None
            ),
            "p90": (
                round(float(valid_closure_days.quantile(0.90)), 2)
                if not valid_closure_days.empty
                else None
            ),
            "p95": (
                round(float(valid_closure_days.quantile(0.95)), 2)
                if not valid_closure_days.empty
                else None
            ),
        },
        "null_counts": null_counts,
        "null_percentages": null_percentages,
        "unique_counts": unique_counts,
        "top_values": top_value_profile,
    }


def print_summary(
    full_df: pd.DataFrame,
    auto_df: pd.DataFrame,
    mvp_df: pd.DataFrame,
    profile: dict,
) -> None:
    print("\n" + "=" * 70)
    print("TDI DATASET SUMMARY")
    print("=" * 70)

    print(f"Full dataset rows:       {len(full_df):,}")
    print(f"Automobile rows:         {len(auto_df):,}")
    print(f"MVP rows:                {len(mvp_df):,}")
    print(f"Columns:                 {len(auto_df.columns)}")

    print(f"Duplicate complaint IDs: {profile['duplicate_complaint_numbers']:,}")

    print(
        "Automobile received range:",
        profile["date_range"]["received_min"],
        "→",
        profile["date_range"]["received_max"],
    )

    print("\nColumns:")
    for column in auto_df.columns:
        print(f"  - {column}")

    for column in [
        "Coverage level",
        "Complaint type",
        "Finding type",
        "Keywords",
        "Complaint filed by",
        "Complainant type",
    ]:
        print(f"\nTop values — {column}")

        counts = (
            auto_df[column].astype("string").fillna("<MISSING>").value_counts(dropna=False).head(10)
        )

        print(counts.to_string())

    print("\nMissingness:")
    missing = auto_df.isna().mean().mul(100).sort_values(ascending=False)

    for column, percentage in missing.items():
        print(f"  {column:<25} {percentage:6.2f}%")

    print("\nLatest 5 automobile complaints:")
    preview_columns = [
        "Complaint number",
        "Received date",
        "Coverage level",
        "Complaint type",
        "Finding type",
        "Keywords",
    ]

    print(auto_df[preview_columns].head().to_string(index=False))

    print("\nGenerated files:")
    print(f"  {RAW_PATH}")
    print(f"  {AUTO_PATH}")
    print(f"  {MVP_PATH}")
    print(f"  {PROFILE_PATH}")
    print(f"  {DOWNLOAD_METADATA_PATH}")


def main() -> None:
    ensure_directories()

    downloaded_at = datetime.now(timezone.utc)

    download_dataset()

    raw_hash = sha256_file(RAW_PATH)

    full_df = load_dataset()
    full_df = parse_dates(full_df)

    auto_df = build_auto_subset(full_df)

    if auto_df.empty:
        raise RuntimeError(
            "No Automobile records were found. Check whether TDI changed Coverage type values."
        )

    mvp_df = auto_df.head(MVP_SIZE).copy()

    auto_df.to_csv(
        AUTO_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    mvp_df.to_csv(
        MVP_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    profile = build_profile(auto_df)

    with PROFILE_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            profile,
            file,
            indent=2,
            ensure_ascii=False,
        )

    download_metadata = {
        "dataset_name": "TDI Insurance complaints: One record / complaint",
        "dataset_id": "jjc8-mxkg",
        "source_url": DATASET_URL,
        "downloaded_at_utc": downloaded_at.isoformat(),
        "raw_file_sha256": raw_hash,
        "full_dataset_rows": int(len(full_df)),
        "automobile_rows": int(len(auto_df)),
        "mvp_rows": int(len(mvp_df)),
        "mvp_definition": (
            f"Latest {MVP_SIZE:,} Automobile complaints ordered by Received date descending."
        ),
    }

    with DOWNLOAD_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            download_metadata,
            file,
            indent=2,
        )

    print_summary(
        full_df=full_df,
        auto_df=auto_df,
        mvp_df=mvp_df,
        profile=profile,
    )


if __name__ == "__main__":
    main()

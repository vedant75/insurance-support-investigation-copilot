from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_read_only(
    database_path: Path,
) -> sqlite3.Connection:
    resolved_path = database_path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Database not found: {resolved_path}. Run scripts/build_database.py first."
        )

    uri = f"file:{resolved_path}?mode=ro"

    connection = sqlite3.connect(
        uri,
        uri=True,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA query_only = ON")

    connection.execute("PRAGMA foreign_keys = ON")

    return connection

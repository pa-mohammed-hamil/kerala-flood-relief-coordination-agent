
#!/usr/bin/env python3
"""
Kerala Flood Relief Coordination Agent
backend/scripts/seed_database.py

Seed a DEVELOPMENT database with synthetic flood-relief datasets.

Supported:
    SQLite (standard library)
    PostgreSQL (pip install "psycopg[binary]")

Features:
    - 18 synthetic datasets
    - Safe JSON validation
    - Database schema initialization
    - JSON payload preservation
    - Foreign-key relationships for key entities
    - Parameterized SQL
    - Duplicate-safe upserts
    - Atomic transactions
    - Dry-run validation
    - Development-only database reset
    - Seeding summary

Examples:
    python backend/scripts/seed_database.py

    python backend/scripts/seed_database.py \\
        --input data/demo \\
        --database sqlite:///data/kerala_flood_demo.db

    python backend/scripts/seed_database.py \\
        --input data/demo \\
        --database postgresql://user:password@localhost:5432/flood_demo

    python backend/scripts/seed_database.py --dry-run

    python backend/scripts/seed_database.py --reset --confirm-reset

SAFETY:
    Synthetic development data only.
    Do not run against production databases.
    No real emergency operations are performed.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DATASETS = [
    "districts",
    "rainfall",
    "rivers",
    "dams",
    "incidents",
    "sos_requests",
    "rescue_teams",
    "shelters",
    "hospitals",
    "volunteers",
    "volunteer_assignments",
    "relief_inventory",
    "supply_chain",
    "relief_requests",
    "reunification_cases",
    "damage_assessments",
    "alerts",
    "audit_logs",
]

PRIMARY_KEYS = {
    "districts": "district_id",
    "rainfall": "rainfall_id",
    "rivers": "gauge_id",
    "dams": "dam_id",
    "incidents": "incident_id",
    "sos_requests": "sos_id",
    "rescue_teams": "team_id",
    "shelters": "shelter_id",
    "hospitals": "hospital_id",
    "volunteers": "volunteer_id",
    "volunteer_assignments": "assignment_id",
    "relief_inventory": "inventory_id",
    "supply_chain": "shipment_id",
    "relief_requests": "request_id",
    "reunification_cases": "case_id",
    "damage_assessments": "assessment_id",
    "alerts": "alert_id",
    "audit_logs": "event_id",
}

DISTRICT_NAMES = {
    "Thiruvananthapuram",
    "Kollam",
    "Pathanamthitta",
    "Alappuzha",
    "Kottayam",
    "Idukki",
    "Ernakulam",
    "Thrissur",
    "Palakkad",
    "Malappuram",
    "Kozhikode",
    "Wayanad",
    "Kannur",
    "Kasaragod",
}

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_RECORDS = 100000


class SeedError(Exception):
    """Safe, expected database-seeding error."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )


def is_postgres(url: str) -> bool:
    return url.startswith(
        ("postgresql://", "postgres://")
    )


def is_sqlite(url: str) -> bool:
    return url.startswith("sqlite:///")


def database_label(url: str) -> str:
    if is_sqlite(url):
        return url.removeprefix("sqlite:///")

    if is_postgres(url):
        parsed = urlsplit(url)
        return (
            f"{parsed.scheme}://"
            f"{parsed.hostname or 'unknown'}"
            f":{parsed.port or 5432}"
            f"{parsed.path}"
        )

    return "unknown"


def validate_database_target(
    database_url: str,
    reset: bool,
    confirm_reset: bool,
) -> None:
    if not (
        is_sqlite(database_url)
        or is_postgres(database_url)
    ):
        raise SeedError(
            "Supported database URLs: "
            "sqlite:///... or postgresql://..."
        )

    if reset and not confirm_reset:
        raise SeedError(
            "--reset requires --confirm-reset."
        )

    if os.environ.get("APP_ENV", "development").lower() in {
        "production",
        "prod",
        "staging",
    }:
        raise SeedError(
            "Seeding is blocked in production/staging."
        )

    if os.environ.get("ALLOW_DEMO_SEED", "1") != "1":
        raise SeedError(
            "Demo seeding disabled by environment."
        )

    if is_postgres(database_url):
        parsed = urlsplit(database_url)

        dbname = parsed.path.lstrip("/").lower()

        if not any(
            token in dbname
            for token in ("demo", "dev", "test", "sandbox")
        ):
            raise SeedError(
                "PostgreSQL database name must contain "
                "demo, dev, test, or sandbox."
            )

        if not dbname:
            raise SeedError(
                "PostgreSQL database name is required."
            )


def read_dataset(
    path: Path,
    name: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SeedError(
            f"Missing dataset: {path}"
        )

    if path.stat().st_size > MAX_FILE_BYTES:
        raise SeedError(
            f"{name}: file exceeds size limit."
        )

    try:
        records = json.loads(
            path.read_text(encoding="utf-8-sig")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SeedError(
            f"Cannot parse {name}: {exc}"
        ) from exc

    if not isinstance(records, list):
        raise SeedError(
            f"{name}: expected a JSON array."
        )

    if len(records) > MAX_RECORDS:
        raise SeedError(
            f"{name}: too many records."
        )

    pk = PRIMARY_KEYS[name]
    ids = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SeedError(
                f"{name}[{index}]: expected object."
            )

        if (
            record.get("synthetic") is not True
            or record.get("operational_use") is not False
        ):
            raise SeedError(
                f"{name}[{index}]: "
                "not explicitly synthetic."
            )

        record_id = record.get(pk)

        if (
            not isinstance(record_id, str)
            or not record_id.strip()
            or len(record_id) > 100
        ):
            raise SeedError(
                f"{name}[{index}]: invalid primary key."
            )

        if record_id in ids:
            raise SeedError(
                f"{name}: duplicate ID {record_id}"
            )

        ids.add(record_id)

    return records


def load_all(
    input_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    return {
        name: read_dataset(
            input_dir / f"{name}.json",
            name,
        )
        for name in DATASETS
    }


def validate_references(
    datasets: dict[str, list[dict]],
) -> None:
    district_names = {
        row["name"]
        for row in datasets["districts"]
    }

    if district_names != DISTRICT_NAMES:
        raise SeedError(
            "District dataset must contain exactly "
            "the 14 Kerala districts."
        )

    incident_ids = {
        row["incident_id"]
        for row in datasets["incidents"]
    }

    volunteer_ids = {
        row["volunteer_id"]
        for row in datasets["volunteers"]
    }

    warehouse_ids = {
        row["warehouse_id"]
        for row in datasets["relief_inventory"]
    }

    for sos in datasets["sos_requests"]:
        if sos["incident_id"] not in incident_ids:
            raise SeedError(
                f"Unknown incident reference: "
                f"{sos['incident_id']}"
            )

    for assignment in datasets["volunteer_assignments"]:
        volunteer_id = assignment.get(
            "volunteer_id"
        )

        if (
            volunteer_id
            and volunteer_id not in volunteer_ids
        ):
            raise SeedError(
                f"Unknown volunteer: {volunteer_id}"
            )

    for shipment in datasets["supply_chain"]:
        if shipment["warehouse_id"] not in warehouse_ids:
            raise SeedError(
                f"Unknown warehouse: "
                f"{shipment['warehouse_id']}"
            )

    for name, rows in datasets.items():
        for row in rows:
            district = row.get("district")

            if (
                district is not None
                and district not in DISTRICT_NAMES
            ):
                raise SeedError(
                    f"{name}: unknown district {district}"
                )


def connect_database(url: str):
    if is_sqlite(url):
        path_string = url.removeprefix(
            "sqlite:///"
        )

        if not path_string:
            raise SeedError(
                "SQLite database path is empty."
            )

        if path_string == ":memory:":
            connection = sqlite3.connect(
                ":memory:"
            )
        else:
            path = Path(path_string)
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            connection = sqlite3.connect(path)

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection, "sqlite"

    try:
        import psycopg
    except ImportError as exc:
        raise SeedError(
            'PostgreSQL requires: pip install "psycopg[binary]"'
        ) from exc

    connection = psycopg.connect(
        url,
        connect_timeout=10,
    )

    return connection, "postgresql"


def execute(
    connection,
    backend: str,
    sql: str,
    params: tuple = (),
):
    if backend == "postgresql":
        sql = sql.replace("?", "%s")

    cursor = connection.cursor()
    cursor.execute(sql, params)
    return cursor


def create_schema(
    connection,
    backend: str,
) -> None:
    execute(
        connection,
        backend,
        """
        CREATE TABLE IF NOT EXISTS seed_metadata (
            dataset_name VARCHAR(100) PRIMARY KEY,
            record_count INTEGER NOT NULL,
            seeded_at VARCHAR(50) NOT NULL,
            synthetic INTEGER NOT NULL
        )
        """,
    )

    execute(
        connection,
        backend,
        """
        CREATE TABLE IF NOT EXISTS demo_districts (
            district_id VARCHAR(100) PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            payload TEXT NOT NULL,
            updated_at VARCHAR(50) NOT NULL
        )
        """,
    )

    execute(
        connection,
        backend,
        """
        CREATE TABLE IF NOT EXISTS demo_incidents (
            incident_id VARCHAR(100) PRIMARY KEY,
            district VARCHAR(100) NOT NULL,
            priority VARCHAR(30),
            status VARCHAR(40),
            payload TEXT NOT NULL,
            FOREIGN KEY (district)
                REFERENCES demo_districts(name)
        )
        """,
    )

    execute(
        connection,
        backend,
        """
        CREATE TABLE IF NOT EXISTS demo_volunteers (
            volunteer_id VARCHAR(100) PRIMARY KEY,
            district VARCHAR(100) NOT NULL,
            review_status VARCHAR(40),
            available INTEGER NOT NULL,
            payload TEXT NOT NULL,
            FOREIGN KEY (district)
                REFERENCES demo_districts(name)
        )
        """,
    )

    execute(
        connection,
        backend,
        """
        CREATE TABLE IF NOT EXISTS demo_sos_requests (
            sos_id VARCHAR(100) PRIMARY KEY,
            incident_id VARCHAR(100) NOT NULL,
            district VARCHAR(100) NOT NULL,
            priority VARCHAR(30),
            status VARCHAR(40),
            payload TEXT NOT NULL,
            FOREIGN KEY (incident_id)
                REFERENCES demo_incidents(incident_id),
            FOREIGN KEY (district)
                REFERENCES demo_districts(name)
        )
        """,
    )

    execute(
        connection,
        backend,
        """
        CREATE TABLE IF NOT EXISTS demo_volunteer_assignments (
            assignment_id VARCHAR(100) PRIMARY KEY,
            volunteer_id VARCHAR(100),
            district VARCHAR(100) NOT NULL,
            status VARCHAR(40),
            payload TEXT NOT NULL,
            FOREIGN KEY (volunteer_id)
                REFERENCES demo_volunteers(volunteer_id),
            FOREIGN KEY (district)
                REFERENCES demo_districts(name)
        )
        """,
    )

    execute(
        connection,
        backend,
        """
        CREATE TABLE IF NOT EXISTS demo_records (
            dataset_name VARCHAR(100) NOT NULL,
            record_id VARCHAR(100) NOT NULL,
            district VARCHAR(100),
            status VARCHAR(50),
            payload TEXT NOT NULL,
            PRIMARY KEY (dataset_name, record_id)
        )
        """,
    )

    execute(
        connection,
        backend,
        """
        CREATE INDEX IF NOT EXISTS idx_demo_records_dataset
        ON demo_records(dataset_name)
        """,
    )

    execute(
        connection,
        backend,
        """
        CREATE INDEX IF NOT EXISTS idx_demo_records_district
        ON demo_records(district)
        """,
    )

    execute(
        connection,
        backend,
        """
        CREATE INDEX IF NOT EXISTS idx_demo_records_status
        ON demo_records(status)
        """,
    )


def reset_schema(
    connection,
    backend: str,
) -> None:
    tables = [
        "demo_volunteer_assignments",
        "demo_sos_requests",
        "demo_volunteers",
        "demo_incidents",
        "demo_districts",
        "demo_records",
        "seed_metadata",
    ]

    for table in tables:
        execute(
            connection,
            backend,
            f"DROP TABLE IF EXISTS {table}",
        )


def payload_json(record: dict) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def upsert(
    connection,
    backend: str,
    table: str,
    columns: list[str],
    values: tuple,
    conflict_columns: list[str],
) -> None:
    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    column_sql = ", ".join(columns)
    conflict_sql = ", ".join(
        conflict_columns
    )

    update_columns = [
        column
        for column in columns
        if column not in conflict_columns
    ]

    update_sql = ", ".join(
        f"{column} = excluded.{column}"
        for column in update_columns
    )

    sql = (
        f"INSERT INTO {table} ({column_sql}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_sql}) "
        f"DO UPDATE SET {update_sql}"
    )

    execute(
        connection,
        backend,
        sql,
        values,
    ).close()


def seed_districts(
    connection,
    backend: str,
    rows: list[dict],
) -> None:
    for row in rows:
        upsert(
            connection,
            backend,
            "demo_districts",
            [
                "district_id",
                "name",
                "payload",
                "updated_at",
            ],
            (
                row["district_id"],
                row["name"],
                payload_json(row),
                row.get("updated_at", now_iso()),
            ),
            ["district_id"],
        )


def seed_incidents(
    connection,
    backend: str,
    rows: list[dict],
) -> None:
    for row in rows:
        upsert(
            connection,
            backend,
            "demo_incidents",
            [
                "incident_id",
                "district",
                "priority",
                "status",
                "payload",
            ],
            (
                row["incident_id"],
                row["district"],
                row.get("priority"),
                row.get("status"),
                payload_json(row),
            ),
            ["incident_id"],
        )


def seed_volunteers(
    connection,
    backend: str,
    rows: list[dict],
) -> None:
    for row in rows:
        upsert(
            connection,
            backend,
            "demo_volunteers",
            [
                "volunteer_id",
                "district",
                "review_status",
                "available",
                "payload",
            ],
            (
                row["volunteer_id"],
                row["district"],
                row.get("review_status"),
                int(row.get("available", False)),
                payload_json(row),
            ),
            ["volunteer_id"],
        )


def seed_sos(
    connection,
    backend: str,
    rows: list[dict],
) -> None:
    for row in rows:
        upsert(
            connection,
            backend,
            "demo_sos_requests",
            [
                "sos_id",
                "incident_id",
                "district",
                "priority",
                "status",
                "payload",
            ],
            (
                row["sos_id"],
                row["incident_id"],
                row["district"],
                row.get("priority"),
                row.get("status"),
                payload_json(row),
            ),
            ["sos_id"],
        )


def seed_assignments(
    connection,
    backend: str,
    rows: list[dict],
) -> None:
    for row in rows:
        volunteer_id = row.get(
            "volunteer_id"
        ) or None

        upsert(
            connection,
            backend,
            "demo_volunteer_assignments",
            [
                "assignment_id",
                "volunteer_id",
                "district",
                "status",
                "payload",
            ],
            (
                row["assignment_id"],
                volunteer_id,
                row["district"],
                row.get("status"),
                payload_json(row),
            ),
            ["assignment_id"],
        )


def seed_generic_dataset(
    connection,
    backend: str,
    name: str,
    rows: list[dict],
) -> None:
    pk = PRIMARY_KEYS[name]

    for row in rows:
        status = (
            row.get("status")
            or row.get("risk_level")
            or row.get("stock_status")
            or row.get("review_status")
        )

        upsert(
            connection,
            backend,
            "demo_records",
            [
                "dataset_name",
                "record_id",
                "district",
                "status",
                "payload",
            ],
            (
                name,
                row[pk],
                row.get("district"),
                status,
                payload_json(row),
            ),
            ["dataset_name", "record_id"],
        )


def seed_metadata(
    connection,
    backend: str,
    name: str,
    count: int,
) -> None:
    upsert(
        connection,
        backend,
        "seed_metadata",
        [
            "dataset_name",
            "record_count",
            "seeded_at",
            "synthetic",
        ],
        (
            name,
            count,
            now_iso(),
            1,
        ),
        ["dataset_name"],
    )


def seed_all(
    connection,
    backend: str,
    datasets: dict[str, list[dict]],
) -> None:
    seed_districts(
        connection,
        backend,
        datasets["districts"],
    )

    seed_incidents(
        connection,
        backend,
        datasets["incidents"],
    )

    seed_volunteers(
        connection,
        backend,
        datasets["volunteers"],
    )

    seed_sos(
        connection,
        backend,
        datasets["sos_requests"],
    )

    seed_assignments(
        connection,
        backend,
        datasets["volunteer_assignments"],
    )

    special = {
        "districts",
        "incidents",
        "volunteers",
        "sos_requests",
        "volunteer_assignments",
    }

    for name in DATASETS:
        if name not in special:
            seed_generic_dataset(
                connection,
                backend,
                name,
                datasets[name],
            )

        seed_metadata(
            connection,
            backend,
            name,
            len(datasets[name]),
        )


def verify_counts(
    connection,
    backend: str,
    datasets: dict[str, list[dict]],
) -> None:
    special_tables = {
        "districts": "demo_districts",
        "incidents": "demo_incidents",
        "volunteers": "demo_volunteers",
        "sos_requests": "demo_sos_requests",
        "volunteer_assignments": "demo_volunteer_assignments",
    }

    for name, expected_rows in datasets.items():
        if name in special_tables:
            table = special_tables[name]

            cursor = execute(
                connection,
                backend,
                f"SELECT COUNT(*) FROM {table}",
            )
        else:
            cursor = execute(
                connection,
                backend,
                """
                SELECT COUNT(*)
                FROM demo_records
                WHERE dataset_name = ?
                """,
                (name,),
            )

        actual = cursor.fetchone()[0]
        cursor.close()

        if actual < len(expected_rows):
            raise SeedError(
                f"Count verification failed for {name}: "
                f"expected at least {len(expected_rows)}, "
                f"found {actual}."
            )


def print_summary(
    datasets: dict[str, list[dict]],
    database_url: str,
    dry_run: bool,
) -> None:
    print()
    print("=" * 66)
    print(" KERALA FLOOD AI — DATABASE SEED SUMMARY")
    print("=" * 66)
    print(f" Database : {database_label(database_url)}")
    print(f" Mode     : {'DRY RUN' if dry_run else 'SEEDED'}")
    print("-" * 66)

    for name in DATASETS:
        print(
            f" {name:<31}"
            f"{len(datasets[name]):>8} records"
        )

    print("-" * 66)

    total = sum(
        len(rows)
        for rows in datasets.values()
    )

    print(f" TOTAL{'':<26}{total:>8} records")
    print("=" * 66)
    print(" SYNTHETIC DEVELOPMENT DATA ONLY")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seed a development database with synthetic "
            "Kerala Flood AI datasets."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/demo"),
        help="Directory containing generated JSON datasets",
    )

    parser.add_argument(
        "--database",
        default="sqlite:///data/kerala_flood_demo.db",
        help="SQLite or PostgreSQL development database URL",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input without connecting to a database",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate demo tables",
    )

    parser.add_argument(
        "--confirm-reset",
        action="store_true",
        help="Required confirmation for --reset",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    connection = None

    try:
        validate_database_target(
            args.database,
            args.reset,
            args.confirm_reset,
        )

        print("Loading synthetic datasets...")

        datasets = load_all(
            args.input
        )

        validate_references(
            datasets
        )

        print("Input validation passed.")

        if args.dry_run:
            print_summary(
                datasets,
                args.database,
                dry_run=True,
            )
            return 0

        connection, backend = connect_database(
            args.database
        )

        if args.reset:
            reset_schema(
                connection,
                backend,
            )

        create_schema(
            connection,
            backend,
        )

        seed_all(
            connection,
            backend,
            datasets,
        )

        verify_counts(
            connection,
            backend,
            datasets,
        )

        connection.commit()

        print_summary(
            datasets,
            args.database,
            dry_run=False,
        )

        return 0

    except (
        SeedError,
        sqlite3.Error,
        OSError,
        ValueError,
    ) as exc:
        if connection is not None:
            connection.rollback()

        print(
            f"SEED ERROR: {exc}",
            file=sys.stderr,
        )

        return 1

    except Exception as exc:
        if connection is not None:
            connection.rollback()

        print(
            f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        return 1

    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())

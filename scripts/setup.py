"""Placeholder module: scripts/setup.py; implementation required."""

#!/usr/bin/env python3
"""
Kerala Flood Relief Coordination Agent
backend/scripts/setup.py

Local development setup and bootstrap utility.

Commands:
    python backend/scripts/setup.py
    python backend/scripts/setup.py --check
    python backend/scripts/setup.py --demo-data
    python backend/scripts/setup.py --seed
    python backend/scripts/setup.py --all
    python backend/scripts/setup.py --all --force

Run from any directory.

Python 3.10+
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sqlite3
import subprocess
import sys

from pathlib import Path
from typing import Sequence


PROJECT_NAME = "Kerala Flood Relief Coordination Agent"
MIN_PYTHON = (3, 10)

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

REQUIRED_DIRECTORIES = [
    "backend/api",
    "backend/core",
    "backend/models",
    "backend/schemas",
    "backend/services",
    "backend/agents",
    "backend/adapters",
    "backend/workers",
    "backend/scripts",
    "backend/tests",
    "frontend",
    "data/raw",
    "data/demo",
    "data/processed/geojson",
    "data/simulation",
    "data/uploads",
    "logs",
    "reports",
    "models",
]

DEPENDENCIES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "sqlalchemy": "sqlalchemy",
    "dotenv": "python-dotenv",
    "httpx": "httpx",
}

OPTIONAL_DEPENDENCIES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "langchain": "langchain",
    "langgraph": "langgraph",
    "redis": "redis",
    "psycopg": "psycopg",
}

ENV_TEMPLATE = """# Kerala Flood AI — local development only
APP_NAME=Kerala Flood Relief Coordination Agent
APP_ENV=development
DEBUG=true
HOST=127.0.0.1
PORT=8000

# Demo database
DATABASE_URL=sqlite:///data/kerala_flood_demo.db
ALLOW_DEMO_SEED=1
DEMO_MODE=true

# Security
# Replace these values before building authentication.
SECRET_KEY=CHANGE_ME_BEFORE_REAL_USE
CORS_ORIGINS=http://127.0.0.1:5500,http://localhost:5500

# Optional integrations — disabled by default
ENABLE_LIVE_ALERTS=false
ENABLE_SMS=false
ENABLE_WHATSAPP=false
ENABLE_EMAIL=false
ENABLE_DISPATCH=false
ENABLE_LIVE_WEATHER=false

# Optional credentials
OPENAI_API_KEY=
GEMINI_API_KEY=
REDIS_URL=
"""

GITIGNORE_ENTRIES = """
# Local configuration
.env
.env.*
!.env.example

# Python
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/

# Local databases
*.db
*.sqlite
*.sqlite3

# Generated datasets and uploads
data/demo/
data/simulation/
data/uploads/
logs/
reports/

# Secrets
*.pem
*.key
"""


def print_header(title: str) -> None:
    print()
    print("=" * 65)
    print(title)
    print("=" * 65)


def status(label: str, message: str) -> None:
    print(f"[{label:<5}] {message}")


def check_python() -> bool:
    version = sys.version_info

    if version < MIN_PYTHON:
        status(
            "FAIL",
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required.",
        )
        return False

    status(
        "OK",
        f"Python {version.major}.{version.minor}.{version.micro}",
    )
    return True


def create_directories() -> None:
    print_header("1. CREATE PROJECT DIRECTORIES")

    for relative in REQUIRED_DIRECTORIES:
        directory = PROJECT_ROOT / relative
        directory.mkdir(parents=True, exist_ok=True)
        status("OK", relative)


def write_if_missing(
    path: Path,
    content: str,
    force: bool = False,
) -> None:
    if path.exists() and not force:
        status("SKIP", f"{path.name} already exists")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    status("OK", f"Created {path.relative_to(PROJECT_ROOT)}")


def create_environment(force: bool = False) -> None:
    print_header("2. ENVIRONMENT CONFIGURATION")

    example_path = PROJECT_ROOT / ".env.example"
    env_path = PROJECT_ROOT / ".env"

    write_if_missing(
        example_path,
        ENV_TEMPLATE,
        force=force,
    )

    # Never overwrite existing local secrets.
    write_if_missing(
        env_path,
        ENV_TEMPLATE,
        force=False,
    )

    if "CHANGE_ME_BEFORE_REAL_USE" in env_path.read_text(
        encoding="utf-8"
    ):
        status(
            "WARN",
            "Development SECRET_KEY is a placeholder; "
            "authentication must not use it.",
        )


def configure_gitignore() -> None:
    print_header("3. GITIGNORE")

    path = PROJECT_ROOT / ".gitignore"

    if not path.exists():
        path.write_text(
            GITIGNORE_ENTRIES.lstrip(),
            encoding="utf-8",
        )
        status("OK", "Created .gitignore")
        return

    existing = path.read_text(encoding="utf-8")

    additions = [
        entry
        for entry in GITIGNORE_ENTRIES.splitlines()
        if entry.strip()
        and not entry.startswith("#")
        and entry not in existing.splitlines()
    ]

    if additions:
        with path.open("a", encoding="utf-8") as file:
            file.write(
                "\n# Kerala Flood AI local development\n"
            )
            file.write("\n".join(additions) + "\n")

        status("OK", "Updated .gitignore")
    else:
        status("SKIP", ".gitignore already configured")


def check_dependencies() -> bool:
    print_header("4. DEPENDENCY CHECK")

    missing_required = []

    for module, package in DEPENDENCIES.items():
        if importlib.util.find_spec(module):
            status("OK", package)
        else:
            status("MISS", package)
            missing_required.append(package)

    print("\nOptional packages:")

    for module, package in OPTIONAL_DEPENDENCIES.items():
        if importlib.util.find_spec(module):
            status("OK", package)
        else:
            status("OPT", f"{package} not installed")

    if missing_required:
        print()
        status(
            "INFO",
            "Install required packages using:",
        )
        print(
            "python -m pip install "
            + " ".join(missing_required)
        )

    return not missing_required


def check_project_files() -> None:
    print_header("5. PROJECT FILE CHECK")

    expected = [
        "backend/main.py",
        "backend/scripts/generate_demo_data.py",
        "backend/scripts/import_geojson.py",
        "backend/scripts/run_simulation.py",
        "backend/scripts/seed_database.py",
        "frontend/index.html",
        "frontend/command-center.html",
        "frontend/volunteer-dashboard.html",
        "frontend/volunteer-registration.html",
        "frontend/volunteer-assignment.html",
    ]

    for relative in expected:
        path = PROJECT_ROOT / relative

        if path.is_file():
            status("OK", relative)
        else:
            status("MISS", relative)


def run_script(
    script_name: str,
    arguments: Sequence[str],
) -> bool:
    script = SCRIPT_DIR / script_name

    if not script.is_file():
        status(
            "FAIL",
            f"Required script not found: {script}",
        )
        return False

    command = [
        sys.executable,
        str(script),
        *arguments,
    ]

    print()
    status(
        "RUN",
        f"{script_name} {' '.join(arguments)}",
    )

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        status(
            "FAIL",
            f"{script_name} exceeded the time limit",
        )
        return False
    except OSError as exc:
        status("FAIL", str(exc))
        return False

    if result.returncode != 0:
        status(
            "FAIL",
            f"{script_name} exited with code "
            f"{result.returncode}",
        )
        return False

    status("OK", f"{script_name} completed")
    return True


def generate_demo_data() -> bool:
    print_header("6. GENERATE SYNTHETIC DATA")

    return run_script(
        "generate_demo_data.py",
        [
            "--output",
            str(PROJECT_ROOT / "data/demo"),
            "--seed",
            "42",
            "--format",
            "both",
        ],
    )


def seed_database() -> bool:
    print_header("7. SEED DEVELOPMENT DATABASE")

    return run_script(
        "seed_database.py",
        [
            "--input",
            str(PROJECT_ROOT / "data/demo"),
            "--database",
            "sqlite:///data/kerala_flood_demo.db",
        ],
    )


def run_simulation() -> bool:
    print_header("8. RUN OFFLINE SIMULATION")

    return run_script(
        "run_simulation.py",
        [
            "--input",
            str(PROJECT_ROOT / "data/demo"),
            "--output",
            str(PROJECT_ROOT / "data/simulation"),
            "--scenario",
            "heavy_rainfall",
            "--steps",
            "24",
            "--seed",
            "42",
        ],
    )


def check_database() -> None:
    print_header("9. DATABASE CHECK")

    path = PROJECT_ROOT / "data/kerala_flood_demo.db"

    if not path.exists():
        status("INFO", "Demo database not created yet")
        return

    try:
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro",
            uri=True,
        )

        try:
            cursor = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table'"
            )

            tables = [
                row[0]
                for row in cursor.fetchall()
            ]

            status(
                "OK",
                f"SQLite database: {len(tables)} tables",
            )

            for table in sorted(tables):
                status("TABLE", table)
        finally:
            connection.close()

    except sqlite3.Error as exc:
        status("FAIL", f"Database check failed: {exc}")


def show_next_steps() -> None:
    print_header("NEXT STEPS")

    print(
        "1. Install backend dependencies:\n"
        "   python -m pip install fastapi uvicorn "
        "pydantic sqlalchemy python-dotenv httpx\n"
    )

    print(
        "2. Start FastAPI after implementing backend/main.py:\n"
        "   python -m uvicorn backend.main:app --reload\n"
    )

    print(
        "3. Serve the frontend from the project root:\n"
        "   python -m http.server 5500 --directory frontend\n"
    )

    print(
        "4. Open:\n"
        "   http://127.0.0.1:5500/index.html\n"
    )

    print(
        "5. Review generated data:\n"
        "   data/demo/\n"
        "   data/simulation/\n"
        "   data/kerala_flood_demo.db\n"
    )

    print(
        "All datasets are fictional. "
        "No live emergency integrations are enabled."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap Kerala Flood AI local development."
        )
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Python, packages and project files only",
    )

    parser.add_argument(
        "--demo-data",
        action="store_true",
        help="Generate synthetic demo datasets",
    )

    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed SQLite development database",
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run the offline simulation",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run complete local setup pipeline",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh .env.example (never overwrites .env)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_header(PROJECT_NAME.upper())
    print(f"Project root: {PROJECT_ROOT}")

    if not check_python():
        return 1

    if args.check:
        dependencies_ok = check_dependencies()
        check_project_files()
        check_database()
        return 0 if dependencies_ok else 1

    create_directories()
    create_environment(force=args.force)
    configure_gitignore()

    dependencies_ok = check_dependencies()
    check_project_files()

    do_generate = args.all or args.demo_data
    do_seed = args.all or args.seed
    do_simulate = args.all or args.simulate

    demo_path = PROJECT_ROOT / "data/demo"

    if do_generate:
        if not generate_demo_data():
            return 1

    if do_seed:
        if not (
            demo_path / "districts.json"
        ).is_file():
            status(
                "FAIL",
                "Demo data missing. Run --demo-data first.",
            )
            return 1

        if not seed_database():
            return 1

    if do_simulate:
        if not (
            demo_path / "districts.json"
        ).is_file():
            status(
                "FAIL",
                "Demo data missing. Run --demo-data first.",
            )
            return 1

        if not run_simulation():
            return 1

    check_database()
    show_next_steps()

    if not dependencies_ok:
        status(
            "WARN",
            "Install missing backend packages "
            "before starting the API.",
        )

    status(
        "OK",
        "Local setup workflow finished",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

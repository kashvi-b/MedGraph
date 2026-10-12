#!/usr/bin/env python
"""Apply scripts/init.sql to DATABASE_URL.

A Python equivalent of `psql -f scripts/init.sql`, because psql is a separate
install that a Python project should not require just to create its own schema.
Every statement is idempotent, so this is safe to re-run.

    python scripts/init_db.py            # create the schema
    python scripts/init_db.py --check    # report what exists, change nothing
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCHEMA = ROOT / "scripts" / "init.sql"


def load_env() -> None:
    """Read .env without adding a runtime dependency on python-dotenv."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def check(cur) -> None:
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    row = cur.fetchone()
    print(f"  pgvector           {row[0] if row else 'NOT INSTALLED'}")

    for table in ("documents", "runs"):
        cur.execute("SELECT to_regclass(%s)", (table,))
        exists = cur.fetchone()[0] is not None
        if exists:
            cur.execute(f"SELECT count(*) FROM {table}")
            print(f"  {table:<18} exists, {cur.fetchone()[0]:,} rows")
        else:
            print(f"  {table:<18} missing")

    cur.execute("""
        SELECT a.atttypmod
        FROM pg_attribute a
        WHERE a.attrelid = 'documents'::regclass AND a.attname = 'embedding'
    """)
    row = cur.fetchone()
    if row:
        print(f"  embedding dims     {row[0]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report current state without applying the schema")
    args = ap.parse_args()

    load_env()
    import medgraph.config  # noqa: F401  (loads .env as a side effect too)

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set (put it in .env)", file=sys.stderr)
        return 1

    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT version()")
        print(f"  connected          {cur.fetchone()[0].split(',')[0]}")

        if not args.check:
            cur.execute(SCHEMA.read_text())
            conn.commit()
            print(f"  applied            {SCHEMA.relative_to(ROOT)}")

        check(cur)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

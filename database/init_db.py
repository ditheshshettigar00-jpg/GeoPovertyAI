"""
init_db.py
----------
Creates the SQLite database (geopoverty.db) from schema.sql.
Run this once before using any other module.

Usage:
    python database/init_db.py
"""

import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")
DB_PATH = os.path.join(BASE_DIR, "geopoverty.db")


def init_db(db_path: str = DB_PATH, schema_path: str = SCHEMA_PATH) -> None:
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema_sql)
        conn.commit()
        print(f"Database initialized successfully at: {db_path}")

        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cur.fetchall()]
        print("Tables created:", tables)
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()

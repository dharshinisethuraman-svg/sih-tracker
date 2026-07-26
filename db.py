"""
Task 2 - database layer.
SQLite is used for simplicity (single-file DB, zero setup), as suggested
in the assessment's tools list.
"""
import sqlite3
import os
import csv

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    course           TEXT NOT NULL,
    batch            TEXT NOT NULL,
    enrolment_date   TEXT NOT NULL,
    attendance_pct   REAL,
    completed        INTEGER,
    placement_status TEXT NOT NULL DEFAULT 'Unknown',
    employer         TEXT
);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def seed_from_csv(csv_path):
    """Loads sample_data.csv into the DB, skipping rows already present.
    Skips the deliberately junk row (candidate_id 99, all blank) because
    a real coordinator screen would never submit it - it exists in the
    CSV to test search/validation, not to be treated as real data."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM candidates")
    if cur.fetchone()["c"] > 0:
        conn.close()
        return  # already seeded

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["name"].strip():
                continue  # skip the junk row on purpose
            attendance = row["attendance_pct"] or None
            completed = row["completed"] or None
            cur.execute(
                """INSERT INTO candidates
                   (candidate_id, name, course, batch, enrolment_date,
                    attendance_pct, completed, placement_status, employer)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (row["candidate_id"], row["name"], row["course"],
                 row["batch"], row["enrolment_date"],
                 float(attendance) if attendance not in (None, "") else None,
                 int(completed) if completed not in (None, "") else None,
                 row["placement_status"] or "Unknown", row["employer"]),
            )
    conn.commit()
    conn.close()

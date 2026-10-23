"""SQLite history. Stores scores and file name only, never the image."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT NOT NULL,
  label TEXT NOT NULL,
  anomaly_score REAL,
  cdr REAL,
  fused_score REAL,
  confidence_level TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
)
"""


def init_db(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as con:
        con.execute(SCHEMA)
        con.commit()


def add_prediction(path: Path, filename: str, result: dict) -> int:
    with closing(sqlite3.connect(path)) as con:
        cur = con.execute(
            "INSERT INTO predictions (filename,label,anomaly_score,cdr,fused_score,confidence_level) VALUES (?,?,?,?,?,?)",
            (filename[:200], result["label"], result["anomaly_score"], result["cdr"], result["fused_score"],
             result["confidence_level"]),
        )
        con.commit()
        return int(cur.lastrowid)


def recent(path: Path, limit: int = 50) -> list[dict]:
    with closing(sqlite3.connect(path)) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

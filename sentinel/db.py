"""
SQLite persistence for Sentinel.

Hybrid schema: real columns for the fields we filter/update on, plus one
`signals_json` TEXT column holding the full nested per-signal breakdown (we
only ever read that back whole). One .db file in the repo => zero setup,
reproducible, safe for a live demo.

Access pattern is read-filter-update:
  - ingest a batch under a session_id
  - the pipeline updates credibility_score / decision / brief after judging
  - the swipe app writes human_verdict back for escalated cases
  - the confusion-matrix script queries by session_id
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from .contract import CaseRecord, DECISION_ESCALATE

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sentinel.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id           TEXT NOT NULL,
    session_id        TEXT NOT NULL,
    return_reason     TEXT,
    credibility_score REAL,
    decision          TEXT,          -- approve | reject | escalate
    runtime_ms        INTEGER,
    created_at        TEXT NOT NULL,
    human_verdict     TEXT,          -- filled by the swipe app
    true_label        TEXT,          -- filled if ground truth exists
    signals_json      TEXT,          -- full nested breakdown (read whole only)
    UNIQUE(session_id, case_id)
);
CREATE INDEX IF NOT EXISTS idx_cases_session ON cases(session_id);
CREATE INDEX IF NOT EXISTS idx_cases_decision ON cases(session_id, decision);

-- Running perceptual-hash store for Rung-0 duplicate/reuse detection. Every piece
-- of media that clears decode gets its pHash recorded here; a new submission whose
-- pHash is near-identical to an EARLIER case's is reused proof -> auto-reject.
-- Hamming distance can't be expressed in SQL cheaply, so we load candidate hashes
-- and compare in Python (fine at hackathon scale). phash is a 64-bit hex string.
CREATE TABLE IF NOT EXISTS media_hashes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    filename    TEXT NOT NULL,
    phash       TEXT NOT NULL,     -- 64-bit perceptual hash, hex
    created_at  TEXT NOT NULL,
    UNIQUE(session_id, case_id, filename)
);
CREATE INDEX IF NOT EXISTS idx_hashes_lookup ON media_hashes(session_id);
"""


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_case(conn: sqlite3.Connection, rec: CaseRecord) -> None:
    """Insert or replace a case (unique per session_id + case_id)."""
    conn.execute(
        """
        INSERT INTO cases (
            case_id, session_id, return_reason, credibility_score, decision,
            runtime_ms, created_at, human_verdict, true_label, signals_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, case_id) DO UPDATE SET
            return_reason     = excluded.return_reason,
            credibility_score = excluded.credibility_score,
            decision          = excluded.decision,
            runtime_ms        = excluded.runtime_ms,
            human_verdict     = excluded.human_verdict,
            true_label        = excluded.true_label,
            signals_json      = excluded.signals_json
        """,
        (
            rec.case_id, rec.session_id, rec.return_reason, rec.credibility_score,
            rec.decision, rec.runtime_ms, rec.created_at, rec.human_verdict,
            rec.true_label, rec.signals_json(),
        ),
    )
    conn.commit()


def upsert_many(conn: sqlite3.Connection, recs: Iterable[CaseRecord]) -> int:
    n = 0
    for rec in recs:
        upsert_case(conn, rec)
        n += 1
    return n


def set_human_verdict(conn: sqlite3.Connection, session_id: str, case_id: str,
                      verdict: str) -> None:
    """Swipe app writes the human decision back onto an escalated case."""
    conn.execute(
        "UPDATE cases SET human_verdict = ? WHERE session_id = ? AND case_id = ?",
        (verdict, session_id, case_id),
    )
    conn.commit()


def get_case(conn: sqlite3.Connection, session_id: str,
             case_id: str) -> Optional[CaseRecord]:
    row = conn.execute(
        "SELECT * FROM cases WHERE session_id = ? AND case_id = ?",
        (session_id, case_id),
    ).fetchone()
    return CaseRecord.from_row(dict(row)) if row else None


def get_cases(conn: sqlite3.Connection, session_id: str,
              decision: Optional[str] = None) -> list[CaseRecord]:
    if decision is None:
        rows = conn.execute(
            "SELECT * FROM cases WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM cases WHERE session_id = ? AND decision = ? ORDER BY id",
            (session_id, decision),
        ).fetchall()
    return [CaseRecord.from_row(dict(r)) for r in rows]


def get_escalated(conn: sqlite3.Connection, session_id: str) -> list[CaseRecord]:
    """Cases the swipe app should show (escalated + not yet judged by a human)."""
    rows = conn.execute(
        """
        SELECT * FROM cases
        WHERE session_id = ? AND decision = ? AND human_verdict IS NULL
        ORDER BY credibility_score ASC
        """,
        (session_id, DECISION_ESCALATE),
    ).fetchall()
    return [CaseRecord.from_row(dict(r)) for r in rows]


def record_phash(conn: sqlite3.Connection, session_id: str, case_id: str,
                 filename: str, phash: str) -> None:
    """Add one media file's pHash to the running store (idempotent per file)."""
    from datetime import datetime, timezone
    conn.execute(
        """
        INSERT INTO media_hashes (session_id, case_id, filename, phash, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id, case_id, filename) DO UPDATE SET phash = excluded.phash
        """,
        (session_id, case_id, filename, phash,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def find_duplicate_phash(conn: sqlite3.Connection, phash: str, session_id: str,
                         exclude_case_id: str, max_distance: int) -> Optional[dict]:
    """Return the nearest EARLIER media whose pHash is within max_distance, or None.

    Reused/near-identical proof across different cases is a strong, cheap, purely
    deterministic fraud signal. We compare against hashes from the same session
    (the batch being judged) but a DIFFERENT case_id, so re-scoring the same case
    never self-matches. Hamming distance is computed in Python.
    """
    target = int(phash, 16)
    rows = conn.execute(
        "SELECT case_id, filename, phash FROM media_hashes WHERE session_id = ? AND case_id != ?",
        (session_id, exclude_case_id),
    ).fetchall()
    best: Optional[dict] = None
    for r in rows:
        dist = bin(target ^ int(r["phash"], 16)).count("1")
        if dist <= max_distance and (best is None or dist < best["distance"]):
            best = {"case_id": r["case_id"], "filename": r["filename"], "distance": dist}
    return best


def list_sessions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT session_id, COUNT(*) AS n, MIN(created_at) AS started
        FROM cases GROUP BY session_id ORDER BY started DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]

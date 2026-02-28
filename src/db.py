"""
Tecnomate Clinical AI — SQLite Persistence Layer
=================================================
Replaces the ephemeral in-memory `_feedback_queue` list with a proper
SQLite database so that:

  - Feedback survives API server restarts
  - Admins can query, filter, and export the full audit trail
  - Multiple workers / processes share the same store

Schema
------
  feedback
    id              INTEGER  PRIMARY KEY AUTOINCREMENT
    session_id      TEXT     NOT NULL
    chosen_key      TEXT     NOT NULL
    chosen_label    TEXT
    ai_predicted_key TEXT
    scan_type       TEXT
    saved_to        TEXT
    timestamp       TEXT     NOT NULL   (ISO-8601)
    overridden      INTEGER  NOT NULL   (0 / 1)
    clinician_name  TEXT
    clinician_id    TEXT
    notes           TEXT

  sessions
    session_id      TEXT     PRIMARY KEY
    filename        TEXT
    scan_type       TEXT
    ai_pred_key     TEXT
    ai_confidence   REAL
    probabilities   TEXT     (JSON blob)
    ocr_text        TEXT
    file_size_bytes INTEGER
    image_width     INTEGER
    image_height    INTEGER
    detected_format TEXT
    created_at      TEXT     NOT NULL   (ISO-8601)

Usage
-----
    from db import FeedbackDB
    db = FeedbackDB()          # opens / creates tecnomate.db in data/
    db.add_feedback(entry)
    rows = db.list_feedback(limit=100, overridden_only=False)
    db.add_session(session_data)
    session = db.get_session(session_id)
    db.close()

    # Context-manager usage
    with FeedbackDB() as db:
        db.add_feedback(entry)
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── DB location ───────────────────────────────────────────────────────────────
_PROJ_ROOT = Path(__file__).parent.parent
_DB_PATH = _PROJ_ROOT / "data" / "tecnomate.db"

# DDL ─────────────────────────────────────────────────────────────────────────
_CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT    NOT NULL,
    chosen_key       TEXT    NOT NULL,
    chosen_label     TEXT    DEFAULT '',
    ai_predicted_key TEXT    DEFAULT '',
    scan_type        TEXT    DEFAULT '',
    saved_to         TEXT    DEFAULT '',
    timestamp        TEXT    NOT NULL,
    overridden       INTEGER NOT NULL DEFAULT 0,
    clinician_name   TEXT    DEFAULT '',
    clinician_id     TEXT    DEFAULT '',
    notes            TEXT    DEFAULT ''
);
"""

_CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT    PRIMARY KEY,
    filename        TEXT    DEFAULT '',
    scan_type       TEXT    DEFAULT '',
    ai_pred_key     TEXT    DEFAULT '',
    ai_confidence   REAL    DEFAULT 0.0,
    probabilities   TEXT    DEFAULT '{}',
    ocr_text        TEXT    DEFAULT '',
    file_size_bytes INTEGER DEFAULT 0,
    image_width     INTEGER DEFAULT 0,
    image_height    INTEGER DEFAULT 0,
    detected_format TEXT    DEFAULT '',
    created_at      TEXT    NOT NULL
);
"""

_CREATE_FEEDBACK_IDX = """
CREATE INDEX IF NOT EXISTS idx_feedback_session
    ON feedback (session_id);
"""

_CREATE_FEEDBACK_TS_IDX = """
CREATE INDEX IF NOT EXISTS idx_feedback_ts
    ON feedback (timestamp);
"""

_CREATE_SESSIONS_IDX = """
CREATE INDEX IF NOT EXISTS idx_sessions_created
    ON sessions (created_at);
"""

_ALL_DDL = [
    _CREATE_FEEDBACK_TABLE,
    _CREATE_SESSIONS_TABLE,
    _CREATE_FEEDBACK_IDX,
    _CREATE_FEEDBACK_TS_IDX,
    _CREATE_SESSIONS_IDX,
]


# ═════════════════════════════════════════════════════════════════════════════
#  FeedbackDB
# ═════════════════════════════════════════════════════════════════════════════


class FeedbackDB:
    """
    Thread-safe wrapper around the SQLite feedback/session database.

    All public methods acquire a per-instance lock so that multiple
    FastAPI async workers (run in the same process via uvicorn) cannot
    corrupt the database.  SQLite itself is opened with
    ``check_same_thread=False`` because FastAPI dispatches to a thread
    pool for sync endpoints.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path) if db_path else _DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            timeout=15,
            isolation_level=None,  # autocommit; we manage transactions manually
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._bootstrap()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        """Create tables and indexes if they do not yet exist."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN;")
            try:
                for ddl in _ALL_DDL:
                    cur.execute(ddl)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "FeedbackDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now().isoformat()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        # Deserialise JSON blobs
        for key in ("probabilities",):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[key] = {}
        # Convert SQLite integer booleans back to Python booleans
        for key in ("overridden",):
            if key in d:
                d[key] = bool(d[key])
        return d

    # ── feedback CRUD ──────────────────────────────────────────────────────────

    def add_feedback(self, entry: Dict[str, Any]) -> int:
        """
        Insert a feedback record.  The dict should contain the same keys
        that were previously appended to ``_feedback_queue``.

        Returns the newly inserted row id.
        """
        sql = """
            INSERT INTO feedback (
                session_id, chosen_key, chosen_label, ai_predicted_key,
                scan_type, saved_to, timestamp, overridden,
                clinician_name, clinician_id, notes
            ) VALUES (
                :session_id, :chosen_key, :chosen_label, :ai_predicted_key,
                :scan_type, :saved_to, :timestamp, :overridden,
                :clinician_name, :clinician_id, :notes
            );
        """
        record = {
            "session_id": entry.get("session_id", ""),
            "chosen_key": entry.get("chosen_key", ""),
            "chosen_label": entry.get("chosen_label", ""),
            "ai_predicted_key": entry.get("ai_predicted_key", ""),
            "scan_type": entry.get("scan_type", ""),
            "saved_to": entry.get("saved_to", ""),
            "timestamp": entry.get("timestamp", self._now()),
            "overridden": int(bool(entry.get("overridden", False))),
            "clinician_name": entry.get("clinician_name", ""),
            "clinician_id": entry.get("clinician_id", ""),
            "notes": entry.get("notes", ""),
        }
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN;")
            try:
                cur.execute(sql, record)
                row_id = cur.lastrowid or 0
                self._conn.commit()
                return row_id
            except Exception:
                self._conn.rollback()
                raise

    def list_feedback(
        self,
        limit: int = 50,
        offset: int = 0,
        overridden_only: bool = False,
        scan_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return feedback records, newest first.

        Parameters
        ----------
        limit          : max rows returned (default 50, max 1000)
        offset         : pagination offset
        overridden_only: if True only return rows where the clinician disagreed
        scan_type      : filter by scan type ("Brain MRI" or "Chest X-Ray")
        """
        limit = min(max(1, limit), 1000)

        conditions = []
        params: Dict[str, Any] = {"limit": limit, "offset": offset}

        if overridden_only:
            conditions.append("overridden = 1")
        if scan_type:
            conditions.append("scan_type = :scan_type")
            params["scan_type"] = scan_type

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT * FROM feedback
            {where}
            ORDER BY timestamp DESC
            LIMIT :limit OFFSET :offset;
        """
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [self._row_to_dict(r) for r in cur.fetchall()]

    def count_feedback(
        self,
        overridden_only: bool = False,
        scan_type: Optional[str] = None,
    ) -> Dict[str, int]:
        """Return total, confirmed, and overridden counts."""
        params: Dict[str, Any] = {}
        conditions: List[str] = []

        if scan_type:
            conditions.append("scan_type = :scan_type")
            params["scan_type"] = scan_type

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql_total = f"SELECT COUNT(*) FROM feedback {where};"
        sql_overridden = f"SELECT COUNT(*) FROM feedback {where} {'AND' if where else 'WHERE'} overridden = 1;"

        with self._lock:
            total = self._conn.execute(sql_total, params).fetchone()[0]
            overridden = self._conn.execute(sql_overridden, params).fetchone()[0]

        return {
            "total": int(total),
            "confirmed": int(total - overridden),
            "overridden": int(overridden),
        }

    def get_feedback_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the feedback record for a specific session, if any."""
        sql = "SELECT * FROM feedback WHERE session_id = ? ORDER BY id DESC LIMIT 1;"
        with self._lock:
            row = self._conn.execute(sql, (session_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def export_feedback_csv(self) -> str:
        """
        Dump the entire feedback table as a CSV string (for admin download).
        """
        import csv
        import io as _io

        with self._lock:
            cur = self._conn.execute("SELECT * FROM feedback ORDER BY timestamp DESC;")
            rows = cur.fetchall()

        if not rows:
            return "id,session_id,chosen_key,chosen_label,ai_predicted_key,scan_type,saved_to,timestamp,overridden,clinician_name,clinician_id,notes\n"

        buf = _io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([desc[0] for desc in cur.description])
        for row in rows:
            writer.writerow(list(row))
        return buf.getvalue()

    # ── session CRUD ───────────────────────────────────────────────────────────

    def add_session(self, data: Dict[str, Any]) -> None:
        """
        Insert (or replace) a prediction session record.

        This is called immediately after a successful /api/predict response
        so that the full prediction context can be retrieved when building
        the PDF report.
        """
        sql = """
            INSERT OR REPLACE INTO sessions (
                session_id, filename, scan_type, ai_pred_key,
                ai_confidence, probabilities, ocr_text,
                file_size_bytes, image_width, image_height,
                detected_format, created_at
            ) VALUES (
                :session_id, :filename, :scan_type, :ai_pred_key,
                :ai_confidence, :probabilities, :ocr_text,
                :file_size_bytes, :image_width, :image_height,
                :detected_format, :created_at
            );
        """
        dims = data.get("image_dimensions", [0, 0])
        record = {
            "session_id": data.get("session_id", ""),
            "filename": data.get("filename", ""),
            "scan_type": data.get("scan_type", ""),
            "ai_pred_key": data.get(
                "mode_predicted_key", data.get("predicted_key", "")
            ),
            "ai_confidence": float(
                data.get("mode_confidence", data.get("confidence", 0.0))
            ),
            "probabilities": json.dumps(
                data.get("mode_probabilities", data.get("probabilities", {}))
            ),
            "ocr_text": data.get("ocr_text", ""),
            "file_size_bytes": int(data.get("file_size_bytes", 0)),
            "image_width": int(dims[0]) if len(dims) > 0 else 0,
            "image_height": int(dims[1]) if len(dims) > 1 else 0,
            "detected_format": data.get("detected_format", ""),
            "created_at": self._now(),
        }
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN;")
            try:
                cur.execute(sql, record)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored session record by ID."""
        sql = "SELECT * FROM sessions WHERE session_id = ? LIMIT 1;"
        with self._lock:
            row = self._conn.execute(sql, (session_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return recent prediction sessions, newest first."""
        limit = min(max(1, limit), 1000)
        sql = "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?;"
        with self._lock:
            cur = self._conn.execute(sql, (limit, offset))
            return [self._row_to_dict(r) for r in cur.fetchall()]

    def purge_old_sessions(self, keep_days: int = 30) -> int:
        """
        Delete session records older than `keep_days` days.
        Returns the number of rows deleted.
        """
        cutoff = (
            datetime.datetime.now() - datetime.timedelta(days=keep_days)
        ).isoformat()
        sql = "DELETE FROM sessions WHERE created_at < ?;"
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN;")
            try:
                cur.execute(sql, (cutoff,))
                deleted = cur.rowcount
                self._conn.commit()
                return deleted
            except Exception:
                self._conn.rollback()
                raise

    # ── statistics ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return aggregate statistics used by the admin dashboard."""
        with self._lock:
            total_sessions = self._conn.execute(
                "SELECT COUNT(*) FROM sessions;"
            ).fetchone()[0]
            total_feedback = self._conn.execute(
                "SELECT COUNT(*) FROM feedback;"
            ).fetchone()[0]
            total_overridden = self._conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE overridden = 1;"
            ).fetchone()[0]

            # Breakdown by class
            class_rows = self._conn.execute(
                """
                SELECT chosen_key, COUNT(*) as cnt
                FROM feedback
                GROUP BY chosen_key
                ORDER BY cnt DESC;
                """
            ).fetchall()

            # Breakdown by scan type
            scan_rows = self._conn.execute(
                """
                SELECT scan_type, COUNT(*) as cnt
                FROM feedback
                GROUP BY scan_type
                ORDER BY cnt DESC;
                """
            ).fetchall()

            # Recent 7 days activity
            seven_days_ago = (
                datetime.datetime.now() - datetime.timedelta(days=7)
            ).isoformat()
            recent_7d = self._conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE timestamp >= ?;",
                (seven_days_ago,),
            ).fetchone()[0]

        return {
            "total_sessions": int(total_sessions),
            "total_feedback": int(total_feedback),
            "total_confirmed": int(total_feedback - total_overridden),
            "total_overridden": int(total_overridden),
            "override_rate_pct": (
                round(total_overridden / total_feedback * 100, 2)
                if total_feedback > 0
                else 0.0
            ),
            "feedback_by_class": {r["chosen_key"]: r["cnt"] for r in class_rows},
            "feedback_by_scan_type": {r["scan_type"]: r["cnt"] for r in scan_rows},
            "feedback_last_7_days": int(recent_7d),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  Module-level singleton (shared by all FastAPI routes)
# ═════════════════════════════════════════════════════════════════════════════

_db_instance: Optional[FeedbackDB] = None
_db_lock = threading.Lock()


def get_db() -> FeedbackDB:
    """
    Return the process-wide FeedbackDB singleton, creating it on first call.

    This mirrors the get_model() pattern in api.py and is safe to call from
    any FastAPI route without any setup boilerplate.
    """
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = FeedbackDB()
    return _db_instance

"""SQLite access with versioned, non-destructive migrations."""
from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

# (version, name, [statements]). Never edit a shipped migration; append a new one.
MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (1, "jobs_stages_errors", [
        """CREATE TABLE jobs (
            id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            input_ref TEXT,
            language TEXT,
            format TEXT,
            status TEXT NOT NULL,
            stage TEXT,
            params_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT)""",
        """CREATE TABLE stages (
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT, finished_at TEXT,
            artifacts_json TEXT, error TEXT,
            PRIMARY KEY (job_id, name))""",
        """CREATE TABLE errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT, stage TEXT, message TEXT, traceback TEXT, created_at TEXT NOT NULL)""",
    ]),
    (2, "research_assets_transcripts_highlights", [
        """CREATE TABLE research_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            url TEXT, title TEXT, publisher TEXT, published TEXT, retrieved_at TEXT,
            license TEXT, extract TEXT)""",
        """CREATE TABLE research_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            source_id INTEGER REFERENCES research_sources(id),
            text TEXT NOT NULL, kind TEXT, confidence TEXT, supported INTEGER DEFAULT 1, flag TEXT)""",
        """CREATE TABLE assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            kind TEXT, path TEXT, source TEXT, license TEXT, license_status TEXT,
            section TEXT, meta_json TEXT)""",
        """CREATE TABLE transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            engine TEXT, language TEXT, path TEXT, timing TEXT, verified_timing INTEGER,
            meta_json TEXT, created_at TEXT)""",
        """CREATE TABLE highlights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            start_s REAL NOT NULL, end_s REAL NOT NULL, score REAL,
            scores_json TEXT, text TEXT, status TEXT NOT NULL DEFAULT 'candidate',
            hook_text TEXT, order_idx INTEGER, user_edited INTEGER DEFAULT 0,
            warnings_json TEXT)""",
    ]),
    (3, "outputs_qa_metadata", [
        """CREATE TABLE outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            kind TEXT, path TEXT, sha256 TEXT, ref_id INTEGER, meta_json TEXT, created_at TEXT)""",
        """CREATE TABLE qa_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            target TEXT, passed INTEGER, checks_json TEXT, created_at TEXT)""",
        """CREATE TABLE metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            target TEXT, json TEXT, created_at TEXT)""",
        "ALTER TABLE jobs ADD COLUMN worker_pid INTEGER",
        "CREATE INDEX idx_highlights_job ON highlights(job_id)",
        "CREATE INDEX idx_outputs_job ON outputs(job_id)",
    ]),
    (4, "task_queue", [
        """CREATE TABLE task_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            action TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
            enqueued_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT)""",
    ]),
]

LATEST_VERSION = MIGRATIONS[-1][0]


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)  # manual transactions
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    return conn


def current_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations ("
                 "version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT)")
    row = conn.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
    return row["v"] or 0


def _backup(db_path: Path, backup_dir: Path, version: int) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"{db_path.stem}_v{version}_{time.strftime('%Y%m%d-%H%M%S')}{db_path.suffix}"
    n = 1
    while dest.exists():
        n += 1
        dest = dest.with_name(f"{dest.stem}_{n}{dest.suffix}")
    shutil.copyfile(db_path, dest)
    return dest


def migrate(conn: sqlite3.Connection, db_path: str | Path | None = None,
            backup_dir: str | Path | None = None, target: int | None = None) -> list[int]:
    """Apply pending migrations in order. Backs up an existing non-empty DB first.

    Each migration runs in its own transaction; a failure rolls back that step and
    leaves earlier steps (and all existing data) intact.
    """
    target = target or LATEST_VERSION
    have = current_version(conn)
    pending = [m for m in MIGRATIONS if have < m[0] <= target]
    if not pending:
        return []
    if have > 0 and db_path and backup_dir:
        conn.execute("PRAGMA wal_checkpoint(FULL)")
        _backup(Path(db_path), Path(backup_dir), have)
    applied = []
    for version, name, statements in pending:
        conn.execute("BEGIN")
        try:
            for stmt in statements:
                conn.execute(stmt)
            conn.execute("INSERT INTO schema_migrations(version, name, applied_at) VALUES (?,?,datetime('now'))",
                         (version, name))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        applied.append(version)
    return applied


def open_database(db_path: str | Path, backup_dir: str | Path | None = None) -> sqlite3.Connection:
    conn = connect(db_path)
    migrate(conn, db_path=db_path, backup_dir=backup_dir)
    return conn

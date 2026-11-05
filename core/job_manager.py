"""Job lifecycle: creation, status, stage checkpoints, records, job directories."""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import resolve_path
from .logging_manager import attach_job_log, detach_job_log, get_logger
from .safe_io import atomic_write_json, sha256_file

log = get_logger("jobs")

JOB_TYPES = ("topic_to_video", "long_form_to_shorts")
STATUSES = ("created", "research_required", "processing", "transcription_required",
            "review_required", "rendering", "qa_failed", "completed", "failed", "interrupted")
JOB_SUBDIRS = ("source", "research", "scripts", "narration", "transcripts", "subtitles", "assets",
               "previews", "final_video", "shorts", "thumbnails", "metadata", "reports")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StageFailed(Exception):
    """A stage cannot continue; carries the status the job should move to."""

    def __init__(self, message: str, status: str = "failed"):
        super().__init__(message)
        self.status = status


class JobManager:
    def __init__(self, conn: sqlite3.Connection, settings: dict):
        self.conn = conn
        self.settings = settings
        self.outputs_root = resolve_path(settings, "outputs")

    # ---- jobs -----------------------------------------------------------------
    def create_job(self, job_type: str, input_ref: str, language: str | None = None,
                   fmt: str | None = None, params: dict | None = None) -> str:
        if job_type not in JOB_TYPES:
            raise ValueError(f"unknown job type {job_type!r}; expected one of {JOB_TYPES}")
        if language and language not in self.settings["languages"]:
            raise ValueError(f"unsupported language {language!r}; supported: {self.settings['languages']}")
        job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
        ts = now_iso()
        self.conn.execute(
            "INSERT INTO jobs(id, job_type, input_ref, language, format, status, stage, params_json,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (job_id, job_type, input_ref, language, fmt, "created", None,
             json.dumps(params or {}, ensure_ascii=False), ts, ts))
        self.job_dir(job_id)
        log.info("created job %s (%s)", job_id, job_type)
        return job_id

    def get_job(self, job_id: str) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such job: {job_id}")
        return row

    def params(self, job_id: str) -> dict:
        return json.loads(self.get_job(job_id)["params_json"] or "{}")

    def update_params(self, job_id: str, **changes) -> dict:
        p = self.params(job_id)
        p.update(changes)
        self.conn.execute("UPDATE jobs SET params_json=?, updated_at=? WHERE id=?",
                          (json.dumps(p, ensure_ascii=False), now_iso(), job_id))
        return p

    def list_jobs(self, status: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
        if status:
            return self.conn.execute("SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ?",
                                     (status, limit)).fetchall()
        return self.conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()

    def set_status(self, job_id: str, status: str, stage: str | None = None,
                   error: str | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        ts = now_iso()
        done = ts if status == "completed" else None
        pid = os.getpid() if status in ("processing", "rendering") else None
        self.conn.execute(
            "UPDATE jobs SET status=?, stage=COALESCE(?, stage), error=?, updated_at=?, "
            "completed_at=COALESCE(?, completed_at), worker_pid=? WHERE id=?",
            (status, stage, error, ts, done, pid, job_id))
        log.info("job %s -> %s%s", job_id, status, f" [{stage}]" if stage else "")

    def job_dir(self, job_id: str) -> Path:
        d = self.outputs_root / job_id
        for sub in JOB_SUBDIRS:
            (d / sub).mkdir(parents=True, exist_ok=True)
        return d

    # ---- stages / checkpoints -------------------------------------------------
    def stage_row(self, job_id: str, name: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM stages WHERE job_id=? AND name=?", (job_id, name)).fetchone()

    def stage_done(self, job_id: str, name: str) -> bool:
        """A stage counts as done only if it completed AND its recorded artifacts still exist."""
        row = self.stage_row(job_id, name)
        if not row or row["status"] != "done":
            return False
        for p in json.loads(row["artifacts_json"] or "[]"):
            if not Path(p).exists():
                return False
        return True

    def run_stage(self, job_id: str, name: str, fn: Callable[[], list[str] | None],
                  status: str = "processing", force: bool = False) -> bool:
        """Run fn() as a checkpointed stage. Returns True if it ran, False if skipped.

        fn returns the list of artifact paths it produced (used to validate resumes).
        """
        if not force and self.stage_done(job_id, name):
            log.info("job %s stage %s already complete; skipping", job_id, name)
            return False
        self.set_status(job_id, status, stage=name)
        self.conn.execute(
            "INSERT INTO stages(job_id,name,status,started_at) VALUES (?,?,?,?) "
            "ON CONFLICT(job_id,name) DO UPDATE SET status='running', started_at=excluded.started_at,"
            " finished_at=NULL, error=NULL", (job_id, name, "running", now_iso()))
        try:
            artifacts = fn() or []
        except StageFailed as e:
            self._stage_error(job_id, name, str(e), None)
            self.set_status(job_id, e.status, stage=name, error=str(e))
            raise
        except BaseException as e:  # noqa: BLE001 - recorded then re-raised
            tb = traceback.format_exc()
            self._stage_error(job_id, name, f"{type(e).__name__}: {e}", tb)
            self.set_status(job_id, "failed", stage=name, error=f"{type(e).__name__}: {e}")
            raise
        self.conn.execute("UPDATE stages SET status='done', finished_at=?, artifacts_json=? "
                          "WHERE job_id=? AND name=?",
                          (now_iso(), json.dumps([str(a) for a in artifacts]), job_id, name))
        return True

    def _stage_error(self, job_id: str, name: str, message: str, tb: str | None) -> None:
        self.conn.execute("UPDATE stages SET status='failed', finished_at=?, error=? WHERE job_id=? AND name=?",
                          (now_iso(), message, job_id, name))
        self.conn.execute("INSERT INTO errors(job_id, stage, message, traceback, created_at) VALUES (?,?,?,?,?)",
                          (job_id, name, message, tb, now_iso()))
        log.error("job %s stage %s failed: %s", job_id, name, message)

    def record_error(self, job_id: str, stage: str, message: str, tb: str | None = None) -> None:
        self.conn.execute("INSERT INTO errors(job_id, stage, message, traceback, created_at) VALUES (?,?,?,?,?)",
                          (job_id, stage, message, tb, now_iso()))

    def job_log(self, job_id: str):
        """Context manager-ish: returns (handler) attached to the per-job log file."""
        return attach_job_log(self.job_dir(job_id), job_id)

    def end_job_log(self, handler) -> None:
        detach_job_log(handler)

    # ---- generic records ------------------------------------------------------
    def add_output(self, job_id: str, kind: str, path: str | Path, ref_id: int | None = None,
                   meta: dict | None = None, hash_it: bool = True) -> int:
        p = Path(path)
        digest = sha256_file(p) if hash_it and p.is_file() else None
        cur = self.conn.execute(
            "INSERT INTO outputs(job_id,kind,path,sha256,ref_id,meta_json,created_at) VALUES (?,?,?,?,?,?,?)",
            (job_id, kind, str(p), digest, ref_id, json.dumps(meta or {}, ensure_ascii=False), now_iso()))
        return cur.lastrowid

    def outputs(self, job_id: str, kind: str | None = None) -> list[sqlite3.Row]:
        if kind:
            return self.conn.execute("SELECT * FROM outputs WHERE job_id=? AND kind=? ORDER BY id",
                                     (job_id, kind)).fetchall()
        return self.conn.execute("SELECT * FROM outputs WHERE job_id=? ORDER BY id", (job_id,)).fetchall()

    def save_metadata(self, job_id: str, target: str, data: Any) -> None:
        self.conn.execute("INSERT INTO metadata(job_id,target,json,created_at) VALUES (?,?,?,?)",
                          (job_id, target, json.dumps(data, ensure_ascii=False), now_iso()))

    def get_metadata(self, job_id: str, target: str) -> Any | None:
        row = self.conn.execute("SELECT json FROM metadata WHERE job_id=? AND target=? ORDER BY id DESC LIMIT 1",
                                (job_id, target)).fetchone()
        return json.loads(row["json"]) if row else None

    def save_qa(self, job_id: str, target: str, passed: bool, checks: list[dict]) -> None:
        self.conn.execute("INSERT INTO qa_results(job_id,target,passed,checks_json,created_at) VALUES (?,?,?,?,?)",
                          (job_id, target, int(passed), json.dumps(checks, ensure_ascii=False), now_iso()))

    def latest_qa(self, job_id: str, target: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM qa_results WHERE job_id=? AND target=? ORDER BY id DESC LIMIT 1",
                                (job_id, target)).fetchone()
        if not row:
            return None
        return {"passed": bool(row["passed"]), "checks": json.loads(row["checks_json"])}

    def write_report(self, job_id: str, name: str, data: Any) -> Path:
        return atomic_write_json(self.job_dir(job_id) / "reports" / name, data)

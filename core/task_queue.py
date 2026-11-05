"""Minimal SQLite-backed FIFO queue so long jobs can be run one at a time (low-RAM devices)."""
from __future__ import annotations

from typing import Callable

from .job_manager import JobManager, now_iso


class TaskQueue:
    def __init__(self, jm: JobManager):
        self.jm = jm
        self.conn = jm.conn

    def enqueue(self, job_id: str, action: str = "run") -> int:
        self.jm.get_job(job_id)
        cur = self.conn.execute("INSERT INTO task_queue(job_id, action, enqueued_at) VALUES (?,?,?)",
                                (job_id, action, now_iso()))
        return cur.lastrowid

    def pending(self) -> list:
        return self.conn.execute("SELECT * FROM task_queue WHERE status='queued' ORDER BY id").fetchall()

    def claim_next(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute("SELECT * FROM task_queue WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
            if row:
                self.conn.execute("UPDATE task_queue SET status='running', started_at=? WHERE id=?",
                                  (now_iso(), row["id"]))
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return row

    def run_next(self, handler: Callable[[str, str], None]) -> dict | None:
        row = self.claim_next()
        if not row:
            return None
        try:
            handler(row["job_id"], row["action"])
            status, err = "done", None
        except BaseException as e:  # noqa: BLE001
            status, err = "failed", f"{type(e).__name__}: {e}"
        self.conn.execute("UPDATE task_queue SET status=?, finished_at=?, error=? WHERE id=?",
                          (status, now_iso(), err, row["id"]))
        return {"id": row["id"], "job_id": row["job_id"], "status": status, "error": err}

    def drain(self, handler: Callable[[str, str], None]) -> list[dict]:
        out = []
        while (r := self.run_next(handler)) is not None:
            out.append(r)
        return out

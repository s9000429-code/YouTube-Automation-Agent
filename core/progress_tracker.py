"""Progress reporting: DB-visible stage plus reports/progress.json and optional callbacks."""
from __future__ import annotations

import time
from typing import Callable

from .job_manager import JobManager, now_iso
from .safe_io import atomic_write_json


class ProgressTracker:
    def __init__(self, jm: JobManager, job_id: str, stages: list[str],
                 callback: Callable[[dict], None] | None = None):
        self.jm = jm
        self.job_id = job_id
        self.stages = stages
        self.callback = callback
        self.state = {"job_id": job_id, "stages": {s: "pending" for s in stages},
                      "current": None, "percent": 0.0, "message": "", "updated_at": now_iso()}
        self._t0 = time.time()

    def _flush(self) -> None:
        done = sum(1 for v in self.state["stages"].values() if v in ("done", "skipped"))
        cur = 0.0
        if self.state["current"]:
            cur = self.state.get("stage_fraction", 0.0)
        self.state["percent"] = round(100.0 * (done + cur) / max(1, len(self.stages)), 1)
        self.state["elapsed_s"] = round(time.time() - self._t0, 1)
        self.state["updated_at"] = now_iso()
        atomic_write_json(self.jm.job_dir(self.job_id) / "reports" / "progress.json", self.state)
        if self.callback:
            self.callback(dict(self.state))

    def start(self, stage: str, message: str = "") -> None:
        self.state["stages"][stage] = "running"
        self.state.update(current=stage, message=message or stage, stage_fraction=0.0)
        self._flush()

    def update(self, fraction: float, message: str = "") -> None:
        self.state["stage_fraction"] = max(0.0, min(1.0, fraction))
        if message:
            self.state["message"] = message
        self._flush()

    def finish(self, stage: str, skipped: bool = False) -> None:
        self.state["stages"][stage] = "skipped" if skipped else "done"
        self.state.update(current=None, stage_fraction=0.0)
        self._flush()

    def fail(self, stage: str, message: str) -> None:
        self.state["stages"][stage] = "failed"
        self.state.update(current=None, message=message)
        self._flush()

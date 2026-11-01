"""Interrupted-job detection and resume planning."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .job_manager import JobManager, now_iso
from .logging_manager import get_logger
from .platform_utils import is_windows

log = get_logger("recovery")
ACTIVE = ("processing", "rendering")


def _pid_alive_windows(pid: int, kernel32) -> bool:
    """Windows: os.kill(pid, 0) would send CTRL_C_EVENT, so ask the OS for the process exit code instead."""
    handle = kernel32.OpenProcess(0x1000, False, int(pid))      # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        import ctypes
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == 259                                 # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if is_windows():
        import ctypes
        return _pid_alive_windows(pid, ctypes.WinDLL("kernel32", use_last_error=True))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def find_interrupted(jm: JobManager) -> list[str]:
    """Jobs recorded as active whose worker process no longer exists."""
    rows = jm.conn.execute("SELECT id, worker_pid FROM jobs WHERE status IN ('processing','rendering')").fetchall()
    return [r["id"] for r in rows if r["worker_pid"] != os.getpid() and not _pid_alive(r["worker_pid"])]


def recover(jm: JobManager) -> list[str]:
    """Mark orphaned jobs/stages interrupted. Never deletes files."""
    ids = find_interrupted(jm)
    for job_id in ids:
        jm.conn.execute("UPDATE stages SET status='interrupted', finished_at=? WHERE job_id=? AND status='running'",
                        (now_iso(), job_id))
        jm.set_status(job_id, "interrupted", error="worker process vanished; resume to continue")
        jm.record_error(job_id, jm.get_job(job_id)["stage"] or "?", "interrupted: worker process vanished")
        log.warning("job %s marked interrupted", job_id)
    return ids


def resume_plan(jm: JobManager, job_id: str) -> dict:
    """Which stages are valid-complete and which must re-run."""
    rows = jm.conn.execute("SELECT name, status, artifacts_json FROM stages WHERE job_id=?", (job_id,)).fetchall()
    plan = {"complete": [], "rerun": []}
    for r in rows:
        ok = jm.stage_done(job_id, r["name"])
        (plan["complete"] if ok else plan["rerun"]).append(r["name"])
    return plan

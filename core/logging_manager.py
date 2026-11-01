"""Central logging: rotating global log plus an optional per-job log file."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_FMT = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
LOGGER_NAME = "yta"


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def setup_logging(log_dir: str | Path, level: int = logging.INFO, console: bool = False) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(level)
    root.propagate = False
    target = str((log_dir / "agent.log").resolve())
    for h in list(root.handlers):
        if getattr(h, "_yta_global", False):
            if getattr(h, "baseFilename", None) == target:
                break
            root.removeHandler(h)
            h.close()
    else:
        pass
    if not any(getattr(h, "baseFilename", None) == target for h in root.handlers):
        fh = logging.handlers.RotatingFileHandler(target, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(_FMT)
        fh._yta_global = True  # type: ignore[attr-defined]
        root.addHandler(fh)
    if console and not any(isinstance(h, logging.StreamHandler) and getattr(h, "_yta_console", False)
                           for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(_FMT)
        ch._yta_console = True  # type: ignore[attr-defined]
        root.addHandler(ch)
    return root


def attach_job_log(job_dir: str | Path, job_id: str) -> logging.Handler:
    reports = Path(job_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(reports / "job.log", encoding="utf-8")
    h.setFormatter(_FMT)
    h._yta_job = job_id  # type: ignore[attr-defined]
    logging.getLogger(LOGGER_NAME).addHandler(h)
    return h


def detach_job_log(handler: logging.Handler) -> None:
    logging.getLogger(LOGGER_NAME).removeHandler(handler)
    handler.close()

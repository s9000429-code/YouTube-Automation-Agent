"""File-safety helpers: never overwrite finals, never touch originals, check storage."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator


class SafetyError(Exception):
    """Raised when an operation would violate a data-safety rule."""


class InsufficientStorage(SafetyError):
    """Raised when there is not enough free disk space for an operation."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def quick_fingerprint(path: str | Path, sample: int = 1 << 20) -> dict:
    """Cheap identity for huge files: size, mtime and hashes of first/last MiB."""
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(sample))
        if st.st_size > sample:
            f.seek(max(0, st.st_size - sample))
            h.update(f.read(sample))
    return {"size": st.st_size, "mtime": int(st.st_mtime), "edge_sha256": h.hexdigest()}


def is_within(child: str | Path, parent: str | Path) -> bool:
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _candidate(path: Path, n: int) -> Path:
    return path if n == 1 else path.with_name(f"{path.stem}_v{n}{path.suffix}")


def reserve_output(path: str | Path) -> Path:
    """Atomically claim a final-output path. Never overwrites: picks name, name_v2, ...

    The returned path exists as an empty file owned by the caller.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        cand = _candidate(path, n)
        try:
            fd = os.open(cand, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return cand
        except FileExistsError:
            n += 1
            if n > 10_000:
                raise SafetyError(f"cannot reserve output near {path}")


@contextlib.contextmanager
def reserved_output(path: str | Path) -> Iterator[Path]:
    """Reserve a final output; if the body raises, remove only the file we created."""
    reserved = reserve_output(path)
    try:
        yield reserved
    except BaseException:
        with contextlib.suppress(OSError):
            if reserved.exists():
                reserved.unlink()
        raise


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write a job-owned intermediate file atomically (replaces previous version)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return path


def atomic_write_json(path: str | Path, obj, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def free_bytes(path: str | Path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def require_free_space(path: str | Path, needed_bytes: int, reserve_bytes: int = 512 * 1024 * 1024) -> None:
    free = free_bytes(path)
    if free < needed_bytes + reserve_bytes:
        raise InsufficientStorage(
            f"Need ~{(needed_bytes + reserve_bytes) / 1e6:.0f} MB free at {path}, "
            f"only {free / 1e6:.0f} MB available. Free space or choose another output drive."
        )


TMP_MARKER = ".agent_job_tmp"


def make_job_tmp(job_dir: str | Path) -> Path:
    tmp = Path(job_dir) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / TMP_MARKER).write_text("created by YouTube Automation Agent; safe to delete\n")
    return tmp


def cleanup_job_tmp(job_dir: str | Path) -> bool:
    """Delete only the marked tmp dir that sits directly inside the job dir."""
    tmp = Path(job_dir) / "tmp"
    if not tmp.is_dir() or not (tmp / TMP_MARKER).exists():
        return False
    if tmp.is_symlink() or not is_within(tmp, job_dir):
        raise SafetyError(f"refusing to delete {tmp}")
    shutil.rmtree(tmp)
    return True


def make_processing_copy(src: str | Path, dest_dir: str | Path) -> tuple[Path, dict]:
    """Copy a source video into the job (never modifies the original).

    Returns (copy_path, fingerprint_of_original). Checks free space first.
    """
    src = Path(src)
    if not src.is_file():
        raise SafetyError(f"source file not found: {src}")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    require_free_space(dest_dir, src.stat().st_size, reserve_bytes=256 * 1024 * 1024)
    fp = quick_fingerprint(src)
    dest = reserve_output(dest_dir / src.name)
    shutil.copyfile(src, dest)
    if dest.stat().st_size != fp["size"]:
        raise SafetyError("processing copy size mismatch")
    return dest, fp

"""Input validation and metadata for the long-form source."""
from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

from core.media import MediaError, probe
from core.safe_io import SafetyError, require_free_space, reserve_output

SUPPORTED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mpg", ".mpeg", ".ts"}


def analyze_video(path: str | Path) -> dict:
    """Probe the source and return {info, warnings, errors}. Errors mean we cannot continue."""
    p = Path(path)
    errors, warnings = [], []
    if not p.is_file():
        return {"info": None, "warnings": [], "errors": [f"file not found: {p}"]}
    if p.suffix.lower() not in SUPPORTED_EXT:
        warnings.append(f"unusual extension {p.suffix!r}; trying anyway")
    try:
        info = probe(p)
    except MediaError as e:
        return {"info": None, "warnings": warnings, "errors": [f"cannot read video: {e}"]}
    if not info["has_video"]:
        errors.append("no video stream")
    if not info["has_audio"]:
        errors.append("no audio stream: there is no speech to transcribe")
    if info["duration"] <= 0:
        errors.append("duration is zero or unknown")
    if info["has_video"]:
        if info["height"] and info["height"] < 480:
            warnings.append(f"low source resolution ({info['width']}x{info['height']}); vertical Shorts will look soft")
        if info.get("fps") and info["fps"] > 61:
            warnings.append("very high frame rate; output is resampled to the configured FPS")
        if info["rotation"]:
            warnings.append(f"source has rotation metadata ({info['rotation']} deg); handled by ffmpeg autorotate")
    if info["audio_streams"] > 1:
        warnings.append(f"{info['audio_streams']} audio streams; the first one is used")
    return {"info": info, "warnings": warnings, "errors": errors}


def estimate_disk_need(info: dict, n_shorts: int, max_dur_s: float, copy_source: bool = True) -> int:
    """Conservative bytes needed: source copy + 16 kHz mono audio + shorts (~6 MB per 60 s at standard quality)."""
    audio = int(info["duration"] * 16000 * 2)
    shorts = int(n_shorts * max_dur_s / 60.0 * 8 * 1024 * 1024) * 2  # previews + finals
    return (info["size_bytes"] if copy_source else 0) + audio + shorts


def download_direct_url(url: str, dest_dir: str | Path, confirm_rights: bool, max_bytes: int = 8 << 30) -> Path:
    """Fetch a DIRECT video-file URL. Refuses without an explicit rights confirmation, and refuses
    web pages / streaming-site URLs (no scraping, no bypassing access controls or download limits)."""
    if not confirm_rights:
        raise SafetyError("URL input requires confirm_rights=True: you must have permission to process this video")
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https"):
        raise SafetyError("only http(s) direct file URLs are supported")
    req = urllib.request.Request(url, headers={"User-Agent": "YouTubeAutomationAgent/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()
        if not (ctype.startswith("video/") or ctype in ("application/octet-stream", "application/mp4")):
            raise SafetyError(f"URL is not a direct video file (Content-Type: {ctype or 'unknown'}). "
                              "Streaming-site pages are not supported; download the video lawfully yourself and pass the file.")
        length = int(r.headers.get("Content-Length") or 0)
        if length > max_bytes:
            raise SafetyError("remote file larger than the configured limit")
        require_free_space(dest_dir, length or 512 << 20)
        name = Path(urllib.parse.unquote(u.path)).name or "download.mp4"
        dest = reserve_output(Path(dest_dir) / name)
        got = 0
        with open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                got += len(chunk)
                if got > max_bytes:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise SafetyError("download exceeded size limit")
                f.write(chunk)
    return dest

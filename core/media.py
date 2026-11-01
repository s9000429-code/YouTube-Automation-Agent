"""FFmpeg / FFprobe wrappers. Everything streams through files; nothing loads video into RAM."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable

from .logging_manager import get_logger

log = get_logger("media")


class MediaError(RuntimeError):
    pass


_THREADS = 0   # 0 = let ffmpeg decide; set from settings (low-RAM machines use 2)


def set_threads(n: int) -> None:
    global _THREADS
    _THREADS = max(0, int(n or 0))


def concat_line(path: str | Path) -> str:
    """One line of an ffmpeg concat list. Forward slashes work on Windows; a ' in the path must be escaped."""
    posix = path.as_posix() if hasattr(path, "as_posix") else Path(path).as_posix()
    p = posix.replace("'", "'\\''")
    return f"file '{p}'\n"


def _tail(text: str, n: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def run(cmd: list[str], timeout: float | None = None, cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(map(str, cmd)))
    try:
        cp = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout,
                            encoding="utf-8", errors="replace", cwd=cwd)
    except FileNotFoundError as e:
        raise MediaError(f"required tool not found: {cmd[0]}") from e
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"{cmd[0]} timed out after {timeout}s") from e
    if cp.returncode != 0:
        raise MediaError(f"{cmd[0]} failed (exit {cp.returncode}):\n{_tail(cp.stderr)}")
    return cp


def ffmpeg(args: list[str], duration: float | None = None,
           progress: Callable[[float], None] | None = None, timeout: float | None = None,
           cwd: str | Path | None = None) -> None:
    """Run ffmpeg; if duration+progress are given, report 0..1 using -progress."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    if _THREADS:
        cmd += ["-threads", str(_THREADS)]
    if progress and duration:
        cmd += ["-progress", "pipe:1", "-nostats"]
    cmd += [str(a) for a in args]
    if not (progress and duration):
        run(cmd, timeout=timeout, cwd=cwd)
        return
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace", cwd=cwd)
    assert proc.stdout is not None and proc.stderr is not None
    for line in proc.stdout:
        m = re.match(r"out_time_(?:us|ms)=(\d+)", line)
        if m:
            progress(min(1.0, int(m.group(1)) / 1e6 / max(duration, 0.001)))
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise MediaError(f"ffmpeg failed (exit {proc.returncode}):\n{_tail(err)}")


def probe(path: str | Path) -> dict:
    """Return normalized media properties via ffprobe."""
    p = Path(path)
    if not p.is_file():
        raise MediaError(f"file not found: {p}")
    cp = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", p])
    data = json.loads(cp.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    a = [s for s in streams if s.get("codec_type") == "audio"]
    fmt = data.get("format", {})
    info: dict = {
        "path": str(p), "size_bytes": int(fmt.get("size") or p.stat().st_size),
        "duration": float(fmt.get("duration") or (v or {}).get("duration") or (a[0].get("duration") if a else 0) or 0),
        "container": fmt.get("format_name"), "bit_rate": int(fmt.get("bit_rate") or 0),
        "has_video": v is not None, "has_audio": bool(a), "audio_streams": len(a),
        "width": None, "height": None, "fps": None, "video_codec": None,
        "audio_codec": a[0].get("codec_name") if a else None,
        "sample_rate": int(a[0].get("sample_rate", 0)) if a else None,
        "channels": a[0].get("channels") if a else None,
        "audio_languages": [s.get("tags", {}).get("language") for s in a],
        "rotation": 0,
    }
    if v:
        w, h = int(v.get("width", 0)), int(v.get("height", 0))
        rot = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(sd["rotation"]) % 360
        if rot in (90, 270):
            w, h = h, w
        num, _, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        info.update(width=w, height=h, fps=round(fps, 3), video_codec=v.get("codec_name"), rotation=rot,
                    pix_fmt=v.get("pix_fmt"))
    return info


def detect_black(path: str | Path, min_duration: float = 0.5, pix_th: float = 0.10) -> list[tuple[float, float]]:
    """Return [(start, end)] spans of black frames."""
    cp = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-vf",
         f"blackdetect=d={min_duration}:pix_th={pix_th}", "-an", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    spans = []
    for m in re.finditer(r"black_start:([\d.]+)\s+black_end:([\d.]+)", cp.stderr):
        spans.append((float(m.group(1)), float(m.group(2))))
    return spans


def decode_check(path: str | Path, min_black: float = 0.5, pix_th: float = 0.10) -> tuple[list[str], list[tuple[float, float]]]:
    """One full decode pass: returns (decode_errors, black_spans). Also proves the file is playable."""
    cp = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info", "-i", str(path), "-vf",
         f"blackdetect=d={min_black}:pix_th={pix_th}", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    errs = [l for l in cp.stderr.splitlines()
            if re.search(r"\b(error|invalid data|corrupt|failed)\b", l, re.I) and "blackdetect" not in l.lower()]
    if cp.returncode != 0:
        errs.append(f"ffmpeg exit {cp.returncode}")
    spans = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+)\s+black_end:([\d.]+)", cp.stderr)]
    return errs, spans


def detect_silence(path: str | Path, noise_db: float = -35.0, min_duration: float = 0.4) -> list[tuple[float, float]]:
    cp = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_duration}", "-vn", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", cp.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", cp.stderr)]
    return [(s, e) for s, e in zip(starts, ends)]


def audio_mean_volume(path: str | Path, start: float = 0.0, duration: float | None = None) -> float | None:
    """Mean volume in dB (None if no audio / undetectable)."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(path), "-af", "volumedetect", "-vn", "-f", "null", "-"]
    cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"mean_volume: (-?[\d.]+|-inf) dB", cp.stderr)
    if not m or m.group(1) == "-inf":
        return None
    return float(m.group(1))


def make_silence(path: str | Path, seconds: float, sample_rate: int = 22050) -> Path:
    ffmpeg(["-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono", "-t", f"{seconds:.3f}",
            "-c:a", "pcm_s16le", path])
    return Path(path)


def concat_audio(files: list[str | Path], out: str | Path, list_file: str | Path, sample_rate: int = 22050) -> Path:
    """Concatenate WAV/other audio files, normalising to mono PCM at one sample rate."""
    list_file = Path(list_file)
    list_file.parent.mkdir(parents=True, exist_ok=True)
    norm_dir = list_file.parent
    lines = []
    for i, f in enumerate(files):
        norm = norm_dir / f"_n{i:05d}.wav"
        ffmpeg(["-i", f, "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", norm])
        lines.append(concat_line(norm))
    list_file.write_text("".join(lines), encoding="utf-8")
    ffmpeg(["-f", "concat", "-safe", "0", "-i", list_file, "-c:a", "pcm_s16le", out])
    for i in range(len(files)):
        (norm_dir / f"_n{i:05d}.wav").unlink(missing_ok=True)
    return Path(out)


def quality_args(settings: dict, profile: str | None = None, preview: bool = False) -> list[str]:
    enc = settings["encoding"]
    preset, crf = enc["preset"], enc["crf"]
    if profile and profile in settings.get("quality_profiles", {}):
        q = settings["quality_profiles"][profile]
        preset, crf = q["preset"], q["crf"]
    if preview:
        preset, crf = "ultrafast", enc.get("preview_crf", 32)
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", enc["audio_bitrate"], "-movflags", "+faststart"]

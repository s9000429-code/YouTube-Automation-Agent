"""Audio extraction (16 kHz mono PCM) and silence-aware chunk planning for staged transcription."""
from __future__ import annotations

from pathlib import Path

from core.media import detect_silence, ffmpeg
from core.safe_io import require_free_space


def extract_audio(video: str | Path, out_wav: str | Path, start: float | None = None, end: float | None = None,
                  duration_hint: float | None = None, progress=None) -> Path:
    """Stream-decode the audio track to WAV. Never loads the video into memory."""
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = (end - start) if (start is not None and end is not None) else duration_hint
    if dur:
        require_free_space(out.parent, int(dur * 16000 * 2) + 1_000_000, reserve_bytes=128 * 1024 * 1024)
    args: list = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    if end is not None:
        args += ["-to", f"{end:.3f}"] if start is None else ["-t", f"{end - start:.3f}"]
    args += ["-i", video, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out]
    ffmpeg(args, duration=dur, progress=progress)
    if not out.exists() or out.stat().st_size < 1000:
        raise RuntimeError("audio extraction produced no data")
    return out


def plan_chunks(duration: float, chunk_s: float = 600.0, silences: list[tuple[float, float]] | None = None,
                search_window: float = 45.0, min_tail: float = 0.25) -> list[tuple[float, float]]:
    """Split [0, duration] into ~chunk_s pieces, cutting inside silences when one is near the target."""
    if duration <= chunk_s * 1.25:
        return [(0.0, duration)]
    mids = [(a + b) / 2 for a, b in (silences or [])]
    cuts = [0.0]
    target = chunk_s
    while target < duration - chunk_s * min_tail:
        near = [m for m in mids if abs(m - target) <= search_window and m > cuts[-1] + chunk_s * 0.4]
        cut = min(near, key=lambda m: abs(m - target)) if near else target
        cuts.append(cut)
        target = cut + chunk_s
    cuts.append(duration)
    return list(zip(cuts, cuts[1:]))


def extract_chunks(wav: str | Path, duration: float, out_dir: str | Path, chunk_s: float = 600.0):
    """Yield (chunk_wav, offset, end). Caller deletes each chunk after use."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sil = detect_silence(wav, noise_db=-35, min_duration=0.5) if duration > chunk_s * 1.25 else []
    for k, (a, b) in enumerate(plan_chunks(duration, chunk_s, sil)):
        cwav = out_dir / f"chunk_{k:03d}.wav"
        ffmpeg(["-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", wav, "-c:a", "pcm_s16le", cwav])
        yield cwav, a, b

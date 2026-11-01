"""Runs a replaceable ASR engine over a source in staged chunks and returns one validated transcript."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from core.captions import Cue, write_srt
from providers.transcription import Transcript, TranscriptionEngine, detect_language
from providers.transcription.base import Segment
from providers.transcription.langid import normalize_code

from .audio_extractor import extract_chunks


def merge_chunks(parts: list[Transcript], engine_name: str) -> Transcript:
    segs: list[Segment] = []
    lang = None
    for p in parts:
        lang = lang or p.language
        for s in p.segments:
            s.id = len(segs)
            segs.append(s)
    t = Transcript(lang, engine_name, segs)
    t.refresh_timing()
    return t


def transcribe_audio(engine: TranscriptionEngine, wav: str | Path, duration: float, language: str | None,
                     workdir: str | Path, chunk_s: float = 600.0,
                     progress: Callable[[float], None] | None = None) -> Transcript:
    """Transcribe `wav` (16 kHz mono). Imported transcripts are used as-is; real engines run per chunk,
    so memory stays bounded on long videos. Offsets map every timestamp back to ORIGINAL video time."""
    lang = None if language in (None, "auto") else language
    if engine.name == "imported":
        t = engine.transcribe(wav, lang)
    else:
        parts = []
        done = 0.0
        for cwav, a, b in extract_chunks(wav, duration, Path(workdir) / "chunks", chunk_s):
            try:
                parts.append(engine.transcribe(cwav, lang, offset=a))
            finally:
                cwav.unlink(missing_ok=True)
            done = b
            if progress:
                progress(min(1.0, done / max(duration, 1)))
        t = merge_chunks(parts, engine.name)
    t.duration = duration
    t.language = normalize_code(t.language) or detect_language(t.text())
    # a Hindi ASR result in Latin script is Hinglish for our purposes
    if t.language in ("en", "hi") and detect_language(t.text()) == "hinglish":
        t.language = "hinglish"
    t.refresh_timing()
    return t


def save_transcript(t: Transcript, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    jp = t.save(out_dir / "transcript.json")
    cues = [Cue(s.start, max(s.end, s.start + 0.2), s.text, timing="verified") for s in t.segments]
    sp = write_srt(cues, out_dir / "transcript.srt")
    return {"json": str(jp), "srt": str(sp)}

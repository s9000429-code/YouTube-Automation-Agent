"""Parsers for transcript formats: whisper.cpp JSON, OpenAI-style whisper JSON, SRT, VTT, native JSON."""
from __future__ import annotations

import json
import re
from pathlib import Path

from core.captions import parse_srt

from .base import Segment, Transcript, Word


def parse_whisper_cpp_json(data: dict, offset: float = 0.0, engine: str = "whisper.cpp") -> Transcript:
    """Parse `whisper-cli -oj` (segments) or `-ojf` (segments + tokens -> words) output."""
    lang = (data.get("result") or {}).get("language")
    segs: list[Segment] = []
    for i, item in enumerate(data.get("transcription", [])):
        off = item.get("offsets") or {}
        s, e = off.get("from", 0) / 1000.0 + offset, off.get("to", 0) / 1000.0 + offset
        text = (item.get("text") or "").strip()
        words: list[Word] = []
        for tok in item.get("tokens", []) or []:
            t = tok.get("text", "")
            if not t or t.startswith("[_") or t.startswith("<|"):
                continue
            to = tok.get("offsets") or {}
            ts, te = to.get("from", 0) / 1000.0 + offset, to.get("to", 0) / 1000.0 + offset
            if (t.startswith(" ") or not words) and t.strip():
                words.append(Word(t.strip(), ts, max(te, ts), tok.get("p")))
            elif words and t.strip():
                w = words[-1]
                w.text += t
                w.end = max(w.end, te)
        if text:
            segs.append(Segment(len(segs), s, max(e, s), text, words))
    t = Transcript(lang, engine, segs)
    t.refresh_timing()
    return t


def parse_openai_json(data: dict, offset: float = 0.0, engine: str = "whisper-json") -> Transcript:
    segs: list[Segment] = []
    all_words = data.get("words") or []
    for item in data.get("segments", []):
        words = [Word((w.get("word") or w.get("text") or "").strip(), float(w["start"]) + offset,
                      float(w["end"]) + offset, w.get("probability"))
                 for w in (item.get("words") or []) if (w.get("word") or w.get("text") or "").strip()]
        if not words and all_words:
            words = [Word((w.get("word") or "").strip(), float(w["start"]) + offset, float(w["end"]) + offset)
                     for w in all_words if item["start"] - 0.01 <= w["start"] <= item["end"] + 0.01
                     and (w.get("word") or "").strip()]
        text = (item.get("text") or "").strip()
        if text:
            segs.append(Segment(len(segs), float(item["start"]) + offset, float(item["end"]) + offset, text,
                                words, item.get("speaker")))
    t = Transcript(data.get("language"), engine, segs)
    t.refresh_timing()
    return t


def parse_vtt(text: str) -> Transcript:
    body = re.sub(r"^WEBVTT.*?\n\n", "", text.replace("\r\n", "\n"), count=1, flags=re.S)
    # normalise VTT timestamps (no hours / '.') into SRT so one parser handles both
    def fix(m):
        t = m.group(0)
        if t.count(":") == 1:
            t = "00:" + t
        return t.replace(".", ",")
    body = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?\.\d{3}", fix, body)
    cues = parse_srt(body)
    segs = [Segment(i, c.start, c.end, c.text.replace("\n", " ")) for i, c in enumerate(cues)]
    t = Transcript(None, "vtt-import", segs)
    t.refresh_timing()
    return t


def parse_srt_transcript(text: str) -> Transcript:
    segs = [Segment(i, c.start, c.end, c.text.replace("\n", " ")) for i, c in enumerate(parse_srt(text))]
    t = Transcript(None, "srt-import", segs)
    t.refresh_timing()
    return t


def load_transcript_file(path: str | Path, offset: float = 0.0) -> Transcript:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"transcript file not found: {p}")
    text = p.read_text(encoding="utf-8")
    suf = p.suffix.lower()
    if suf == ".srt":
        t = parse_srt_transcript(text)
    elif suf == ".vtt":
        t = parse_vtt(text)
    elif suf == ".json":
        data = json.loads(text)
        if "transcription" in data:
            t = parse_whisper_cpp_json(data, 0.0, "whisper.cpp-import")
        elif "segments" in data and data["segments"] and "id" in data["segments"][0] and "engine" in data:
            t = Transcript.from_dict(data)
        elif "segments" in data:
            t = parse_openai_json(data, 0.0, "whisper-json-import")
        else:
            raise ValueError(f"unrecognised transcript JSON layout in {p}")
    else:
        raise ValueError(f"unsupported transcript format {suf!r} (use .json, .srt or .vtt)")
    return t.shifted(offset) if offset else t

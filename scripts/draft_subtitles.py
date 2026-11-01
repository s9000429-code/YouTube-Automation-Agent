"""Subtitles for Mode A from the measured narration timeline.

Labeling (never over-claim):
  measured_sentence : each sentence's start/end are measured from the synthesized audio.
  interpolated      : when a sentence is split into several cues, the split points inside it are proportional
                      to character count. NOT aligned to the actual spoken words.
Neither is 'verified speech-aligned'. For word-accurate captions, run the narration through a transcription
engine (`agent.py topic align`) — only then are cues marked verified."""
from __future__ import annotations

from pathlib import Path

from core.captions import (TIMING_INTERPOLATED, TIMING_MEASURED, Cue, readability_check, wrap_text, write_srt)


def cues_from_timeline(timeline: list[dict], max_chars: int = 42, max_lines: int = 2, language: str = "en") -> list[Cue]:
    cues: list[Cue] = []
    for item in timeline:
        text = item["text"].strip()
        start, end = item["start"], item["end"]
        cap = max_chars * max_lines
        if len(text) <= cap:
            cues.append(Cue(start, end, "\n".join(wrap_text(text, max_chars)), timing=TIMING_MEASURED))
            continue
        # split at clause boundaries into chunks <= cap, time proportionally
        import re
        parts, cur = [], ""
        for tok in re.split(r"(?<=[,;:—–])\s+|\s+", text):
            if cur and len(cur) + 1 + len(tok) > cap:
                parts.append(cur)
                cur = tok
            else:
                cur = f"{cur} {tok}".strip()
        if cur:
            parts.append(cur)
        total = sum(len(p) for p in parts)
        t = start
        for p in parts:
            d = (end - start) * len(p) / total
            cues.append(Cue(t, t + d, "\n".join(wrap_text(p, max_chars)), timing=TIMING_INTERPOLATED))
            t += d
    for k in range(len(cues) - 1):               # strict ordering
        cues[k].end = min(cues[k].end, cues[k + 1].start) if cues[k].end > cues[k + 1].start else cues[k].end
    return [c for c in cues if c.end > c.start]


def write_subtitles(cues: list[Cue], out_dir: str | Path, stem: str = "narration") -> dict:
    srt = write_srt(cues, Path(out_dir) / f"{stem}.srt")
    timings = sorted({c.timing for c in cues})
    return {"srt": str(srt), "timing_labels": timings, "verified_speech_aligned": False,
            "readability": readability_check(cues)}

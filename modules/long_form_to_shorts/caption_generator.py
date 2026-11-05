"""Caption cues for a clip, built from the ACTUAL transcript (never re-written by a model)."""
from __future__ import annotations

from dataclasses import dataclass

from core.captions import (TIMING_INTERPOLATED, TIMING_VERIFIED, Cue, build_ass, group_words_into_cues,
                           readability_check, write_srt)
from pathlib import Path

from providers.transcription.base import Transcript


@dataclass
class ClipCaptions:
    cues: list[Cue]
    timing: str            # 'verified_word' | 'interpolated' | 'none'
    word_highlight_ok: bool
    warnings: list[str]


def clip_cues(tr: Transcript, start: float, end: float, max_chars: int = 22, max_words: int = 5) -> ClipCaptions:
    """Cues with times RELATIVE to the clip start. Word times come straight from the engine when present;
    otherwise words are spread across the segment and flagged 'interpolated' (no word highlight)."""
    dur = end - start
    warnings: list[str] = []
    words: list[tuple[str, float, float]] = []
    interpolated = False
    for s in tr.segments:
        if s.end <= start or s.start >= end:
            continue
        if s.words:
            for w in s.words:
                # keep a word only if it lies (almost) wholly inside the clip
                if w.end > start + 0.02 and w.start < end - 0.02 and w.start >= start - 0.15 and w.end <= end + 0.15:
                    words.append((w.text, max(0.0, w.start - start), min(dur, w.end - start)))
        else:
            interpolated = True
            toks = s.text.split()
            if not toks:
                continue
            step = (s.end - s.start) / len(toks)
            for k, t in enumerate(toks):
                ws, we = s.start + k * step, s.start + (k + 1) * step
                if ws >= start - 0.05 and we <= end + 0.05:
                    words.append((t, max(0.0, ws - start), min(dur, we - start)))
    if not words:
        return ClipCaptions([], "none", False, ["no transcript words fall inside this clip"])
    timing = TIMING_INTERPOLATED if interpolated else TIMING_VERIFIED
    cues = group_words_into_cues(words, max_chars=max_chars, max_words=max_words, timing=timing)
    # make cues strictly sequential (no overlap) and give very short ones a minimum on-screen time
    for k, c in enumerate(cues):
        nxt = cues[k + 1].start if k + 1 < len(cues) else dur
        c.end = min(max(c.end, c.start + min(0.6, max(0.0, nxt - c.start))), nxt, dur)
    cues = [c for c in cues if c.end > c.start]
    if interpolated:
        warnings.append("transcript has segment-level timestamps only: caption words are interpolated inside each "
                        "segment, so word-level timing is approximate and word highlighting is disabled")
    return ClipCaptions(cues, "interpolated" if interpolated else "verified_word", not interpolated, warnings)


def write_caption_files(cc: ClipCaptions, out_dir: str | Path, stem: str, width: int, height: int, style: str,
                        language: str, y_frac: float, hook_text: str | None, hook_y_frac: float) -> dict:
    out_dir = Path(out_dir)
    srt = write_srt(cc.cues, out_dir / f"{stem}.srt")
    ass_text, warns = build_ass(cc.cues, width, height, style=style, language=language, y_frac=y_frac,
                                hook_text=hook_text, hook_y_frac=hook_y_frac,
                                word_highlight=None if cc.word_highlight_ok else False)
    ass = out_dir / f"{stem}.ass"
    ass.write_text(ass_text, encoding="utf-8")
    return {"srt": str(srt), "ass": str(ass), "warnings": warns + cc.warnings,
            "readability": readability_check(cc.cues, max_cps=22, max_line_chars=30, max_lines=3, min_dur=0.3)}

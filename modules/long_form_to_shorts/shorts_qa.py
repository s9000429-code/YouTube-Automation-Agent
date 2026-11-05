"""Quality assurance for a rendered Short. A Short is 'passed' only if every level=='fail' check is OK.

Limits worth knowing: 'misleading context' cannot be proven by software. We check structural signs
(sentence-aligned cuts, dependent openings, dangling questions) and flag them as warnings; the
human approval step is the real safeguard."""
from __future__ import annotations

import json
from pathlib import Path

from core.captions import readability_check, validate_srt_text, parse_srt
from core.media import MediaError, audio_mean_volume, decode_check, probe
from core.safe_io import sha256_file

from .clip_selector import Clip, is_duplicate, validate_clip
from .shorts_metadata import validate_metadata
from .text_utils import Sentence


def qa_short(path: str | Path, clip: Clip, settings: dict, sentences: list[Sentence], media_duration: float,
             srt_path: str | Path | None, metadata_path: str | Path | None, metadata: dict | None,
             others: list[Clip], other_hashes: dict[str, str], expected_wh: tuple[int, int] = (1080, 1920),
             expected_fps: float = 30.0, caption_timing: str | None = None) -> dict:
    cfg = settings["shorts"]
    checks: list[dict] = []

    def add(check, ok, detail="", level="fail"):
        checks.append({"check": check, "ok": bool(ok), "detail": str(detail), "level": level})

    p = Path(path)
    add("file_exists", p.is_file() and p.stat().st_size > 1000, p)
    info = None
    if p.is_file():
        try:
            info = probe(p)
        except MediaError as e:
            add("playable", False, e)
    if info:
        errs, black = decode_check(p)
        add("playable", info["has_video"] and info["duration"] > 0 and not errs, "; ".join(errs[:2]))
        w, h = info["width"], info["height"]
        add("resolution", (w, h) == expected_wh, f"{w}x{h} (expected {expected_wh[0]}x{expected_wh[1]})")
        add("aspect_9_16", w and h and abs(w / h - 9 / 16) < 0.005, f"{w}/{h}")
        add("frame_rate", info["fps"] and abs(info["fps"] - expected_fps) <= 0.6, f"{info['fps']} fps")
        add("audio_present", info["has_audio"])
        if info["has_audio"]:
            vol = audio_mean_volume(p)
            add("audio_audible", vol is not None and vol > -55, f"mean {vol} dB")
        lo, hi = cfg["min_duration_s"], cfg["max_duration_s"]
        add("duration_within_limits", lo - 0.5 <= info["duration"] <= hi + 0.5, f"{info['duration']:.1f}s limits {lo}-{hi}s")
        add("duration_matches_clip", abs(info["duration"] - clip.duration) <= 0.6,
            f"{info['duration']:.2f}s vs planned {clip.duration:.2f}s")
        bad_black = [(a, b) for a, b in black if b - a >= 0.5]
        add("no_black_frames", not bad_black, f"black spans: {bad_black[:3]}")
        digest = sha256_file(p)
        dup = [k for k, v in other_hashes.items() if v == digest and k != str(p)]
        add("no_duplicate_output", not dup, dup[:1])
    # boundaries / context
    for r in validate_clip(clip, media_duration, cfg["min_duration_s"], cfg["max_duration_s"], sentences, others):
        add(r["check"], r["ok"], r["detail"], r["level"])
    for k in others:
        if k is not clip and is_duplicate(clip.text, k.text, 0.6):
            add("not_near_duplicate_of_other_clip", False, f"text overlaps clip at {k.start:.0f}s")
    for wn in clip.warnings:
        add("context_warning", False, wn, "warn")
    # captions
    if srt_path and Path(srt_path).is_file():
        text = Path(srt_path).read_text(encoding="utf-8")
        probs = validate_srt_text(text)
        add("captions_valid_srt", not probs, "; ".join(probs[:3]))
        cues = parse_srt(text)
        if info and cues:
            add("captions_within_video", cues[-1].end <= info["duration"] + 0.3 and cues[0].start >= 0,
                f"last cue ends {cues[-1].end:.2f}s")
        rd = [i for i in readability_check(cues, max_cps=22, max_line_chars=30, max_lines=3, min_dur=0.3)]
        add("captions_readable", not [i for i in rd if i["severity"] == "error"], rd[:2], "fail")
        if [i for i in rd if i["severity"] == "warn"]:
            add("captions_readability_warnings", False, f"{len(rd)} notes, e.g. {rd[0]['issue']}", "warn")
        if caption_timing and caption_timing != "verified_word":
            add("caption_timing_is_verified_word_level", False, f"timing is {caption_timing}", "warn")
    else:
        add("captions_valid_srt", False, "no caption file")
    # metadata
    add("metadata_saved", bool(metadata_path) and Path(metadata_path).is_file(), metadata_path)
    if metadata:
        for r in validate_metadata(metadata, clip.text, None):
            add("meta_" + r["check"], r["ok"], r["detail"], r["level"])
    failed = [c for c in checks if not c["ok"] and c["level"] == "fail"]
    return {"passed": not failed, "checks": checks, "failed": [c["check"] for c in failed],
            "warnings": [c for c in checks if not c["ok"] and c["level"] == "warn"]}

"""Mode A quality gate."""
from __future__ import annotations

from pathlib import Path

from core.captions import validate_srt_text
from core.media import audio_mean_volume, decode_check, probe
from scripts.render_video import FORMATS


def qa_video(video: str | Path, fmt: str, narration: dict, srt: str | Path | None, thumb: str | Path | None,
             script_checks: list[dict], meta_checks: list[dict], research: dict, assets: list[dict],
             sub_info: dict | None) -> dict:
    checks: list[dict] = []

    def add(check, ok, detail="", level="fail"):
        checks.append({"check": check, "ok": bool(ok), "detail": str(detail), "level": level})

    v = Path(video)
    add("video_exists", v.is_file() and v.stat().st_size > 1000)
    if v.is_file():
        info = probe(v)
        errs, black = decode_check(v)
        add("playable", info["has_video"] and not errs, "; ".join(errs[:2]))
        w, h = FORMATS[fmt]
        add("resolution", (info["width"], info["height"]) == (w, h), f"{info['width']}x{info['height']} expected {w}x{h}")
        add("audio_present", info["has_audio"])
        if info["has_audio"]:
            vol = audio_mean_volume(v)
            add("audio_audible", vol is not None and vol > -50, f"mean {vol} dB")
        add("duration_matches_narration", abs(info["duration"] - narration["duration"]) < 0.8,
            f"{info['duration']:.1f}s vs {narration['duration']:.1f}s")
        add("no_black_frames", not [b for b in black if b[1] - b[0] > 1.0], black[:3])
    add("real_narration", not narration["placeholder"], "placeholder TTS produces silence, not narration")
    add("narration_timeline_consistent", narration["timeline_drift_s"] < 1.0, f"drift {narration['timeline_drift_s']}s", "warn")
    if not narration["natural_voice"]:
        add("voice_is_basic_synthetic", False, "basic synthetic voice; not a natural voice", "warn")
    if srt and Path(srt).is_file():
        probs = validate_srt_text(Path(srt).read_text(encoding="utf-8"))
        add("subtitles_valid_srt", not probs, "; ".join(probs[:3]))
        add("subtitles_verified_speech_aligned", False,
            "subtitle times are measured per sentence from the synthesized audio, not ASR-verified", "warn")
        for i in (sub_info or {}).get("readability", [])[:3]:
            add("subtitle_readability", False, f"cue {i['cue']}: {i['issue']}", "warn")
    else:
        add("subtitles_valid_srt", False, "missing")
    if thumb:
        t = Path(thumb)
        add("thumbnail_ok", t.is_file() and t.stat().st_size < 2_000_000)
    for r in script_checks + meta_checks:
        add("script_" + r["check"] if r in script_checks else "meta_" + r["check"], r["ok"], r["detail"], r["level"])
    add("research_sources_present", len(research["sources"]) > 0)
    add("uncertain_claims_flagged", True, f"{sum(1 for f in research['facts'] if f['confidence'] != 'supported')} uncertain facts", "warn")
    bad = [a for a in assets if a["license_status"] != "ok"]
    add("asset_licences_clear", not bad, [a["path"] for a in bad][:3], "warn")
    failed = [c["check"] for c in checks if not c["ok"] and c["level"] == "fail"]
    return {"passed": not failed, "checks": checks, "failed": failed,
            "warnings": [c for c in checks if not c["ok"] and c["level"] == "warn"]}

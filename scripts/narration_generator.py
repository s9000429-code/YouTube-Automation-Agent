"""Narration: per-sentence TTS, measured durations, pauses, one WAV + a timeline.

Subtitle timing is derived from the MEASURED length of each synthesized sentence. That is sentence-accurate
for this audio, but word positions inside a sentence are interpolated (see subtitles)."""
from __future__ import annotations

from pathlib import Path

from core.media import concat_audio, make_silence, probe
from providers.tts import TTSProvider, TTSUnavailable


def narrate(script: dict, tts: TTSProvider, out_dir: str | Path, language: str, voice: str | None, speed: int | None,
            pitch: int | None, pause_sentence_ms: int, pause_section_ms: int, progress=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    timeline: list[dict] = []
    t = 0.0
    sil_s = make_silence(out_dir / "_pause_s.wav", pause_sentence_ms / 1000.0)
    sil_x = make_silence(out_dir / "_pause_x.wav", pause_section_ms / 1000.0)
    flat = [(sec, i, s) for sec in script["sections"] for i, s in enumerate(sec["sentences"])]
    for k, (sec, i, s) in enumerate(flat):
        wav = out_dir / f"_seg_{k:04d}.wav"
        tts.synthesize(s["text"], wav, language, voice, speed, pitch)
        dur = probe(wav)["duration"]
        parts.append(wav)
        timeline.append({"section": sec["id"], "index": i, "text": s["text"], "start": round(t, 3),
                         "end": round(t + dur, 3), "fact_ids": s["fact_ids"], "kind": s["kind"]})
        t += dur
        last_in_sec = i == len(sec["sentences"]) - 1
        if k < len(flat) - 1:
            gap = pause_section_ms if last_in_sec else pause_sentence_ms
            parts.append(sil_x if last_in_sec else sil_s)
            t += gap / 1000.0
        if progress:
            progress((k + 1) / len(flat))
    out_wav = out_dir / "narration.wav"
    concat_audio(parts, out_wav, out_dir / "_concat.txt")
    total = probe(out_wav)["duration"]
    for p in out_dir.glob("_seg_*.wav"):
        p.unlink(missing_ok=True)
    for p in (sil_s, sil_x, out_dir / "_concat.txt"):
        Path(p).unlink(missing_ok=True)
    drift = abs(total - t)
    return {"wav": str(out_wav), "duration": total, "timeline": timeline, "tts": tts.describe(),
            "timeline_drift_s": round(drift, 3), "natural_voice": tts.natural,
            "placeholder": tts.kind == "placeholder"}

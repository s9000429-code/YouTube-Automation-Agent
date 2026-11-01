"""Optional ONLINE neural TTS via the `edge-tts` package. Needs internet; sends script text to a
third-party service. Not used unless explicitly selected."""
from __future__ import annotations

import asyncio
from pathlib import Path

from core.media import ffmpeg
from core.resources import has_module

from .base import TTSProvider, TTSUnavailable

DEFAULT_VOICES = {"en": "en-US-AriaNeural", "hi": "hi-IN-SwaraNeural", "kn": "kn-IN-SapnaNeural",
                  "te": "te-IN-ShrutiNeural", "hinglish": "hi-IN-SwaraNeural"}


class EdgeTTS(TTSProvider):
    name = "edge_online"
    kind = "online"
    natural = True
    languages = ("en", "hi", "kn", "te", "hinglish")

    def available(self):
        if not has_module("edge_tts"):
            return False, "pip install edge-tts (requires internet at run time)"
        return True, "ok (network required; unverified offline)"

    def synthesize(self, text, out_path, language, voice=None, speed=None, pitch=None):
        if not has_module("edge_tts"):
            raise TTSUnavailable("edge-tts not installed")
        import edge_tts  # type: ignore

        v = voice or {**DEFAULT_VOICES, **self.config.get("voices", {})}.get(language, DEFAULT_VOICES["en"])
        rate = f"{int(round(((speed or 160) / 160 - 1) * 100)):+d}%"
        hz = f"{int(((pitch if pitch is not None else 50) - 50) * 2):+d}Hz"
        out = Path(out_path)
        mp3 = out.with_suffix(".mp3")

        async def go():
            await edge_tts.Communicate(text, v, rate=rate, pitch=hz).save(str(mp3))

        try:
            asyncio.run(go())
        except Exception as e:  # network/service errors vary
            raise TTSUnavailable(f"edge-tts failed: {e}") from e
        ffmpeg(["-i", mp3, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", out])
        mp3.unlink(missing_ok=True)
        return out

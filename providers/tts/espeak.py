"""eSpeak NG: basic offline formant TTS. Intelligible, NOT natural-sounding, NOT voice cloning."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import TTSProvider, TTSUnavailable

DEFAULT_VOICE_MAP = {"en": "en-us", "hi": "hi", "kn": "kn", "te": "te",
                     # Romanised Hinglish has no dedicated eSpeak voice; 'hi' is the closest guess and is
                     # UNVERIFIED for Latin-script input. Override via providers.json voice_map.
                     "hinglish": "hi"}


class EspeakTTS(TTSProvider):
    name = "espeak"
    kind = "offline"
    natural = False
    languages = ("en", "hi", "kn", "te", "hinglish")

    def _exe(self) -> str | None:
        return shutil.which("espeak-ng") or shutil.which("espeak")

    def available(self):
        return (True, "ok") if self._exe() else (False, "espeak-ng not installed (Termux: pkg install espeak)")

    def synthesize(self, text, out_path, language, voice=None, speed=None, pitch=None):
        exe = self._exe()
        if not exe:
            raise TTSUnavailable("espeak-ng not installed")
        vmap = {**DEFAULT_VOICE_MAP, **self.config.get("voice_map", {})}
        v = voice or vmap.get(language, "en-us")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [exe, "-v", v, "-s", str(int(speed or 160)), "-p", str(int(pitch if pitch is not None else 50)),
               "--stdin", "-w", str(out)]
        cp = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding="utf-8", timeout=120)
        if cp.returncode != 0 or not out.exists() or out.stat().st_size < 100:
            raise TTSUnavailable(f"espeak failed (voice {v}): {cp.stderr.strip()[:200]}")
        return out

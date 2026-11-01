"""Generic external-command TTS (e.g. Piper). Config: {"command": ["piper","--model","x.onnx",
"--output_file","{out}"], "stdin": true}. Placeholders: {out} {voice} {lang} {speed} {pitch} {text}."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import TTSProvider, TTSUnavailable


class CommandTTS(TTSProvider):
    name = "command"
    kind = "offline"
    natural = False  # depends on the engine; the user must declare it
    languages = ("en", "hi", "kn", "te", "hinglish")

    def __init__(self, config=None):
        super().__init__(config)
        self.natural = bool(self.config.get("natural", False))

    def available(self):
        cmd = self.config.get("command")
        if not cmd:
            return False, "no 'command' configured in providers.json -> tts.command"
        if not shutil.which(cmd[0]):
            return False, f"{cmd[0]} not found on PATH"
        return True, "ok"

    def synthesize(self, text, out_path, language, voice=None, speed=None, pitch=None):
        ok, why = self.available()
        if not ok:
            raise TTSUnavailable(why)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        subs = {"{out}": str(out), "{voice}": str(voice or ""), "{lang}": language,
                "{speed}": str(speed or 160), "{pitch}": str(pitch if pitch is not None else 50), "{text}": text}
        cmd = []
        for part in self.config["command"]:
            for k, v in subs.items():
                part = part.replace(k, v)
            cmd.append(part)
        cp = subprocess.run(cmd, input=text if self.config.get("stdin", True) else None, capture_output=True,
                            text=True, encoding="utf-8", timeout=300)
        if cp.returncode != 0 or not out.exists() or out.stat().st_size < 100:
            raise TTSUnavailable(f"TTS command failed: {cp.stderr.strip()[:200]}")
        return out

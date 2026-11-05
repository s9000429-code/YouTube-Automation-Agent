"""Windows built-in offline TTS (System.Speech / SAPI5) driven through PowerShell. No install needed on Windows 10.

Honest scope: these are the classic Windows voices (e.g. David, Zira). They are clearer than eSpeak but are
still basic synthetic voices, NOT neural/natural voices and NOT voice cloning. Hindi/Kannada/Telugu voices exist
only if the matching Windows speech pack is installed (Settings > Time & Language > Speech). If no voice for the
language is installed we fail with instructions instead of silently reading it with a wrong-language voice."""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from core.platform_utils import is_windows

from .base import TTSProvider, TTSUnavailable

# preferred culture prefixes per project language (first match wins)
CULTURES = {"en": ["en-IN", "en-US", "en"], "hinglish": ["en-IN", "en"],   # Latin-script Hinglish: English voice
            "hi": ["hi"], "kn": ["kn"], "te": ["te"]}

SPEAK_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo })
$chosen = $null
if ($env:YTA_VOICE) { $chosen = $voices | Where-Object { $_.Name -eq $env:YTA_VOICE } | Select-Object -First 1 }
if (-not $chosen) {
  foreach ($c in ($env:YTA_CULTURES -split ',')) {
    $chosen = $voices | Where-Object { $_.Culture.Name -like "$c*" } | Select-Object -First 1
    if ($chosen) { break }
  }
}
if (-not $chosen) { [Console]::Error.WriteLine('NO_VOICE'); exit 3 }
$s.SelectVoice($chosen.Name)
$s.Rate = [int]$env:YTA_RATE
$text = [IO.File]::ReadAllText($env:YTA_TEXT, [Text.Encoding]::UTF8)
$s.SetOutputToWaveFile($env:YTA_OUT)
$s.Speak($text)
$s.Dispose()
Write-Output $chosen.Name
"""

LIST_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name + '|' + $_.VoiceInfo.Culture.Name }
"""


def encode_command(script: str) -> str:
    """PowerShell -EncodedCommand wants base64 of UTF-16LE; this avoids every quoting problem."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def rate_from_speed(speed: int | None) -> int:
    """words/min (160 = normal) -> SAPI rate -10..10"""
    return max(-10, min(10, round(((speed or 160) - 160) / 20)))


def _ps() -> str | None:
    return shutil.which("powershell") or shutil.which("pwsh")


def _run_ps(script: str, env: dict | None = None, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([_ps(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-EncodedCommand", encode_command(script)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                          env={**os.environ, **(env or {})})


class SapiTTS(TTSProvider):
    name = "sapi"
    kind = "offline"
    natural = False
    languages = ("en", "hinglish", "hi", "kn", "te")

    def available(self):
        if not is_windows():
            return False, "Windows only (System.Speech)"
        if not _ps():
            return False, "PowerShell not found"
        return True, "ok (voices per language depend on installed Windows speech packs)"

    def list_voices(self) -> list[tuple[str, str]]:
        cp = _run_ps(LIST_SCRIPT, timeout=60)
        return [tuple(l.strip().split("|", 1)) for l in cp.stdout.splitlines() if "|" in l]

    def synthesize(self, text, out_path, language, voice=None, speed=None, pitch=None):
        ok, why = self.available()
        if not ok:
            raise TTSUnavailable(why)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            env = {"YTA_TEXT": tmp, "YTA_OUT": str(out), "YTA_RATE": str(rate_from_speed(speed)),
                   "YTA_VOICE": voice or "", "YTA_CULTURES": ",".join(CULTURES.get(language, CULTURES["en"]))}
            cp = _run_ps(SPEAK_SCRIPT, env)
        finally:
            Path(tmp).unlink(missing_ok=True)
        if cp.returncode == 3 or "NO_VOICE" in cp.stderr:
            raise TTSUnavailable(
                f"no Windows voice installed for language '{language}'. Install one in Settings > Time & Language > "
                f"Speech (add the language's speech pack), or choose another TTS provider.")
        if cp.returncode != 0 or not out.exists() or out.stat().st_size < 100:
            raise TTSUnavailable(f"Windows speech failed: {cp.stderr.strip()[:200]}")
        return out

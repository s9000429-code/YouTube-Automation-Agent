"""TTS provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TTSUnavailable(RuntimeError):
    pass


class TTSProvider(ABC):
    name = "base"
    kind = "offline"          # offline | online | placeholder
    natural = False           # True only for neural voices; basic formant TTS must stay False
    supports_speed = True
    supports_pitch = True
    languages: tuple[str, ...] = ()

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(ok, reason). Must not raise."""

    @abstractmethod
    def synthesize(self, text: str, out_path: str | Path, language: str, voice: str | None = None,
                   speed: int | None = None, pitch: int | None = None) -> Path:
        """Write a WAV file for `text`. speed ~ words/min (default 160), pitch 0-99 (default 50)."""

    def describe(self) -> dict:
        ok, why = self.available()
        return {"name": self.name, "kind": self.kind, "natural_voice": self.natural, "available": ok,
                "reason": why, "languages": list(self.languages)}

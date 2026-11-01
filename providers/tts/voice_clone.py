"""Consent-gated voice cloning INTERFACE.

No cloning engine ships with this project. This module only enforces the rule that a voice may be
cloned solely with a documented, scoped consent record, and lets you plug in a backend later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from .base import TTSProvider, TTSUnavailable


class ConsentError(PermissionError):
    pass


@dataclass
class ConsentRecord:
    speaker_name: str
    reference_audio: str
    statement: str            # e.g. "I, <name>, consent to my voice being cloned for <channel>."
    granted_on: str           # ISO date
    scope: list[str]          # must include "youtube_narration"
    is_own_voice: bool = False
    permission_doc: str | None = None   # path/URL to written permission when not own voice
    expires_on: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "ConsentRecord":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self, today: date | None = None) -> None:
        today = today or date.today()
        if not self.speaker_name.strip():
            raise ConsentError("consent record has no speaker name")
        if "consent" not in self.statement.lower() or len(self.statement.split()) < 6:
            raise ConsentError("consent statement missing or too vague")
        if "youtube_narration" not in self.scope:
            raise ConsentError("consent scope must include 'youtube_narration'")
        if not self.is_own_voice and not self.permission_doc:
            raise ConsentError("cloning someone else's voice requires a written permission_doc")
        if not Path(self.reference_audio).is_file():
            raise ConsentError(f"reference audio not found: {self.reference_audio}")
        try:
            granted = date.fromisoformat(self.granted_on)
        except ValueError as e:
            raise ConsentError("granted_on must be an ISO date (YYYY-MM-DD)") from e
        if granted > today:
            raise ConsentError("consent date is in the future")
        if self.expires_on and date.fromisoformat(self.expires_on) < today:
            raise ConsentError("consent has expired")


_BACKEND: Callable[..., Path] | None = None


def register_backend(fn: Callable[..., Path]) -> None:
    """fn(text, out_path, language, reference_audio, speed, pitch) -> Path"""
    global _BACKEND
    _BACKEND = fn


class VoiceCloneTTS(TTSProvider):
    name = "voice_clone"
    kind = "offline"
    natural = True
    languages = ("en", "hi", "kn", "te", "hinglish")

    def __init__(self, config=None):
        super().__init__(config)
        self.consent: ConsentRecord | None = None
        path = self.config.get("consent_file")
        if path and Path(path).is_file():
            self.consent = ConsentRecord.load(path)

    def available(self):
        if _BACKEND is None:
            return False, "no voice-cloning backend registered (none ships with this project)"
        if self.consent is None:
            return False, "no consent record (config tts.consent_file)"
        try:
            self.consent.validate()
        except ConsentError as e:
            return False, f"consent invalid: {e}"
        return True, "ok"

    def synthesize(self, text, out_path, language, voice=None, speed=None, pitch=None):
        if self.consent is None:
            raise ConsentError("voice cloning refused: no consent record")
        self.consent.validate()
        if _BACKEND is None:
            raise TTSUnavailable("no voice-cloning backend registered")
        return _BACKEND(text, Path(out_path), language, self.consent.reference_audio, speed, pitch)

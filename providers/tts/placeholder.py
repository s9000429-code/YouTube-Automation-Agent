"""Silent placeholder 'TTS': produces SILENCE sized to the text so timelines/renders can be previewed.
It is NOT speech. It is only usable when settings.topic_to_video.allow_placeholder_tts is true, and
anything built with it is flagged by QA."""
from __future__ import annotations

from pathlib import Path

from core.media import make_silence

from .base import TTSProvider


class PlaceholderTTS(TTSProvider):
    name = "placeholder_silent"
    kind = "placeholder"
    natural = False
    languages = ("en", "hi", "kn", "te", "hinglish")

    def available(self):
        if self.config.get("allow", False):
            return True, "ok (silence only - not real narration)"
        return False, "disabled: set topic_to_video.allow_placeholder_tts=true to preview without a TTS engine"

    def synthesize(self, text, out_path, language, voice=None, speed=None, pitch=None):
        words = max(1, len(text.split()))
        seconds = max(0.6, words / ((speed or 160) / 60.0))
        return make_silence(Path(out_path), seconds, 22050)

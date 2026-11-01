from __future__ import annotations

from .base import TTSProvider, TTSUnavailable
from .command_tts import CommandTTS
from .edge_online import EdgeTTS
from .espeak import EspeakTTS
from .placeholder import PlaceholderTTS
from .sapi import SapiTTS
from .voice_clone import ConsentError, ConsentRecord, VoiceCloneTTS

REGISTRY = {"espeak": EspeakTTS, "sapi": SapiTTS, "edge_online": EdgeTTS, "command": CommandTTS,
            "placeholder_silent": PlaceholderTTS, "voice_clone": VoiceCloneTTS}
AUTO_ORDER = ["espeak", "sapi"]     # offline engines only: 'auto' never sends text to an online service


def get_tts(name: str, config: dict | None = None) -> TTSProvider:
    if name == "auto":
        for n in AUTO_ORDER:
            t = REGISTRY[n]((config or {}))
            if t.available()[0]:
                return t
        raise TTSUnavailable("no offline TTS found: install espeak-ng, or on Windows use the built-in 'sapi' voice")
    if name not in REGISTRY:
        raise KeyError(f"unknown TTS provider {name!r}; options: {sorted(REGISTRY)} or 'auto'")
    return REGISTRY[name](config or {})


def list_tts(configs: dict | None = None) -> list[dict]:
    configs = configs or {}
    return [REGISTRY[n](configs.get(n, {})).describe() for n in REGISTRY]


__all__ = ["get_tts", "list_tts", "TTSProvider", "TTSUnavailable", "ConsentError", "ConsentRecord", "REGISTRY"]

"""Settings loading: defaults merged with config/settings.json (never overwritten)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SETTINGS: dict = {
    "paths": {"outputs": "outputs", "assets": "assets", "projects": "projects",
              "backups": "backups", "logs": "logs", "database": "projects/agent.db"},
    "languages": ["en", "hi", "kn", "te", "hinglish"],
    "default_language": "en",
    "formats": {
        "landscape": {"width": 1920, "height": 1080},
        "shorts": {"width": 1080, "height": 1920},
        "square": {"width": 1080, "height": 1080},
    },
    "fps": 30,
    "encoding": {"preset": "veryfast", "crf": 23, "audio_bitrate": "128k", "preview_crf": 32},
    "storage": {"reserve_mb": 512},
    "topic_to_video": {
        "style": "educational", "duration_s": 120, "format": "landscape",
        "words_per_minute": {"en": 150, "hinglish": 140, "hi": 130, "kn": 120, "te": 120},
        "pause_between_sentences_ms": 250, "pause_between_sections_ms": 600,
        "allow_placeholder_tts": False,
        "tts": {"provider": "auto", "speed": 160, "pitch": 50, "voice": None},
        "research": {"providers": ["manual"], "max_sources": 6},
        "llm": {"provider": "none"},
        "music": {"path": None, "volume": 0.12},
    },
    "shorts": {
        "count": "auto", "target_duration_s": 45, "min_duration_s": 15, "max_duration_s": 60,
        "caption_style": "bold_pop", "editing_style": "podcast", "clip_selection": "auto",
        "reframe": "auto", "zoom": False, "hook_text": True, "music": False,
        "generate_metadata": True, "normalize_audio": True,
        "width": 1080, "height": 1920, "fps": 30, "quality": "standard",
        "transcription": {"engine": "auto", "chunk_s": 600, "model_path": None,
                          "whisper_binary": None},
    },
    "quality_profiles": {
        "draft": {"preset": "ultrafast", "crf": 30},
        "standard": {"preset": "veryfast", "crf": 23},
        "high": {"preset": "medium", "crf": 20},
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_settings(root: str | Path | None = None, overrides: dict | None = None) -> dict:
    root = Path(root) if root else ROOT
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    cfg = root / "config" / "settings.json"
    if cfg.exists():
        try:
            settings = deep_merge(settings, json.loads(cfg.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            raise ValueError(f"{cfg} is not valid JSON: {e}") from e
    if overrides:
        settings = deep_merge(settings, overrides)
    settings["_root"] = str(root)
    return settings


def resolve_path(settings: dict, key: str) -> Path:
    p = Path(settings["paths"][key])
    return p if p.is_absolute() else Path(settings["_root"]) / p

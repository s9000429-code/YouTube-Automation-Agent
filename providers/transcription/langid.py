"""Lightweight language/script identification for transcripts (used only when the ASR gives no language)."""
from __future__ import annotations

import re

HINGLISH_WORDS = {
    "hai", "hain", "nahi", "nahin", "kya", "toh", "to", "aur", "ki", "ka", "ke", "ko", "mein", "main", "se",
    "yeh", "ye", "woh", "wo", "bhi", "tha", "thi", "hoga", "karna", "kar", "kuch", "bahut", "accha", "acha",
    "matlab", "lekin", "par", "isliye", "kyunki", "jab", "tab", "abhi", "sab", "log", "apna", "aap", "tum",
    "hum", "mera", "meri", "tera", "uska", "chahiye", "raha", "rahe", "rahi", "gaya", "gayi", "liye", "wala",
}


def script_counts(text: str) -> dict[str, int]:
    return {
        "devanagari": len(re.findall(r"[ऀ-ॿ]", text)),
        "kannada": len(re.findall(r"[ಀ-೿]", text)),
        "telugu": len(re.findall(r"[ఀ-౿]", text)),
        "latin": len(re.findall(r"[A-Za-z]", text)),
    }


def detect_language(text: str) -> str:
    """Return one of en, hi, kn, te, hinglish (best effort, text-only)."""
    c = script_counts(text)
    total = sum(c.values()) or 1
    for script, code in (("devanagari", "hi"), ("kannada", "kn"), ("telugu", "te")):
        if c[script] / total > 0.3:
            return code
    words = re.findall(r"[a-z']+", text.lower())
    if words:
        share = sum(1 for w in words if w in HINGLISH_WORDS) / len(words)
        if share >= 0.12:
            return "hinglish"
    return "en"


def normalize_code(code: str | None) -> str | None:
    """Map engine language codes to project codes."""
    if not code:
        return None
    code = code.lower()
    return {"english": "en", "hindi": "hi", "kannada": "kn", "telugu": "te"}.get(code, code.split("-")[0])

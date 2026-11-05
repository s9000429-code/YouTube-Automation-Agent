"""Transcript data model + engine interface. Times are always seconds in ORIGINAL video time."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from core.safe_io import atomic_write_json


class EngineUnavailable(RuntimeError):
    pass


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float | None = None


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None


@dataclass
class Transcript:
    language: str | None
    engine: str
    segments: list[Segment]
    timing: str = "segment"        # "word" if every segment has words, else "segment"
    duration: float | None = None
    meta: dict = field(default_factory=dict)

    # -- helpers --
    def has_words(self) -> bool:
        return bool(self.segments) and all(s.words for s in self.segments)

    def all_words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    def refresh_timing(self) -> None:
        self.timing = "word" if self.has_words() else "segment"

    def shifted(self, offset: float) -> "Transcript":
        segs = [Segment(s.id, s.start + offset, s.end + offset, s.text,
                        [Word(w.text, w.start + offset, w.end + offset, w.prob) for w in s.words], s.speaker)
                for s in self.segments]
        return Transcript(self.language, self.engine, segs, self.timing, self.duration, dict(self.meta))

    def window(self, start: float, end: float) -> list[Segment]:
        return [s for s in self.segments if s.end > start and s.start < end]

    # -- (de)serialisation --
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        segs = [Segment(s["id"], s["start"], s["end"], s["text"],
                        [Word(**w) for w in s.get("words", [])], s.get("speaker")) for s in d["segments"]]
        t = cls(d.get("language"), d.get("engine", "unknown"), segs, d.get("timing", "segment"),
                d.get("duration"), d.get("meta", {}))
        t.refresh_timing()
        return t

    def save(self, path: str | Path) -> Path:
        return atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self, media_duration: float | None = None, tol: float = 0.25) -> list[str]:
        problems: list[str] = []
        if not self.segments:
            return ["transcript has no segments"]
        prev_end, prev_start = 0.0, -1.0
        for s in self.segments:
            if s.end < s.start:
                problems.append(f"segment {s.id}: end < start")
            if s.start < prev_start - 1e-6:
                problems.append(f"segment {s.id}: not in chronological order")
            if s.start < prev_end - tol:
                problems.append(f"segment {s.id}: overlaps previous by {prev_end - s.start:.2f}s")
            if media_duration is not None and s.end > media_duration + tol:
                problems.append(f"segment {s.id}: ends at {s.end:.2f}s, after media end {media_duration:.2f}s")
            if s.start < -1e-6:
                problems.append(f"segment {s.id}: negative start")
            pw = None
            for w in s.words:
                if w.end < w.start:
                    problems.append(f"segment {s.id}: word {w.text!r} end < start")
                if pw is not None and w.start < pw - tol:
                    problems.append(f"segment {s.id}: word {w.text!r} out of order")
                if w.start < s.start - tol or w.end > s.end + tol:
                    problems.append(f"segment {s.id}: word {w.text!r} outside its segment")
                pw = w.end
            prev_end, prev_start = max(prev_end, s.end), s.start
        return problems


class TranscriptionEngine(ABC):
    name = "base"
    word_timestamps = False
    local = True

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        ...

    @abstractmethod
    def transcribe(self, audio_path: str | Path, language: str | None = None, offset: float = 0.0,
                   progress: Callable[[float], None] | None = None) -> Transcript:
        """Transcribe one audio file; returned times already include `offset`."""

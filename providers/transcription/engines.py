"""Concrete transcription engines. Models are NEVER downloaded automatically."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from core.resources import check_model_fits, disk_free_mb, has_module, mem_available_mb

from .base import EngineUnavailable, Segment, Transcript, TranscriptionEngine, Word
from .langid import normalize_code
from .parsers import load_transcript_file, parse_whisper_cpp_json


def _model_size_from_name(path: str) -> str | None:
    m = re.search(r"(tiny|base|small|medium|large)", Path(path).name.lower())
    return m.group(1) if m else None


class WhisperCppEngine(TranscriptionEngine):
    """Wraps the whisper.cpp CLI (`whisper-cli`). Needs a ggml model file YOU downloaded."""
    name = "whisper_cpp"
    word_timestamps = True

    def _binary(self) -> str | None:
        b = self.config.get("binary")
        if b:
            return b if shutil.which(b) or Path(b).is_file() else None
        return shutil.which("whisper-cli") or shutil.which("whisper-cpp") or shutil.which("whisper.cpp")

    def available(self):
        if not self._binary():
            return False, "whisper.cpp binary not found (build it, then set shorts.transcription.whisper_binary)"
        mp = self.config.get("model_path")
        if not mp or not Path(mp).is_file():
            return False, ("whisper.cpp model file not set/found (shorts.transcription.model_path). "
                           "Download one manually, e.g. ggml-base.bin (~142 MB) or ggml-small.bin (~466 MB)")
        size = _model_size_from_name(mp)
        if size:
            ok, why = check_model_fits(size, mem_available_mb(), disk_free_mb(mp))
            if not ok:
                return False, why
        return True, "ok"

    def transcribe(self, audio_path, language=None, offset=0.0, progress=None):
        ok, why = self.available()
        if not ok:
            raise EngineUnavailable(why)
        lang = {"hinglish": "hi"}.get(language or "", language) or "auto"   # Hinglish audio is Hindi to the model
        with tempfile.TemporaryDirectory(prefix="wcpp_") as td:
            prefix = Path(td) / "out"
            cmd = [self._binary(), "-m", self.config["model_path"], "-f", str(audio_path), "-l", lang,
                   "-ojf", "-of", str(prefix), "-t", str(self.config.get("threads", 4)), "-np"]
            cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            out = prefix.with_suffix(".json")
            if cp.returncode != 0 or not out.exists():
                raise EngineUnavailable(f"whisper.cpp failed: {cp.stderr.strip()[-300:]}")
            data = json.loads(out.read_text(encoding="utf-8"))
        t = parse_whisper_cpp_json(data, offset, self.name)
        t.language = normalize_code(t.language) or (language if language != "auto" else None)
        if progress:
            progress(1.0)
        return t


class FasterWhisperEngine(TranscriptionEngine):
    """faster-whisper (CTranslate2). Uses ONLY a local model dir; never downloads."""
    name = "faster_whisper"
    word_timestamps = True

    def available(self):
        if not has_module("faster_whisper"):
            return False, "faster-whisper not installed"
        mp = self.config.get("model_path")
        if not mp or not Path(mp).is_dir():
            return False, "set shorts.transcription.model_path to a local faster-whisper model directory"
        return True, "ok"

    def transcribe(self, audio_path, language=None, offset=0.0, progress=None):
        ok, why = self.available()
        if not ok:
            raise EngineUnavailable(why)
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel(self.config["model_path"], device="cpu", compute_type="int8",
                             local_files_only=True)
        lang = {"hinglish": "hi"}.get(language or "", language)
        gen, info = model.transcribe(str(audio_path), language=None if lang in (None, "auto") else lang,
                                     word_timestamps=True, vad_filter=True)
        segs = []
        for s in gen:
            words = [Word(w.word.strip(), w.start + offset, w.end + offset, w.probability)
                     for w in (s.words or []) if w.word.strip()]
            if s.text.strip():
                segs.append(Segment(len(segs), s.start + offset, s.end + offset, s.text.strip(), words))
            if progress and info.duration:
                progress(min(1.0, s.end / info.duration))
        t = Transcript(normalize_code(info.language), self.name, segs)
        t.refresh_timing()
        return t


class ImportedTranscriptEngine(TranscriptionEngine):
    """Use a transcript the user supplies (.json/.srt/.vtt). Timestamps are as accurate as their source."""
    name = "imported"
    word_timestamps = False

    def available(self):
        p = self.config.get("path")
        return (True, "ok") if p and Path(p).is_file() else (False, "no transcript file given")

    def transcribe(self, audio_path, language=None, offset=0.0, progress=None):
        t = load_transcript_file(self.config["path"], offset)
        t.engine = f"imported:{t.engine}"
        return t


REGISTRY = {"whisper_cpp": WhisperCppEngine, "faster_whisper": FasterWhisperEngine,
            "imported": ImportedTranscriptEngine}


def get_engine(name: str, config: dict | None = None) -> TranscriptionEngine:
    if name not in REGISTRY:
        raise KeyError(f"unknown transcription engine {name!r}; options: {sorted(REGISTRY)}")
    return REGISTRY[name](config or {})


def choose_engine(settings_tx: dict, transcript_file: str | None = None) -> TranscriptionEngine:
    """Pick an engine. A supplied transcript wins; otherwise the first locally available engine."""
    if transcript_file:
        return ImportedTranscriptEngine({"path": transcript_file})
    wanted = settings_tx.get("engine", "auto")
    cfg = {"model_path": settings_tx.get("model_path"), "binary": settings_tx.get("whisper_binary"),
           "threads": settings_tx.get("threads", 4)}
    names = [wanted] if wanted != "auto" else ["whisper_cpp", "faster_whisper"]
    reasons = []
    for n in names:
        eng = get_engine(n, cfg)
        ok, why = eng.available()
        if ok:
            return eng
        reasons.append(f"{n}: {why}")
    raise EngineUnavailable("no transcription engine available -> " + " | ".join(reasons) +
                            ". Alternatively pass an existing transcript with --transcript FILE.")

"""Research providers return real, traceable sources only. Nothing is generated or guessed here."""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


@dataclass
class Source:
    url: str
    title: str
    publisher: str = ""
    published: str = ""
    license: str = "unknown"
    extract: str = ""
    language: str = "en"
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class ResearchProvider(ABC):
    name = "base"
    needs_network = False

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def search(self, query: str, language: str = "en", limit: int = 5) -> list[Source]:
        ...


def _http_json(url: str, timeout: float = 20.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "YouTubeAutomationAgent/1.0 (research; contact: local)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class WikipediaProvider(ResearchProvider):
    """Wikipedia via the public MediaWiki API (text is CC BY-SA 4.0 and must be attributed)."""
    name = "wikipedia"
    needs_network = True
    LANG = {"en": "en", "hinglish": "en", "hi": "hi", "kn": "kn", "te": "te"}

    def __init__(self, config=None, fetch: Callable[[str], dict] | None = None):
        super().__init__(config)
        self._fetch = fetch or _http_json

    def search(self, query, language="en", limit=5):
        wl = self.LANG.get(language, "en")
        params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": query,
                  "gsrlimit": str(limit), "prop": "extracts|info", "exintro": "1", "explaintext": "1",
                  "inprop": "url", "redirects": "1"}
        url = f"https://{wl}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        data = self._fetch(url)
        pages = sorted((data.get("query", {}).get("pages", {}) or {}).values(), key=lambda p: p.get("index", 0))
        out = []
        for p in pages:
            ext = (p.get("extract") or "").strip()
            if len(ext) < 80:
                continue
            out.append(Source(url=p.get("fullurl") or f"https://{wl}.wikipedia.org/?curid={p.get('pageid')}",
                              title=p.get("title", ""), publisher="Wikipedia", published=(p.get("touched") or "")[:10],
                              license="CC BY-SA 4.0", extract=ext, language=wl))
        return out


class ManualProvider(ResearchProvider):
    """Reads local notes/articles you provide. Optional header lines: Title:, URL:, Publisher:, Date:, License:.
    Without a URL the source is recorded as a local file (still traceable, flagged as non-public)."""
    name = "manual"

    def search(self, query, language="en", limit=5):
        paths = [Path(p) for p in self.config.get("paths", [])]
        files: list[Path] = []
        for p in paths:
            if p.is_dir():
                files += sorted(x for x in p.rglob("*") if x.suffix.lower() in (".txt", ".md"))
            elif p.is_file():
                files.append(p)
        out = []
        for f in files:
            raw = f.read_text(encoding="utf-8", errors="replace")
            head, body = {}, raw
            m = re.match(r"((?:[A-Za-z]+:.*\n)+)\n", raw)
            if m:
                for line in m.group(1).splitlines():
                    k, _, v = line.partition(":")
                    head[k.strip().lower()] = v.strip()
                body = raw[m.end():]
            out.append(Source(url=head.get("url") or f.resolve().as_uri(), title=head.get("title") or f.stem,
                              publisher=head.get("publisher", ""), published=head.get("date", ""),
                              license=head.get("license", "unknown"), extract=body.strip(), language=language))
        return out[:limit] if limit else out


REGISTRY = {"wikipedia": WikipediaProvider, "manual": ManualProvider}


def get_research(name: str, config: dict | None = None) -> ResearchProvider:
    if name not in REGISTRY:
        raise KeyError(f"unknown research provider {name!r}; options: {sorted(REGISTRY)}")
    return REGISTRY[name](config or {})

"""Research: collect real sources, extract quoted facts, track provenance, flag weak claims.

Facts are EXTRACTED sentences from fetched/provided text (verbatim), each tied to a source. Nothing is
generated here, so nothing can be fabricated. A fact is flagged when it is a single-source number, hedged,
or comes from a source with an unknown licence."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from core.job_manager import JobManager
from providers.research import ResearchProvider, Source, get_research

SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+(?=[A-Z0-9ऀ-ॿಀ-೿ఀ-౿\"“])")
NUM = re.compile(r"\d")
HEDGE = re.compile(r"\b(reportedly|allegedly|may|might|could|possibly|believed to|claimed|some say|rumou?red|"
                   r"estimated|approximately|about|around)\b", re.I)
JUNK = re.compile(r"(\[\d+\]|\(\s*[;,]?\s*\)|\bcitation needed\b)", re.I)


@dataclass
class Fact:
    text: str
    source_idx: int
    kind: str                       # statistic | explanation | example | definition
    confidence: str                 # supported | uncertain
    flags: list[str] = field(default_factory=list)


def _kind(s: str) -> str:
    if NUM.search(s):
        return "statistic"
    if re.search(r"\b(is|are|was|were|refers to|means|known as)\b", s[:80], re.I):
        return "definition"
    if re.search(r"\b(for example|such as|for instance)\b", s, re.I):
        return "example"
    return "explanation"


def extract_facts(sources: list[Source], query: str, max_per_source: int = 8) -> list[Fact]:
    qwords = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 3}
    facts: list[Fact] = []
    seen: set[str] = set()
    for si, src in enumerate(sources):
        cands = []
        for raw in SENT_SPLIT.split(src.extract.replace("\n", " ")):
            s = JUNK.sub("", raw).strip()
            if not (40 <= len(s) <= 320):
                continue
            key = re.sub(r"\W+", "", s.lower())[:80]
            if key in seen:
                continue
            overlap = len(qwords & set(re.findall(r"\w+", s.lower())))
            cands.append((overlap + (0.5 if NUM.search(s) else 0), s, key))
        for _, s, key in sorted(cands, key=lambda x: -x[0])[:max_per_source]:
            seen.add(key)
            flags = []
            if HEDGE.search(s):
                flags.append("hedged_wording")
            if src.license in ("unknown", ""):
                flags.append("source_license_unknown")
            facts.append(Fact(s, si, _kind(s), "uncertain" if "hedged_wording" in flags else "supported", flags))
    # a number only one source states is "uncertain" until a second source agrees
    nums_by_fact = [set(re.findall(r"\d[\d,\.]*", f.text)) for f in facts]
    for i, f in enumerate(facts):
        if f.kind == "statistic" and nums_by_fact[i]:
            corroborated = any(j != i and facts[j].source_idx != f.source_idx and nums_by_fact[i] & nums_by_fact[j]
                               for j in range(len(facts)))
            if not corroborated:
                f.flags.append("single_source_number")
                f.confidence = "uncertain"
    return facts


class ResearchManager:
    def __init__(self, jm: JobManager, settings: dict, providers: list[ResearchProvider] | None = None):
        self.jm, self.settings = jm, settings
        cfg = settings["topic_to_video"]["research"]
        self.providers = providers if providers is not None else [get_research(n, cfg.get(n, {})) for n in cfg["providers"]]
        self.max_sources = cfg.get("max_sources", 6)

    def run(self, job_id: str, topic: str, language: str = "en") -> dict:
        sources: list[Source] = []
        errors: list[str] = []
        for p in self.providers:
            try:
                sources += p.search(topic, language, self.max_sources)
            except Exception as e:  # network / provider errors must not kill the job silently
                errors.append(f"{p.name}: {type(e).__name__}: {e}")
        uniq, seen = [], set()
        for s in sources:
            if s.url not in seen:
                seen.add(s.url)
                uniq.append(s)
        uniq = uniq[: self.max_sources]
        facts = extract_facts(uniq, topic)
        c = self.jm.conn
        c.execute("DELETE FROM research_facts WHERE job_id=?", (job_id,))
        c.execute("DELETE FROM research_sources WHERE job_id=?", (job_id,))
        ids = []
        for s in uniq:
            cur = c.execute("INSERT INTO research_sources(job_id,url,title,publisher,published,retrieved_at,license,extract)"
                            " VALUES (?,?,?,?,?,?,?,?)",
                            (job_id, s.url, s.title, s.publisher, s.published, s.retrieved_at, s.license, s.extract[:4000]))
            ids.append(cur.lastrowid)
        fact_ids = []
        for f in facts:
            cur = c.execute("INSERT INTO research_facts(job_id,source_id,text,kind,confidence,supported,flag) VALUES (?,?,?,?,?,?,?)",
                            (job_id, ids[f.source_idx], f.text, f.kind, f.confidence,
                             int(f.confidence == "supported"), ",".join(f.flags)))
            fact_ids.append(cur.lastrowid)
        report = {"topic": topic, "language": language, "errors": errors,
                  "sources": [{"id": i, **{k: v for k, v in asdict(s).items() if k != "extract"}} for i, s in zip(ids, uniq)],
                  "facts": [{"id": fid, "source_id": ids[f.source_idx], **asdict(f)} for fid, f in zip(fact_ids, facts)],
                  "needs_more_research": len(facts) < 4,
                  "notes": ["Facts are verbatim extracts. 'uncertain' facts are not used as firm claims by the script writer."]}
        return report

    def load(self, job_id: str) -> dict:
        c = self.jm.conn
        srcs = [dict(r) for r in c.execute("SELECT id,url,title,publisher,published,license FROM research_sources WHERE job_id=?", (job_id,))]
        facts = [dict(r) for r in c.execute("SELECT id,source_id,text,kind,confidence,flag FROM research_facts WHERE job_id=?", (job_id,))]
        return {"sources": srcs, "facts": facts}

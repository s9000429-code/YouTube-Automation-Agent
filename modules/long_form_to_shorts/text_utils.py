"""Sentence building and text helpers shared by understanding, highlighting and captions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from providers.transcription.base import Transcript, Word

TERMINAL = re.compile(r"[.!?…।॥]+[\"'”’)\]]*$")
SPLIT_RE = re.compile(r"(?<=[.!?…।॥])\s+")

STOPWORDS = set("""a an the and or but if then so of to in on at for from by with about as into over after before is are was were be
been being it its this that these those i you he she we they me my your our their him her them what which who whom how why
when where do does did done have has had not no yes can could will would should may might just very really also than too
up out there here more most some any all each other such only own same s t don im ive id its thats theres hes shes
um uh like you know okay ok well yeah right actually basically literally kind sort lot get got going go gonna""".split())
STOPWORDS_HI = set("""hai hain ho hoga hogi tha thi the ka ki ke ko se me mein main mai par pe aur ya toh to yeh ye woh wo is us in un
bhi hi nahi nahin kya kyun kyunki ek do jo jab tab ab abhi phir kuch sab log apna apne aap tum hum mera meri tera teri uska
uski raha rahe rahi gaya gayi liye wala wali karna kar karte karta karti hota hoti hote
है हैं हो था थी थे का की के को से में पर और या तो यह ये वह वो इस उस भी ही नहीं क्या एक जो जब तब अब फिर कुछ सब""".split())
FILLERS = {"um", "uh", "umm", "uhh", "hmm", "mm", "er", "ah"}


@dataclass
class Sentence:
    idx: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    timing: str = "word"              # word | interpolated
    segment_id: int = 0
    gap_after: float = 0.0
    gap_before: float = 0.0
    ends_clean: bool = True            # terminal punctuation or a real pause

    @property
    def duration(self) -> float:
        return self.end - self.start


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if len(t) > 2 and t not in STOPWORDS and t not in STOPWORDS_HI
            and not t.isdigit()]


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _finish(sents: list[Sentence]) -> list[Sentence]:
    for i, s in enumerate(sents):
        s.idx = i
        s.gap_after = max(0.0, sents[i + 1].start - s.end) if i + 1 < len(sents) else 0.0
        s.gap_before = max(0.0, s.start - sents[i - 1].end) if i > 0 else s.start
        s.ends_clean = bool(TERMINAL.search(s.text.strip())) or s.gap_after >= 0.7 or i == len(sents) - 1
    return sents


def build_sentences(tr: Transcript, pause_split: float = 0.9, max_words: int = 45) -> list[Sentence]:
    """Turn a transcript into timed sentences. Uses word times when present, else interpolates
    inside each segment (flagged timing='interpolated')."""
    sents: list[Sentence] = []
    if tr.has_words():
        cur: list[Word] = []
        seg_of: dict[int, int] = {}
        flat: list[tuple[Word, int]] = [(w, s.id) for s in tr.segments for w in s.words]

        def close():
            nonlocal cur
            if cur:
                text = " ".join(w.text for w in cur)
                sents.append(Sentence(0, cur[0].start, cur[-1].end, text, list(cur), "word", seg_of.get(id(cur[0]), 0)))
                cur = []

        for i, (w, sid) in enumerate(flat):
            seg_of[id(w)] = sid
            cur.append(w)
            nxt = flat[i + 1][0] if i + 1 < len(flat) else None
            gap = (nxt.start - w.end) if nxt else 99
            if TERMINAL.search(w.text) or gap >= pause_split or len(cur) >= max_words:
                close()
        close()
    else:
        for s in tr.segments:
            parts = [p.strip() for p in SPLIT_RE.split(s.text.strip()) if p.strip()]
            total = sum(len(p) for p in parts) or 1
            t = s.start
            for p in parts:
                dur = (s.end - s.start) * len(p) / total
                sents.append(Sentence(0, t, t + dur, p, [], "interpolated", s.id))
                t += dur
    return _finish(sents)


def global_keywords(sentences: list[Sentence], top: int = 15) -> list[tuple[str, float]]:
    from collections import Counter
    c: Counter = Counter()
    for s in sentences:
        c.update(set(content_tokens(s.text)))
    n = max(1, len(sentences))
    scored = [(w, cnt * (1.0 if cnt < 0.5 * n else 0.3)) for w, cnt in c.items() if cnt >= 2]
    return sorted(scored, key=lambda x: -x[1])[:top]


def fmt_hms(t: float) -> str:
    t = int(round(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

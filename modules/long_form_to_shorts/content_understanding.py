"""Transcript understanding: topics, sub-topics, and per-sentence content tags.

HONEST SCOPE: this is rule/lexicon + statistics based (no LLM required). It tags English, Hindi (Devanagari),
and romanised Hinglish cues, plus language-independent signals (numbers, questions, punctuation). Kannada
and Telugu get only the language-independent signals unless you add cue lists to CUES. An optional LLM can
add commentary but never overrides timestamps. Visual understanding is limited to face detection used
later for reframing; scene content is not analysed.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from .text_utils import Sentence, content_tokens, global_keywords, tokenize

CUES: dict[str, list[str]] = {
    "surprise": ["surprising", "shocking", "unbelievable", "never", "actually", "turns out", "secret", "mistake",
                 "wrong", "nobody", "no one", "biggest", "worst", "insane", "crazy", "weird", "wow", "amazing",
                 "incredible", "truth", "myth", "unexpected", "hidden", "the real reason",
                 "hairaan", "sach", "galat", "raaz", "kamaal", "pagal", "kabhi nahi", "koi nahi",
                 "सच", "गलत", "हैरान", "राज़", "राज", "अविश्वसनीय", "कभी नहीं", "कोई नहीं"],
    "emotion": ["love", "hate", "afraid", "fear", "angry", "cried", "tears", "happy", "proud", "heartbreaking",
                "scared", "passion", "lost", "pain", "struggle", "grateful", "regret", "hurt", "dream", "alone",
                "pyaar", "pyar", "dar ", "dard", "khushi", "gussa", "rona", "ummeed",
                "प्यार", "डर", "दर्द", "खुशी", "गुस्सा", "उम्मीद", "सपना"],
    "story": ["when i was", "one day", "years ago", "the story", "remember when", "back then", "i was ", "we were",
              "and then", "suddenly", "the first time", "growing up", "it all started", "that night", "i remember",
              "ek baar", "ek din", "saal pehle", "bachpan", "mujhe yaad", "pehli baar",
              "एक बार", "एक दिन", "साल पहले", "बचपन", "पहली बार", "मुझे याद"],
    "resolution": ["in the end", "finally", "turned out", "that's when", "that is when", "ever since", "learned",
                   "aakhir", "ant mein", "tab se", "seekha", "आखिर", "अंत में", "तब से", "सीखा"],
    "opinion": ["i think", "i believe", "in my opinion", "honestly", "i disagree", "i don't think", "should",
                "must", "the problem is", "overrated", "underrated", "wrong about",
                "mujhe lagta", "meri nazar", "mera maanna", "mere hisaab", "मुझे लगता", "मेरा मानना", "मेरे हिसाब"],
    "takeaway": ["the key", "the lesson", "takeaway", "remember", "important", "here's the thing", "the secret",
                 "what matters", "in short", "that's why", "bottom line", "the rule", "tip", "step one", "rule of",
                 "yaad rakho", "sabse zaroori", "zaroori", "isliye", "seekh", "sabak",
                 "याद रखो", "सबसे ज़रूरी", "सबसे जरूरी", "इसलिए", "सीख", "सबक"],
    "definition": ["means", "is called", "because", "for example", "for instance", "let me explain", "how it works",
                   "which means", "in other words", "is when", "is a ", "refers to",
                   "matlab", "kyunki", "jaise ki", "udaharan", "yaani", "मतलब", "क्योंकि", "जैसे कि", "यानी", "उदाहरण"],
    "conclusion": ["in conclusion", "to sum up", "at the end of the day", "so that's", "thank you", "hope this helps",
                   "see you", "that's all", "toh yeh tha", "umeed hai", "dhanyavaad", "शुक्रिया", "धन्यवाद"],
    "fact": ["according to", "study", "research", "percent", "million", "billion", "thousand", "founded", "scientists",
             "data shows", "survey", "record", "crore", "lakh", "प्रतिशत", "करोड़", "लाख", "अध्ययन"],
    "question_words": ["what", "why", "how", "who", "when", "where", "which", "is it", "are you", "do you", "did you",
                       "can you", "kya", "kyun", "kaise", "kaun", "kab", "kahan", "क्या", "क्यों", "कैसे", "कौन", "कब"],
}
DEPENDENT_STARTS = {"and", "but", "so", "because", "which", "that", "this", "these", "those", "it", "they", "he", "she",
                    "then", "also", "however", "therefore", "or", "plus", "anyway", "aur", "lekin", "toh", "isliye",
                    "woh", "wo", "yeh", "ye", "uska", "unka", "par", "magar", "और", "लेकिन", "तो", "इसलिए", "वो", "यह", "ये"}
BACKREF = ["as i said", "as i mentioned", "like i said", "earlier", "previously", "last time", "as we saw", "we talked about",
           "jaisa maine kaha", "pehle bataya", "जैसा मैंने कहा", "पहले बताया"]
NUM_RE = re.compile(r"(?:\d[\d,\.]*\s?(?:%|percent|x|k|m|million|billion|crore|lakh)?)|[०-९]+")


def _has(text_l: str, cues: list[str]) -> int:
    return sum(1 for c in cues if c in text_l)


@dataclass
class SentenceTags:
    idx: int
    question: bool = False
    number: bool = False
    surprise: int = 0
    emotion: int = 0
    story: int = 0
    resolution: int = 0
    opinion: int = 0
    takeaway: int = 0
    definition: int = 0
    conclusion: int = 0
    fact: int = 0
    dependent_start: bool = False
    backref: bool = False
    exclaim: bool = False
    words: int = 0
    fillers: int = 0

    def labels(self) -> list[str]:
        out = []
        for name in ("question", "number", "surprise", "emotion", "story", "opinion", "takeaway", "definition",
                     "conclusion", "fact"):
            if getattr(self, name):
                out.append(name)
        return out


def tag_sentence(s: Sentence) -> SentenceTags:
    t = s.text.strip()
    tl = " " + t.lower() + " "
    toks = tokenize(t)
    first = toks[0] if toks else ""
    q = t.endswith("?") or (bool(toks) and any(tl.lstrip().startswith(w + " ") for w in CUES["question_words"])
                            and len(toks) <= 25 and t.endswith("?"))
    return SentenceTags(
        idx=s.idx, question=q, number=bool(NUM_RE.search(t)),
        surprise=_has(tl, CUES["surprise"]), emotion=_has(tl, CUES["emotion"]), story=_has(tl, CUES["story"]),
        resolution=_has(tl, CUES["resolution"]), opinion=_has(tl, CUES["opinion"]),
        takeaway=_has(tl, CUES["takeaway"]), definition=_has(tl, CUES["definition"]),
        conclusion=_has(tl, CUES["conclusion"]), fact=_has(tl, CUES["fact"]) + (1 if NUM_RE.search(t) else 0),
        dependent_start=first in DEPENDENT_STARTS, backref=any(b in tl for b in BACKREF),
        exclaim=t.endswith("!"), words=len(toks), fillers=sum(1 for x in toks if x in {"um", "uh", "umm", "uhh", "hmm"}))


@dataclass
class Topic:
    start_idx: int
    end_idx: int
    start: float
    end: float
    keywords: list[str]


@dataclass
class ContentMap:
    language: str | None
    duration: float
    main_topic_keywords: list[str]
    topics: list[Topic]
    tags: list[SentenceTags]
    qa_pairs: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = [{**asdict(t), "labels": t.labels()} for t in self.tags]
        return d


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    num = sum(a[k] * b[k] for k in a.keys() & b.keys())
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def segment_topics(sentences: list[Sentence], block: int = 6, min_len: int = 8) -> list[Topic]:
    """TextTiling-style: boundaries where adjacent sentence blocks share the least vocabulary."""
    n = len(sentences)
    if n == 0:
        return []
    toks = [Counter(content_tokens(s.text)) for s in sentences]
    cuts: list[int] = []
    if n >= 2 * min_len:
        sims = []
        for i in range(block, n - block + 1):
            a, b = Counter(), Counter()
            for k in range(i - block, i):
                a.update(toks[k])
            for k in range(i, min(n, i + block)):
                b.update(toks[k])
            sims.append((i, _cos(a, b)))
        if sims:
            mean = sum(v for _, v in sims) / len(sims)
            sd = math.sqrt(sum((v - mean) ** 2 for _, v in sims) / len(sims))
            thr = mean - 0.5 * sd
            for k, (i, v) in enumerate(sims):
                left = sims[k - 1][1] if k > 0 else 9
                right = sims[k + 1][1] if k + 1 < len(sims) else 9
                if v < thr and v <= left and v <= right and (not cuts or i - cuts[-1] >= min_len) and i >= min_len and n - i >= min_len:
                    cuts.append(i)
    bounds = [0] + cuts + [n]
    topics = []
    for a, b in zip(bounds, bounds[1:]):
        c: Counter = Counter()
        for k in range(a, b):
            c.update(set(content_tokens(sentences[k].text)))
        kws = [w for w, _ in c.most_common(6)]
        topics.append(Topic(a, b - 1, sentences[a].start, sentences[b - 1].end, kws))
    return topics


def find_qa_pairs(sentences: list[Sentence], tags: list[SentenceTags], max_gap: float = 3.0) -> list[dict]:
    pairs = []
    for i, t in enumerate(tags[:-1]):
        if t.question and sentences[i].gap_after <= max_gap and not tags[i + 1].question:
            end = i + 1
            while end + 1 < len(sentences) and end - i < 4 and sentences[end].gap_after < 1.0 and not tags[end + 1].question:
                end += 1
            pairs.append({"question_idx": i, "answer_end_idx": end, "start": sentences[i].start, "end": sentences[end].end})
    return pairs


def understand(sentences: list[Sentence], language: str | None, duration: float) -> ContentMap:
    tags = [tag_sentence(s) for s in sentences]
    kws = [w for w, _ in global_keywords(sentences, 8)]
    counts = Counter()
    for t in tags:
        for lab in t.labels():
            counts[lab] += 1
    return ContentMap(language=language, duration=duration, main_topic_keywords=kws,
                      topics=segment_topics(sentences), tags=tags, qa_pairs=find_qa_pairs(sentences, tags),
                      stats={"sentences": len(sentences), "label_counts": dict(counts),
                             "cue_coverage": "en/hi/hinglish lexicons + language-independent signals"})

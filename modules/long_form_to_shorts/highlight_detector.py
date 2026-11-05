"""Candidate generation and scoring. Every candidate is a run of whole sentences, so it never starts or ends
mid-sentence by construction. Scores are heuristic (0..1) and fully explained in `scores` / `reasons`."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import accumulate

from .content_understanding import ContentMap, SentenceTags
from .text_utils import Sentence, content_tokens, global_keywords

CRITERIA = ("hook", "usefulness", "surprise", "emotion", "story", "education", "standalone", "relevance", "ending")
BASE_WEIGHTS = {"hook": .18, "usefulness": .12, "surprise": .10, "emotion": .08, "story": .10,
                "education": .10, "standalone": .16, "relevance": .06, "ending": .10}
STYLE_MULT = {
    "educational": {"education": 1.6, "usefulness": 1.4, "emotion": .5},
    "tutorial": {"education": 1.6, "usefulness": 1.6, "emotion": .4, "story": .6},
    "podcast": {"story": 1.4, "emotion": 1.3, "surprise": 1.2},
    "storytelling": {"story": 1.8, "emotion": 1.4, "education": .6},
    "motivational": {"emotion": 1.6, "story": 1.3, "usefulness": 1.2},
    "comedy": {"surprise": 1.6, "emotion": 1.3, "education": .4},
    "entertainment": {"surprise": 1.4, "emotion": 1.3, "story": 1.2},
    "debate": {"hook": 1.3, "surprise": 1.2, "usefulness": .8},
    "news": {"usefulness": 1.5, "education": 1.2, "emotion": .6},
    "science": {"education": 1.5, "surprise": 1.3},
    "technology": {"education": 1.4, "usefulness": 1.3},
    "documentary": {"story": 1.4, "education": 1.2},
    "facts": {"surprise": 1.5, "education": 1.3},
}


def weights_for(style: str | None) -> dict:
    w = dict(BASE_WEIGHTS)
    for k, m in STYLE_MULT.get((style or "").lower().split("_")[0], {}).items():
        w[k] *= m
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


@dataclass
class Candidate:
    i: int                # first sentence idx
    j: int                # last sentence idx (inclusive)
    start: float
    end: float
    score: float
    scores: dict
    text: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"i": self.i, "j": self.j, "start": round(self.start, 3), "end": round(self.end, 3),
                "score": round(self.score, 4), "scores": {k: round(v, 3) for k, v in self.scores.items()},
                "text": self.text, "reasons": self.reasons, "warnings": self.warnings}


def _prefix(vals):
    return [0] + list(accumulate(vals))


def generate_candidates(sentences: list[Sentence], cmap: ContentMap, min_dur: float = 15.0, max_dur: float = 60.0,
                        style: str | None = None, min_words_per_s: float = 1.0) -> list[Candidate]:
    n = len(sentences)
    if n == 0:
        return []
    tags: list[SentenceTags] = cmap.tags
    w = weights_for(style)
    kw = {k: v for k, v in global_keywords(sentences, 20)}
    kmax = max(kw.values()) if kw else 1.0
    rel = [min(1.0, sum(kw.get(t, 0) for t in set(content_tokens(s.text))) / (kmax * 2.0)) for s in sentences]

    P = {name: _prefix([getattr(t, name) if not isinstance(getattr(t, name), bool) else int(getattr(t, name))
                        for t in tags])
         for name in ("question", "number", "surprise", "emotion", "story", "resolution", "opinion",
                      "takeaway", "definition", "conclusion", "fact", "exclaim", "words", "fillers")}
    Prel = _prefix(rel)
    cands: list[Candidate] = []

    def cnt(name, i, j):
        return P[name][j + 1] - P[name][i]

    for i in range(n):
        if i > 0 and not sentences[i - 1].ends_clean and sentences[i].gap_before < 0.3:
            continue  # previous sentence flows straight into this one: don't start mid-thought
        for j in range(i, n):
            dur = sentences[j].end - sentences[i].start
            if dur > max_dur:
                break
            if dur < min_dur or not sentences[j].ends_clean:
                continue
            m = j - i + 1
            words = cnt("words", i, j)
            if words / max(dur, 1e-6) < min_words_per_s:
                continue  # mostly silence / music
            first, last = tags[i], tags[j]
            f0 = sentences[i]
            # --- hook (first sentence, plus second as fallback)
            hk = 0.0
            hk += 0.30 if first.question else 0.0
            hk += 0.25 if (first.number or first.surprise) else 0.0
            hk += 0.20 if first.opinion or first.story else 0.0
            hk += 0.15 if 3 <= first.words <= 20 else 0.0
            hk += 0.10 if not first.dependent_start else -0.15
            if m > 1 and not (first.question or first.surprise):
                s2 = tags[i + 1]
                hk += 0.10 if (s2.surprise or s2.question) else 0.0
            hook = max(0.0, min(1.0, hk))
            use = min(1.0, 2.0 * (cnt("takeaway", i, j) + cnt("definition", i, j) + min(cnt("fact", i, j), m)) / m)
            sur = min(1.0, 2.5 * cnt("surprise", i, j) / m + (0.15 if cnt("number", i, j) and cnt("surprise", i, j) else 0))
            emo = min(1.0, 3.0 * cnt("emotion", i, j) / m + 0.4 * cnt("exclaim", i, j) / m)
            st_hits = cnt("story", i, j)
            story = 0.0 if st_hits == 0 else min(1.0, 0.45 + 0.1 * st_hits + (0.3 if cnt("resolution", i, j) else 0.0))
            edu = min(1.0, 1.8 * (cnt("definition", i, j) + cnt("takeaway", i, j) + cnt("fact", i, j)) / m
                      + 0.25 * (Prel[j + 1] - Prel[i]) / m)
            # --- standalone clarity
            sa = 1.0
            if first.dependent_start:
                sa -= 0.35
            if first.backref or any(tags[k].backref for k in range(i, min(j + 1, i + 2))):
                sa -= 0.30
            q_unanswered = (last.question and not tags[j].conclusion)
            if q_unanswered:
                sa -= 0.35
            elif cnt("question", i, j) and not any(tags[k].question and k < j for k in range(i, j + 1)):
                sa -= 0.0
            if f0.gap_before < 0.15 and i > 0 and not sentences[i - 1].ends_clean:
                sa -= 0.2
            standalone = max(0.0, min(1.0, sa))
            relevance = (Prel[j + 1] - Prel[i]) / m
            # --- ending
            en = 0.30 if sentences[j].text.strip()[-1:] in ".!?।…\"'”" else 0.1
            en += 0.35 if (last.takeaway or last.conclusion or last.resolution) else 0.0
            en += 0.15 if (last.opinion or last.surprise) else 0.0
            en += 0.20 if sentences[j].gap_after >= 0.5 else 0.0
            if last.dependent_start and last.words < 5:
                en -= 0.2
            if last.question:
                en -= 0.3
            ending = max(0.0, min(1.0, en))
            scores = {"hook": hook, "usefulness": use, "surprise": sur, "emotion": emo, "story": story,
                      "education": edu, "standalone": standalone, "relevance": min(1.0, relevance), "ending": ending}
            total = sum(w[k] * scores[k] for k in CRITERIA)
            fill = cnt("fillers", i, j) / max(1, words)
            total *= (1.0 - min(0.3, fill * 4))
            cand = Candidate(i, j, sentences[i].start, sentences[j].end, total, scores,
                             " ".join(s.text for s in sentences[i:j + 1]))
            cand.reasons = [f"{k}={scores[k]:.2f}" for k in sorted(scores, key=lambda x: -w[x] * scores[x])[:3]
                            if scores[k] >= 0.4]
            if first.dependent_start:
                cand.warnings.append("opens with a connective (may need earlier context)")
            if first.backref:
                cand.warnings.append("refers back to earlier material")
            if q_unanswered:
                cand.warnings.append("ends on an unanswered question")
            if any(s.timing == "interpolated" for s in sentences[i:j + 1]):
                cand.warnings.append("sentence timing is interpolated (segment-level transcript); boundaries approximate")
            cands.append(cand)
    cands.sort(key=lambda c: -c.score)
    return cands

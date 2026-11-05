"""Per-Short metadata. Everything is extracted from the clip's own transcript text, so it cannot claim
more than the clip says. An optional LLM may reword, but its output is rejected if it adds facts."""
from __future__ import annotations

import json
import re
from collections import Counter

from providers.llm import LLMProvider, LLMUnavailable

from .content_understanding import DEPENDENT_STARTS, tag_sentence
from .text_utils import Sentence, content_tokens, fmt_hms, jaccard, tokenize

CLICKBAIT = ["you won't believe", "you wont believe", "shocking", "gone wrong", "100% guaranteed", "must watch",
             "doctors hate", "this will blow your mind", "insane", "(gone", "number 1 secret"]
LEAD_STRIP = re.compile(r"^(?:(?:and|but|so|well|okay|ok|now|then|also|like|yeah|right|actually|aur|toh|to|lekin|par)\b[\s,]*)+", re.I)


def _clean_title(text: str, max_len: int = 62) -> str:
    t = LEAD_STRIP.sub("", text.strip())
    t = re.sub(r"\s+", " ", t).strip(" ,;:-–—\"'")
    t = re.sub(r"[.!…।]+$", "", t)
    if len(t) > max_len:
        cut = t[:max_len].rsplit(" ", 1)[0]
        words = cut.split()
        while words and words[-1].lower() in {"a", "an", "the", "and", "or", "but", "of", "to", "in", "is", "that", "with", "for"}:
            words.pop()
        t = " ".join(words).rstrip(",;:-")
    if t and t[0].isalpha():
        t = t[0].upper() + t[1:]
    if t.isupper() and len(t) > 6:
        t = t.capitalize()
    return t


def _title_candidates(sents: list[Sentence], keywords: list[str]) -> list[tuple[float, str]]:
    out = []
    for k, s in enumerate(sents[:6]):
        tg = tag_sentence(s)
        text = _clean_title(s.text)
        if len(text.split()) < 3:
            continue
        score = 0.0
        score += 1.0 if tg.question else 0
        score += 0.8 if (tg.surprise or tg.number) else 0
        score += 0.6 if (tg.takeaway or tg.opinion) else 0
        score += 0.4 if len(text) <= 62 else 0
        score -= 0.5 if tg.dependent_start else 0
        score -= 0.1 * k
        score += 0.15 * sum(1 for w in set(content_tokens(text)) if w in keywords)
        out.append((score, text))
    return sorted(out, key=lambda x: -x[0])


def _hook(sents: list[Sentence]) -> str | None:
    for _, text in _title_candidates(sents, [])[:3]:
        words = text.split()
        if 3 <= len(words) <= 9 and text.lower().split()[0] not in DEPENDENT_STARTS:
            return text
    if sents:
        words = _clean_title(sents[0].text, 48).split()
        if len(words) >= 3:
            return " ".join(words[:8])
    return None


def hashtags(language: str, keywords: list[str], n: int = 4) -> list[str]:
    tags = ["#Shorts"]
    for k in keywords:
        h = re.sub(r"[^\w]", "", k, flags=re.UNICODE)
        if 2 < len(h) <= 24 and not h.isdigit():
            tag = "#" + (h.capitalize() if language in ("en", "hinglish") else h)
            if tag not in tags:
                tags.append(tag)
        if len(tags) >= n + 1:
            break
    return tags


def generate_clip_metadata(sents: list[Sentence], clip_start: float, clip_end: float, source_name: str,
                           language: str, global_keywords: list[str], used_titles: list[str] | None = None,
                           llm: LLMProvider | None = None, source_title: str | None = None) -> dict:
    used_titles = used_titles or []
    text = " ".join(s.text for s in sents)
    c = Counter(t for s in sents for t in content_tokens(s.text))
    ranked = sorted(c, key=lambda w: (-(c[w] + (1.5 if w in global_keywords else 0)), w))
    keywords = ranked[:8]
    cands = _title_candidates(sents, global_keywords)
    title = None
    for _, cand in cands:
        if all(jaccard(set(tokenize(cand)), set(tokenize(u))) < 0.7 for u in used_titles):
            title = cand
            break
    if not title:
        base = cands[0][1] if cands else (" ".join(keywords[:3]).capitalize() or "Short")
        title = f"{base} ({fmt_hms(clip_start)})"[:70]
    desc_lines = []
    summary = " ".join(s.text.strip() for s in sents[:2])
    if len(summary) > 220:
        summary = summary[:220].rsplit(" ", 1)[0] + "…"
    desc_lines.append(summary)
    ref = f"From: {source_title or source_name} [{fmt_hms(clip_start)}–{fmt_hms(clip_end)}]"
    desc_lines.append(ref)
    tags = hashtags(language, keywords)
    meta = {"title": title, "description": "\n\n".join(desc_lines) + "\n\n" + " ".join(tags),
            "hashtags": tags, "keywords": keywords, "hook_text": _hook(sents),
            "source": {"file": source_name, "start": round(clip_start, 2), "end": round(clip_end, 2),
                       "start_hms": fmt_hms(clip_start), "end_hms": fmt_hms(clip_end)},
            "method": "extractive", "language": language}
    if llm is not None:
        meta = _llm_refine(meta, text, llm)
    return meta


def _llm_refine(meta: dict, clip_text: str, llm: LLMProvider) -> dict:
    prompt = ("You write YouTube Shorts metadata. Use ONLY facts stated in the transcript below. No clickbait, "
              "no claims not in the transcript. Reply with JSON: {\"title\":\"<=60 chars\",\"hook_text\":\"<=8 words\"}.\n\n"
              f"TRANSCRIPT:\n{clip_text[:3000]}")
    try:
        data = json.loads(re.search(r"\{.*\}", llm.complete(prompt, max_tokens=200), re.S).group(0))
    except (LLMUnavailable, AttributeError, ValueError):
        return meta
    ok_vocab = set(tokenize(clip_text))
    for key, limit in (("title", 62), ("hook_text", 60)):
        val = (data.get(key) or "").strip()
        toks = [t for t in tokenize(val) if len(t) > 3]
        if val and len(val) <= limit and toks and sum(t in ok_vocab for t in toks) / len(toks) >= 0.6 \
                and not any(b in val.lower() for b in CLICKBAIT):
            meta[key] = val
            meta["method"] = "llm_validated"
    return meta


def validate_metadata(meta: dict, clip_text: str, other_titles: list[str] | None = None) -> list[dict]:
    out = []

    def add(check, ok, detail="", level="fail"):
        out.append({"check": check, "ok": bool(ok), "detail": detail, "level": level})

    t = meta.get("title", "")
    add("title_present", bool(t.strip()))
    add("title_length", 0 < len(t) <= 100, f"{len(t)} chars")
    add("title_not_all_caps", not (t.isupper() and len(t) > 6))
    add("no_clickbait_phrases", not any(b in t.lower() or b in meta.get("description", "").lower() for b in CLICKBAIT),
        level="warn")
    toks = [x for x in tokenize(t) if len(x) > 3]
    vocab = set(tokenize(clip_text))
    grounded = (sum(x in vocab for x in toks) / len(toks)) if toks else 1.0
    add("title_grounded_in_clip", grounded >= 0.6 or "(" in t, f"{grounded:.0%} of title words appear in the clip")
    add("description_present", len(meta.get("description", "").strip()) > 20)
    tags = meta.get("hashtags", [])
    add("hashtags_valid", 0 < len(tags) <= 15 and all(h.startswith("#") and " " not in h for h in tags))
    add("source_reference", bool(meta.get("source", {}).get("file")) and "start" in meta.get("source", {}))
    if other_titles is not None:
        add("title_unique", all(jaccard(set(tokenize(t)), set(tokenize(o))) < 0.8 for o in other_titles))
    return out

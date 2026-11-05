"""YouTube metadata for Mode A, built from the script and sources only (no invented claims)."""
from __future__ import annotations

import re
from collections import Counter

from modules.long_form_to_shorts.text_utils import content_tokens

CLICKBAIT = ["you won't believe", "shocking", "gone wrong", "will blow your mind", "doctors hate", "insane", "100% guaranteed"]
GENERIC = {"through", "found", "study", "people", "other", "their", "which", "while", "using", "where", "being",
           "could", "would", "should", "about", "often", "likely", "makes", "within", "between", "video"}
CATEGORIES = {"science": "Science & Technology (28)", "technology": "Science & Technology (28)", "history": "Education (27)",
              "finance": "Education (27)", "health": "Howto & Style (26)", "general": "Education (27)"}


def fmt_ts(t: float) -> str:
    t = int(t)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chapters(spans: list[tuple[str, float, float]], script: dict) -> list[tuple[str, str]]:
    """YouTube chapters: first at 0:00, >= 3 chapters, each >= 10 s."""
    titles = {s["id"]: s["title"] for s in script["sections"]}
    ch = [(fmt_ts(a), titles[sid]) for sid, a, b in spans if b - a >= 10]
    if len(ch) < 3:
        return []
    ch[0] = ("0:00", ch[0][1])
    return ch


def generate_metadata(topic: str, script: dict, sources: list[dict], category: str, language: str,
                      spans: list[tuple[str, float, float]], disclosure: str | None = None) -> dict:
    # keywords come from researched claim sentences only (template/structure text would add noise)
    text = " ".join(s["text"] for sec in script["sections"] for s in sec["sentences"] if s["kind"] == "claim") or \
        " ".join(s["text"] for sec in script["sections"] for s in sec["sentences"])
    c = Counter(t for t in content_tokens(text) if len(t) >= 5 and t not in GENERIC)
    ranked = [w for w, n in c.most_common(30) if n >= 2] or [w for w, _ in c.most_common(8)]
    topic_words = [w for w in content_tokens(topic) if len(w) >= 4]
    kws = list(dict.fromkeys([topic.lower()] + topic_words + [w for w in ranked[:14] if w not in topic.lower()]))
    title = topic.strip().rstrip("?.!")
    title = title[0].upper() + title[1:] if title else title
    if len(title) < 25 and language in ("en", "hinglish"):
        title = f"{title}: What You Need to Know" if language == "en" else f"{title}: Jaanne Layak Baatein"
    title = title[:100]
    hashtags = ["#" + re.sub(r"\W", "", topic.title().replace(" ", ""))[:24]] + \
               ["#" + re.sub(r"\W", "", w.capitalize()) for w in kws[1:3] if c.get(w, 0) >= 2]
    hashtags = [h for h in dict.fromkeys(hashtags) if len(h) > 2][:3]
    chs = chapters(spans, script)
    lead = next((s["text"] for sec in script["sections"] for s in sec["sentences"] if s["kind"] == "claim"), topic)
    lines = [lead, "", f"In this video: {topic}."]
    if chs:
        lines += ["", "Chapters:"] + [f"{t} {n}" for t, n in chs]
    lines += ["", "Sources:"] + [f"- {s['title'] or s['url']} ({s['publisher'] or 'source'}) {s['url']}" +
                                 (f" [licence: {s['license']}]" if s.get("license") and s["license"] != "unknown" else "")
                                 for s in sources]
    if disclosure:
        lines += ["", disclosure]
    lines += ["", " ".join(hashtags)]
    desc = "\n".join(lines)
    tags, total = [], 0
    for k in kws:
        if total + len(k) + 1 > 450:
            break
        tags.append(k)
        total += len(k) + 1
    return {"title": title, "description": desc, "keywords": kws[:15], "tags": tags, "hashtags": hashtags,
            "category_suggestion": CATEGORIES.get(category, CATEGORIES["general"]), "language": language,
            "pinned_comment": f"Which part of {topic} surprised you most? Sources are listed in the description.",
            "chapters": chs, "disclosure": disclosure}


def validate_metadata(m: dict) -> list[dict]:
    out = []

    def add(check, ok, detail="", level="fail"):
        out.append({"check": check, "ok": bool(ok), "detail": str(detail), "level": level})

    add("title_length_1_100", 1 <= len(m["title"]) <= 100, len(m["title"]))
    add("description_under_5000", len(m["description"]) <= 5000, len(m["description"]))
    add("tags_under_500_chars", sum(len(t) + 1 for t in m["tags"]) <= 500)
    add("hashtags_max_15", len(m["hashtags"]) <= 15)
    add("no_clickbait", not any(b in (m["title"] + m["description"]).lower() for b in CLICKBAIT), level="warn")
    words = re.findall(r"\w+", " ".join(m["keywords"]).lower())
    top = Counter(words).most_common(1)
    add("no_keyword_stuffing", not top or top[0][1] <= 4, top, "warn")
    add("sources_listed", "Sources:" in m["description"])
    for i, (ts, _) in enumerate(m["chapters"]):
        if i == 0:
            add("chapters_start_0_00", ts == "0:00")
    return out

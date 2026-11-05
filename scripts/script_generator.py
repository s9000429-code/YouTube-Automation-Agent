"""Script writing. Every factual sentence is a research fact (verbatim) and carries its fact id, so claims stay
traceable. Structural sentences (hook, transitions, call to action) are template text and carry no claim.

Honest scope: without an LLM the script is an organised read-through of the researched facts, not creative
prose. With an LLM configured, section text is reworded but REJECTED if it introduces numbers/names absent
from the cited facts. Structural templates exist for en / hi / hinglish; kn/te need a supplied script."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from providers.llm import LLMProvider, LLMUnavailable

TEMPLATES = {
    "en": {
        "hook_q": "Did you know this about {topic}?", "hook_fact": "Here is something worth knowing about {topic}.",
        "intro": "In this video we look at {topic}, and the key things worth understanding.",
        "transition": ["Let's move on.", "Next, a closer look.", "Now consider this."],
        "outro": "That is the essential picture of {topic}.",
        "cta": "If you found this useful, subscribe for more.",
        "sec_title": "Part {n}"},
    "hinglish": {
        "hook_q": "Kya aap {topic} ke baare mein yeh jaante hain?", "hook_fact": "{topic} ke baare mein yeh baat jaanna zaroori hai.",
        "intro": "Is video mein hum {topic} ko samjhenge, aur zaroori baatein dekhenge.",
        "transition": ["Chalo aage badhte hain.", "Ab ek aur baat dekhte hain.", "Ab yeh dekhiye."],
        "outro": "Yeh tha {topic} ka mukhya saar.",
        "cta": "Agar video pasand aaya ho toh subscribe zaroor karein.", "sec_title": "Bhaag {n}"},
    "hi": {
        "hook_q": "क्या आप {topic} के बारे में यह जानते हैं?", "hook_fact": "{topic} के बारे में यह बात जानना ज़रूरी है।",
        "intro": "इस वीडियो में हम {topic} को समझेंगे और ज़रूरी बातें देखेंगे।",
        "transition": ["चलिए आगे बढ़ते हैं।", "अब एक और बात देखते हैं।", "अब यह देखिए।"],
        "outro": "यह था {topic} का मुख्य सार।",
        "cta": "अगर वीडियो पसंद आया हो तो सब्सक्राइब ज़रूर करें।", "sec_title": "भाग {n}"},
}
STYLE_NOTES = {
    "educational": "clear, structured, neutral", "motivational": "energetic, second person",
    "documentary": "measured, narrative", "storytelling": "narrative arc", "facts": "short punchy facts",
    "technology": "precise, practical", "science": "precise, curious", "news": "neutral, attributed",
    "tutorial": "step by step", "entertainment": "light", "comedy": "light", "podcast": "conversational",
    "debate": "balanced", "motivational ": "energetic"}


@dataclass
class Sentence:
    text: str
    fact_ids: list[int] = field(default_factory=list)
    kind: str = "claim"                 # claim | structure
    hedged: bool = False


@dataclass
class Section:
    id: str
    title: str
    kind: str                           # hook | intro | body | conclusion
    sentences: list[Sentence]
    visual_hint: str = ""


def understand_topic(topic: str, style: str, language: str) -> dict:
    t = topic.lower()
    category = next((c for c, kws in {
        "science": ["science", "physics", "space", "biology", "climate", "quantum", "chemistry"],
        "technology": ["ai", "software", "computer", "phone", "internet", "tech", "robot", "chip", "programming"],
        "history": ["history", "war", "empire", "ancient", "revolution", "century"],
        "finance": ["money", "invest", "stock", "economy", "bank", "tax"],
        "health": ["health", "sleep", "diet", "exercise", "medicine", "habit"]}.items() if any(k in t.split() or k in t for k in kws)),
        "general")
    structure = {"tutorial": ["hook", "intro", "steps", "recap"], "storytelling": ["hook", "setup", "turn", "resolution"],
                 "news": ["hook", "what happened", "context", "outlook"]}.get(style, ["hook", "intro", "main points", "conclusion", "call to action"])
    return {"topic": topic, "category": category, "style": style, "tone": STYLE_NOTES.get(style, "clear"),
            "audience": "general viewers new to the topic", "language": language, "suggested_structure": structure}


def _tpl(language: str) -> dict | None:
    return TEMPLATES.get(language)


def target_words(duration_s: int, language: str, wpm_table: dict) -> int:
    return int(duration_s / 60 * wpm_table.get(language, 140))


def generate_script(topic: str, facts: list[dict], language: str, style: str, duration_s: int, wpm_table: dict,
                    llm: LLMProvider | None = None) -> dict:
    warnings: list[str] = []
    tpl = _tpl(language)
    if tpl is None:
        warnings.append(f"no structural templates for '{language}': script contains researched facts only "
                        f"(no hook/intro/outro wording). Supply --script-file for full {language} narration.")
        tpl = {"hook_q": "", "hook_fact": "", "intro": "", "transition": [], "outro": "", "cta": "", "sec_title": "{n}"}
    firm = [f for f in facts if f["confidence"] == "supported"]
    soft = [f for f in facts if f["confidence"] != "supported"]
    budget = target_words(duration_s, language, wpm_table)
    used, picked = 0, []
    fixed = sum(len(tpl[k].split()) for k in ("hook_q", "intro", "outro", "cta")) + 6
    for f in firm:                                    # firm facts first; hedged ones only if short on material
        n = len(f["text"].split())
        if used + n + fixed > budget * 1.15 and picked:
            continue
        picked.append((f, False))
        used += n
    if used + fixed < budget * 0.6:
        for f in soft:
            n = len(f["text"].split())
            if used + n + fixed > budget * 1.15:
                continue
            picked.append((f, True))
            used += n
        if any(h for _, h in picked):
            warnings.append("not enough firm facts for the requested duration: hedged/single-source facts were "
                            "included and are attributed with 'reportedly'-style framing flagged in the script")
    if used + fixed < budget * 0.5:
        warnings.append(f"researched material covers ~{used + fixed} of ~{budget} target words; video will be shorter "
                        "than requested. Add more sources instead of padding.")
    sections: list[Section] = []
    if not picked:
        return {"sections": [], "warnings": warnings + ["no usable facts: research more before scripting"],
                "words": 0, "target_words": budget}
    lead = picked[0][0]
    q = tpl["hook_q"].format(topic=topic) if tpl["hook_q"] else ""
    hook_sents = [Sentence(q, kind="structure")] if q else []
    hook_sents.append(Sentence(lead["text"], [lead["id"]], hedged=picked[0][1]))
    sections.append(Section("s0", "Hook", "hook", hook_sents, topic))
    if tpl["intro"]:
        sections.append(Section("s1", "Introduction", "intro", [Sentence(tpl["intro"].format(topic=topic), kind="structure")], topic))
    rest = picked[1:]
    per = max(2, min(4, round(len(rest) / 4) or 2))
    for n, i in enumerate(range(0, len(rest), per), 1):
        grp = rest[i:i + per]
        sents = []
        if n > 1 and tpl["transition"]:
            sents.append(Sentence(tpl["transition"][(n - 2) % len(tpl["transition"])], kind="structure"))
        sents += [Sentence(f["text"], [f["id"]], hedged=h) for f, h in grp]
        hint = " ".join(w for w in re.findall(r"[A-Za-zऀ-ॿ]{5,}", grp[0][0]["text"])[:4])
        sections.append(Section(f"s{len(sections)}", tpl["sec_title"].format(n=n), "body", sents, hint or topic))
    closing = [Sentence(tpl["outro"].format(topic=topic), kind="structure")] if tpl["outro"] else []
    if tpl["cta"]:
        closing.append(Sentence(tpl["cta"], kind="structure"))
    if closing:
        sections.append(Section(f"s{len(sections)}", "Conclusion", "conclusion", closing, topic))
    script = {"topic": topic, "language": language, "style": style, "sections": [_sec(s) for s in sections],
              "warnings": warnings, "method": "extractive_template", "target_words": budget}
    if llm is not None:
        script = _llm_polish(script, facts, llm)
    script["words"] = sum(len(s["text"].split()) for sec in script["sections"] for s in sec["sentences"])
    return script


def _sec(s: Section) -> dict:
    return {"id": s.id, "title": s.title, "kind": s.kind, "visual_hint": s.visual_hint,
            "sentences": [{"text": x.text, "fact_ids": x.fact_ids, "kind": x.kind, "hedged": x.hedged} for x in s.sentences]}


def _nums(t: str) -> set[str]:
    return set(re.findall(r"\d[\d,\.]*", t))


def _caps(t: str) -> set[str]:
    return {w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", t)}


def _llm_polish(script: dict, facts: list[dict], llm: LLMProvider) -> dict:
    """Reword claim sentences for flow; accept a rewrite only if it adds no new numbers or proper names."""
    by_id = {f["id"]: f["text"] for f in facts}
    changed = 0
    for sec in script["sections"]:
        for s in sec["sentences"]:
            if s["kind"] != "claim" or not s["fact_ids"]:
                continue
            src = " ".join(by_id[i] for i in s["fact_ids"])
            try:
                out = llm.complete("Rewrite this fact as one clear spoken sentence for a narrator. Add NOTHING new: no "
                                   f"numbers, names, or claims not present.\nFACT: {src}\nSENTENCE:", max_tokens=120).strip().split("\n")[0]
            except LLMUnavailable:
                script["warnings"].append("LLM unavailable; kept extractive wording")
                return script
            if out and _nums(out) <= _nums(src) and _caps(out) <= _caps(src) | {"The", "This", "It", "That", "They"} and len(out.split()) <= 1.6 * len(src.split()):
                s["text"] = out
                changed += 1
    script["method"] = f"llm_validated({changed} sentences reworded)"
    return script


def validate_script(script: dict, facts: list[dict], language: str) -> list[dict]:
    out = []

    def add(check, ok, detail="", level="fail"):
        out.append({"check": check, "ok": bool(ok), "detail": str(detail), "level": level})

    secs = script.get("sections", [])
    kinds = [s["kind"] for s in secs]
    ids = {f["id"] for f in facts}
    add("has_sections", bool(secs))
    add("opening_hook", bool(kinds) and kinds[0] == "hook" and bool(secs[0]["sentences"]))
    add("has_body", "body" in kinds)
    if language in TEMPLATES:
        add("has_intro", "intro" in kinds)
        add("has_conclusion", kinds[-1:] == ["conclusion"])
    claims = [s for sec in secs for s in sec["sentences"] if s["kind"] == "claim"]
    add("claims_traceable", all(s["fact_ids"] and set(s["fact_ids"]) <= ids for s in claims),
        f"{sum(1 for s in claims if not s['fact_ids'])} untraced claim sentences")
    add("no_empty_sentences", all(s["text"].strip() for sec in secs for s in sec["sentences"]))
    tw, w = script.get("target_words", 0), script.get("words", 0)
    add("length_near_target", tw == 0 or 0.5 * tw <= w <= 1.3 * tw, f"{w} words vs target {tw}", "warn")
    add("hedged_claims_marked", all(s["hedged"] or not any(f["confidence"] != "supported" for f in facts if f["id"] in s["fact_ids"]) for s in claims),
        "", "warn")
    return out

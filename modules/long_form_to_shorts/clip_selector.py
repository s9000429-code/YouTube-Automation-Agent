"""Selection of non-overlapping, non-duplicate clips; boundary padding/snapping and validation."""
from __future__ import annotations

from dataclasses import dataclass, field

from .highlight_detector import Candidate
from .text_utils import Sentence, content_tokens, jaccard


@dataclass
class Clip:
    start: float
    end: float
    score: float
    text: str
    i: int | None = None
    j: int | None = None
    hook_text: str | None = None
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1e-6, min(a1 - a0, b1 - b0))


def is_duplicate(a_text: str, b_text: str, threshold: float = 0.5) -> bool:
    return jaccard(set(content_tokens(a_text)), set(content_tokens(b_text))) >= threshold


def auto_count(duration_s: float, n_candidates_ok: int) -> int:
    """Roughly one Short per ~6 minutes of source, clamped 3..10 (1..3 for very short sources)."""
    if duration_s < 180:
        want = 1 if duration_s < 90 else 2 if duration_s < 150 else 3
    else:
        want = max(3, min(10, round(duration_s / 360)))
    return min(want, n_candidates_ok)


def select_clips(cands: list[Candidate], count: int | str, duration_s: float, target_dur: float = 45.0,
                 min_score: float = 0.30, max_overlap: float = 0.15, dup_threshold: float = 0.5
                 ) -> tuple[list[Clip], list[str]]:
    """Greedy selection by score (with a mild preference for the target duration)."""
    notes: list[str] = []
    ok = [c for c in cands if c.score >= min_score]
    if not ok:
        return [], [f"no candidate reached the minimum score {min_score}; lower it or choose clips manually"]
    ranked = sorted(ok, key=lambda c: -(c.score + 0.04 * (1 - min(1.0, abs(c.duration - target_dur) / target_dur))))
    want = auto_count(duration_s, 10_000) if count == "auto" else int(count)
    chosen: list[Candidate] = []
    for c in ranked:
        if len(chosen) >= (10 if count == "auto" else want):
            break
        if any(overlap_ratio(c.start, c.end, k.start, k.end) > max_overlap for k in chosen):
            continue
        if any(is_duplicate(c.text, k.text, dup_threshold) for k in chosen):
            continue
        chosen.append(c)
    if count == "auto":
        want = auto_count(duration_s, len(chosen))
        chosen = chosen[:want]
    elif len(chosen) < want:
        notes.append(f"requested {want} Shorts but only {len(chosen)} distinct candidates met the quality bar "
                     f"(score >= {min_score}); not padding with weak clips")
    chosen.sort(key=lambda c: c.start)
    clips = [Clip(c.start, c.end, c.score, c.text, c.i, c.j, warnings=list(c.warnings), reasons=list(c.reasons),
                  scores=dict(c.scores)) for c in chosen]
    return clips, notes


def refine_boundaries(sentences: list[Sentence], clip: Clip, media_duration: float,
                      pre_pad: float = 0.20, post_pad: float = 0.35) -> Clip:
    """Add small breathing room without ever reaching into neighbouring speech."""
    if clip.i is None or clip.j is None:
        return clip
    a, b = sentences[clip.i], sentences[clip.j]
    pad_a = min(pre_pad, a.gap_before * 0.5)
    pad_b = min(post_pad, b.gap_after * 0.6) if b.gap_after > 0 else min(post_pad, 0.1)
    clip.start = max(0.0, a.start - pad_a)
    clip.end = min(media_duration, b.end + pad_b)
    return clip


def snap_to_sentences(sentences: list[Sentence], start: float, end: float) -> tuple[int, int, list[str]]:
    """For manual edits: choose the sentence run that best covers [start, end]; report any adjustment."""
    notes = []
    i = min(range(len(sentences)), key=lambda k: abs(sentences[k].start - start))
    j = min(range(len(sentences)), key=lambda k: abs(sentences[k].end - end))
    if j < i:
        j = i
    if abs(sentences[i].start - start) > 0.25:
        notes.append(f"start moved {start:.2f}s -> {sentences[i].start:.2f}s to avoid cutting a sentence")
    if abs(sentences[j].end - end) > 0.25:
        notes.append(f"end moved {end:.2f}s -> {sentences[j].end:.2f}s to avoid cutting a sentence")
    return i, j, notes


def validate_clip(clip: Clip, media_duration: float, min_dur: float, max_dur: float,
                  sentences: list[Sentence] | None = None, others: list[Clip] | None = None,
                  tol: float = 0.6) -> list[dict]:
    """Return list of {check, ok, detail}. Hard failures have ok False and level 'fail'."""
    out = []

    def add(check, ok, detail="", level="fail"):
        out.append({"check": check, "ok": bool(ok), "detail": detail, "level": level})

    add("start_before_end", clip.start < clip.end, f"{clip.start:.2f}->{clip.end:.2f}")
    add("within_media", clip.start >= 0 and clip.end <= media_duration + 0.05,
        f"media {media_duration:.2f}s")
    add("duration_in_limits", min_dur - 0.5 <= clip.duration <= max_dur + 0.5,
        f"{clip.duration:.1f}s (limits {min_dur}-{max_dur})")
    if sentences:
        starts = any(abs(s.start - clip.start) <= tol for s in sentences)
        ends = any(abs(s.end - clip.end) <= tol for s in sentences)
        add("starts_at_sentence_boundary", starts, "", "fail")
        add("ends_at_sentence_boundary", ends, "", "fail")
        # no spoken word is sliced by the cut points
        for s in sentences:
            for w in s.words:
                if w.start < clip.start - 0.02 < w.end - 0.05 or w.start < clip.end + 0.02 < w.end - 0.05:
                    add("no_word_cut", False, f"cut inside word {w.text!r}")
                    break
            else:
                continue
            break
        else:
            add("no_word_cut", True)
    for o in others or []:
        if o is not clip:
            add("no_overlap_with_other_clips", overlap_ratio(clip.start, clip.end, o.start, o.end) <= 0.15,
                f"vs {o.start:.1f}-{o.end:.1f}")
    return out

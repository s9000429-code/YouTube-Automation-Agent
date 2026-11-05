"""Captions shared by both modes: cue model, SRT/ASS writers, readability checks, fonts."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# timing quality labels. Only "verified" means aligned to recognised speech.
TIMING_VERIFIED = "verified"                 # word/segment times from an ASR engine or supplied transcript
TIMING_MEASURED = "measured_sentence"        # sentence times measured from synthesized audio
TIMING_INTERPOLATED = "interpolated"         # words spread across a measured/verified span
TIMING_ESTIMATED = "estimated"               # guessed from text length only. NOT speech-aligned.


@dataclass
class Cue:
    start: float
    end: float
    text: str
    words: list[tuple[str, float, float]] = field(default_factory=list)  # (word, start, end)
    timing: str = TIMING_ESTIMATED

    @property
    def duration(self) -> float:
        return self.end - self.start


def fmt_srt_time(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt_time(s: str) -> float:
    m = re.match(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})", s.strip())
    if not m:
        raise ValueError(f"bad SRT timestamp: {s!r}")
    h, mi, se, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000.0


def write_srt(cues: list[Cue], path: str | Path) -> Path:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{fmt_srt_time(c.start)} --> {fmt_srt_time(c.end)}\n{c.text}\n")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out), encoding="utf-8")
    return p


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = [l for l in block.split("\n") if l.strip() != ""]
        if not lines:
            continue
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            raise ValueError(f"SRT block without timing line: {block[:40]!r}")
        a, _, b = lines[idx].partition("-->")
        cues.append(Cue(parse_srt_time(a), parse_srt_time(b.split()[0]), "\n".join(lines[idx + 1:]),
                        timing=TIMING_VERIFIED))
    return cues


def validate_srt_text(text: str) -> list[str]:
    """Structural check of an SRT: numbering, monotonic non-overlapping times, non-empty text."""
    problems: list[str] = []
    blocks = [b for b in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()) if b.strip()]
    if not blocks:
        return ["no cues"]
    prev_end = -1.0
    for n, block in enumerate(blocks, 1):
        lines = block.split("\n")
        if lines[0].strip() != str(n):
            problems.append(f"cue {n}: expected index {n}, got {lines[0].strip()!r}")
        if len(lines) < 3 or "-->" not in lines[1]:
            problems.append(f"cue {n}: malformed (needs index, time line, text)")
            continue
        try:
            a, _, b = lines[1].partition("-->")
            s, e = parse_srt_time(a), parse_srt_time(b)
        except ValueError as ex:
            problems.append(f"cue {n}: {ex}")
            continue
        if e <= s:
            problems.append(f"cue {n}: end <= start")
        if s < prev_end - 1e-3:
            problems.append(f"cue {n}: overlaps previous cue")
        prev_end = e
    return problems


def readability_check(cues: list[Cue], max_cps: float = 20.0, max_line_chars: int = 42,
                      max_lines: int = 2, min_dur: float = 0.7, max_dur: float = 7.0) -> list[dict]:
    issues: list[dict] = []
    for i, c in enumerate(cues, 1):
        txt = c.text.replace("\\N", "\n")
        lines = txt.split("\n")
        chars = len(txt.replace("\n", " "))
        if c.duration <= 0:
            issues.append({"cue": i, "severity": "error", "issue": "non-positive duration"})
            continue
        cps = chars / c.duration
        if cps > max_cps:
            issues.append({"cue": i, "severity": "warn", "issue": f"too fast: {cps:.1f} chars/s (max {max_cps})"})
        if len(lines) > max_lines:
            issues.append({"cue": i, "severity": "warn", "issue": f"{len(lines)} lines (max {max_lines})"})
        if max(len(l) for l in lines) > max_line_chars:
            issues.append({"cue": i, "severity": "warn", "issue": f"line longer than {max_line_chars} chars"})
        if c.duration < min_dur and chars > 3:
            issues.append({"cue": i, "severity": "warn", "issue": f"shown only {c.duration:.2f}s"})
        if c.duration > max_dur:
            issues.append({"cue": i, "severity": "warn", "issue": f"shown {c.duration:.1f}s (max {max_dur})"})
        if i > 1 and c.start < cues[i - 2].end - 1e-3:
            issues.append({"cue": i, "severity": "error", "issue": "overlaps previous cue"})
    return issues


def wrap_text(text: str, max_chars: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def group_words_into_cues(words: list[tuple[str, float, float]], max_chars: int = 28, max_words: int = 7,
                          max_dur: float = 3.2, gap_break: float = 0.6, timing: str = TIMING_VERIFIED) -> list[Cue]:
    """Pack timed words into short cues (a cue ends at sentence punctuation, long gaps, or size limits)."""
    cues: list[Cue] = []
    cur: list[tuple[str, float, float]] = []

    def flush():
        nonlocal cur
        if cur:
            cues.append(Cue(cur[0][1], cur[-1][2], " ".join(w for w, _, _ in cur), list(cur), timing))
            cur = []

    for w in words:
        if cur:
            chars = len(" ".join(x for x, _, _ in cur)) + 1 + len(w[0])
            if (w[1] - cur[-1][2] > gap_break or chars > max_chars * 2 or len(cur) >= max_words
                    or (w[2] - cur[0][1]) > max_dur):
                flush()
        cur.append(w)
        if re.search(r"[.!?।॥]$", w[0]):
            flush()
    flush()
    return cues


# ---- fonts -------------------------------------------------------------------------------

_SCRIPT_FONTS = {
    "hi": ["Noto Sans Devanagari", "Mukta", "Lohit Devanagari", "Noto Sans"],
    "hinglish": ["Poppins", "Inter", "DejaVu Sans", "Noto Sans"],
    "kn": ["Noto Sans Kannada", "Lohit Kannada", "Tunga"],
    "te": ["Noto Sans Telugu", "Lohit Telugu", "Gautami"],
    "en": ["Poppins", "Inter", "DejaVu Sans", "Liberation Sans", "Arial"],
}
_LANG_CODE = {"hi": "hi", "kn": "kn", "te": "te"}


def _installed_families() -> set[str]:
    if not shutil.which("fc-list"):
        return set()
    try:
        out = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    fams: set[str] = set()
    for line in out.splitlines():
        for f in line.split(","):
            fams.add(f.replace("\\", "").strip())
    return fams


def _is_windows() -> bool:
    from .platform_utils import is_windows
    return is_windows()


def windows_fonts_dir() -> Path:
    return Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"


# Windows ships Nirmala UI, which covers Devanagari, Kannada and Telugu.
_WIN_FONTS = {"Nirmala UI": (["Nirmala.ttf", "Nirmala.ttc"], ["NirmalaB.ttf", "Nirmala.ttc"]),
              "Segoe UI": (["segoeui.ttf"], ["segoeuib.ttf"]),
              "Arial": (["arial.ttf"], ["arialbd.ttf"])}
_WIN_PICK = {"hi": ["Nirmala UI"], "kn": ["Nirmala UI"], "te": ["Nirmala UI"],
             "hinglish": ["Segoe UI", "Arial"], "en": ["Segoe UI", "Arial"]}


def _win_font_path(family: str, bold: bool) -> Path | None:
    regular, bolds = _WIN_FONTS.get(family, ([], []))
    for name in (bolds if bold else regular) or regular:
        p = windows_fonts_dir() / name
        if p.exists():
            return p
    return None


def font_file(language: str, bold: bool = True) -> Path | None:
    """Path of a font file that covers `language`, or None. Works with fontconfig (Linux) or Windows fonts."""
    if _is_windows():
        for fam in _WIN_PICK.get(language, _WIN_PICK["en"]):
            p = _win_font_path(fam, bold)
            if p:
                return p
        return None
    fam, _ = pick_font(language)
    if not shutil.which("fc-match"):
        return None
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}", f"{fam}:bold" if bold else fam], capture_output=True,
                             text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return Path(out) if out and Path(out).exists() else None


def stage_fonts(language: str, work_dir: str | Path) -> str | None:
    """Windows: copy the caption font next to the render so libass finds it (no fontconfig needed).
    Returns the relative fontsdir to use in the ass filter, or None when not needed."""
    if not _is_windows():
        return None
    d = Path(work_dir) / "_fonts"
    d.mkdir(parents=True, exist_ok=True)
    found = False
    for bold in (False, True):
        p = font_file(language, bold)
        if p:
            shutil.copyfile(p, d / p.name)
            found = True
    return "_fonts" if found else None


def pick_font(language: str) -> tuple[str, str | None]:
    """Return (font_family, warning). Warns when no installed font is known to cover the script."""
    if _is_windows():
        for fam in _WIN_PICK.get(language, _WIN_PICK["en"]):
            if _win_font_path(fam, False):
                return fam, None
        warn = (f"Windows font for '{language}' not found in {windows_fonts_dir()}; captions may show boxes"
                if language in ("hi", "kn", "te") else None)
        return "Arial", warn
    fams = _installed_families()
    for cand in _SCRIPT_FONTS.get(language, _SCRIPT_FONTS["en"]):
        if cand in fams:
            return cand, None
    # any font claiming the language via fontconfig
    code = _LANG_CODE.get(language)
    if code and shutil.which("fc-list"):
        try:
            out = subprocess.run(["fc-list", f":lang={code}", "family"], capture_output=True, text=True,
                                 timeout=20).stdout.strip().splitlines()
            names = [o.split(",")[0].replace("\\", "").strip() for o in out if o.strip()]
            good = [n for n in names if "unifont" not in n.lower()]
            if good:
                return good[0], None
            if names:
                return names[0], ("only a bitmap fallback font covers this script; "
                                  "install a Noto Sans font for better caption quality")
        except (OSError, subprocess.TimeoutExpired):
            pass
    fallback = _SCRIPT_FONTS.get(language, _SCRIPT_FONTS["en"])[0]
    warn = None
    if language in ("hi", "kn", "te"):
        warn = (f"no installed font verified for '{language}' script; captions may show boxes. "
                f"Install e.g. '{_SCRIPT_FONTS[language][0]}'.")
    return fallback, warn


# ---- ASS ---------------------------------------------------------------------------------

def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", "\\N")


CAPTION_STYLES = {
    # size = fraction of video width; colors are ASS &HBBGGRR
    "classic":  {"size": 0.062, "bold": 1, "primary": "&H00FFFFFF", "hi": "&H0000FFFF", "outline": 0.006,
                 "back": None, "upper": False, "highlight": False},
    "bold_pop": {"size": 0.075, "bold": 1, "primary": "&H00FFFFFF", "hi": "&H0000E5FF", "outline": 0.009,
                 "back": None, "upper": True, "highlight": True},
    "minimal":  {"size": 0.048, "bold": 0, "primary": "&H00FFFFFF", "hi": "&H00FFFFFF", "outline": 0.0,
                 "back": "&H99000000", "upper": False, "highlight": False},
    "karaoke":  {"size": 0.068, "bold": 1, "primary": "&H00FFFFFF", "hi": "&H0000FF00", "outline": 0.008,
                 "back": None, "upper": False, "highlight": True},
}


def build_ass(cues: list[Cue], width: int, height: int, style: str = "classic", language: str = "en",
              y_frac: float = 0.74, hook_text: str | None = None, hook_seconds: float = 3.0,
              hook_y_frac: float = 0.16, word_highlight: bool | None = None,
              max_line_chars: int | None = None) -> tuple[str, list[str]]:
    """Build an ASS subtitle script. Returns (ass_text, warnings)."""
    st = CAPTION_STYLES.get(style)
    if st is None:
        raise ValueError(f"unknown caption style {style!r}; options: {sorted(CAPTION_STYLES)}")
    font, warn = pick_font(language)
    warnings = [warn] if warn else []
    size = int(round(width * st["size"]))
    outline = max(0, int(round(width * st["outline"])))
    border_style = 3 if st["back"] else 1
    back = st["back"] or "&H64000000"
    line_chars = max_line_chars or max(12, int(width / (size * 0.58)))
    do_hi = st["highlight"] if word_highlight is None else word_highlight
    cx = width // 2

    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
        "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,"
        "MarginV,Encoding\n"
        f"Style: Cap,{font},{size},{st['primary']},&H000000FF,&H00000000,{back},{st['bold']},0,0,0,100,100,0,0,"
        f"{border_style},{outline},0,5,{int(width*0.06)},{int(width*0.06)},40,1\n"
        f"Style: Hook,{font},{int(size*0.95)},&H00FFFFFF,&H000000FF,&H00000000,&H96000000,1,0,0,0,100,100,0,0,3,"
        f"{max(2, outline)},0,5,{int(width*0.08)},{int(width*0.08)},40,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    ev: list[str] = []
    pos = f"{{\\an5\\pos({cx},{int(height * y_frac)})}}"

    def fmt(text: str) -> str:
        return text.upper() if st["upper"] and language in ("en", "hinglish") else text

    for c in cues:
        if c.end <= c.start:
            continue
        if do_hi and c.words:
            ws = [(fmt(w), s, e) for w, s, e in c.words]
            # wrap at word level
            lines, cur, cur_len = [], [], 0
            for i, (w, _, _) in enumerate(ws):
                if cur and cur_len + 1 + len(w) > line_chars:
                    lines.append(cur)
                    cur, cur_len = [], 0
                cur.append(i)
                cur_len += len(w) + (1 if cur_len else 0)
            if cur:
                lines.append(cur)
            for k, (_, ws_s, ws_e) in enumerate(ws):
                end = ws[k + 1][1] if k + 1 < len(ws) and ws[k + 1][1] - ws_e < 0.35 else ws_e
                if k == len(ws) - 1:
                    end = max(end, c.end)
                start = c.start if k == 0 else ws_s
                parts = []
                for line in lines:
                    parts.append(" ".join(
                        (f"{{\\c{st['hi']}}}{_esc(ws[i][0])}{{\\c{st['primary']}}}" if i == k else _esc(ws[i][0]))
                        for i in line))
                joined = "\\N".join(parts)
                ev.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Cap,,0,0,0,,{pos}{joined}")
        else:
            lines = wrap_text(fmt(c.text.replace("\n", " ")), line_chars)
            ev.append(f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Cap,,0,0,0,,{pos}"
                      f"{_esc(chr(10).join(lines))}")
    if hook_text:
        hl = wrap_text(hook_text, max(10, int(line_chars * 0.9)))
        hpos = f"{{\\an5\\pos({cx},{int(height * hook_y_frac)})\\fad(150,200)}}"
        ev.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(hook_seconds)},Hook,,0,0,0,,{hpos}{_esc(chr(10).join(hl))}")
    return header + "\n".join(ev) + "\n", warnings

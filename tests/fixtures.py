"""Synthetic test media + transcripts. No real speech: tones stand in for audio, and the transcript is supplied,
so tests exercise everything EXCEPT real speech recognition."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

# (text, gap_before_seconds). Each sentence is spoken at 0.32 s/word.
LONG_SCRIPT = [
    ("Welcome back to the channel everyone.", 0.4),
    ("Today we are talking about how habits actually form.", 0.3),
    ("Most people think willpower is the secret to change.", 0.3),
    ("Here is the surprising truth about that.", 0.4),
    ("Research shows that ninety percent of daily actions are automatic, not chosen.", 0.3),
    ("That means your environment matters far more than your motivation.", 0.3),
    ("So the key lesson is simple.", 0.4),
    ("Change the environment and the behavior follows without a fight.", 0.3),
    ("Remember that when you plan your week.", 1.2),
    ("Now let me tell you a story from my own life.", 0.8),
    ("When I was twenty years old I tried to run every single morning.", 0.3),
    ("I failed for months and I felt terrible about myself.", 0.3),
    ("One day I simply left my running shoes next to the bed.", 0.4),
    ("And then everything changed without any extra effort.", 0.3),
    ("In the end I ran every day for three years.", 0.3),
    ("The lesson I learned is that small design choices beat big promises.", 0.5),
    ("That is why I never rely on willpower anymore.", 1.4),
    ("Why do some companies grow so much faster than others?", 0.9),
    ("It turns out the answer is not talent or money.", 0.4),
    ("According to a study of two thousand startups, speed of learning was the biggest factor.", 0.3),
    ("Teams that ran more experiments learned faster and survived longer.", 0.3),
    ("The takeaway is to run small tests every single week.", 0.4),
    ("That is how you build a company that keeps improving.", 0.3),
    ("Thanks for watching and see you next time.", 1.0),
]
WPS = 0.32


def build_transcript(script=LONG_SCRIPT, word_level=True):
    t = 1.0
    segs, sid = [], 0
    for text, gap in script:
        t += gap
        words, toks = [], text.split()
        s0 = t
        for w in toks:
            words.append({"text": w, "start": round(t, 3), "end": round(t + WPS * 0.9, 3)})
            t += WPS
        seg = {"id": sid, "start": round(s0, 3), "end": round(t - WPS * 0.1, 3), "text": text,
               "words": words if word_level else []}
        segs.append(seg)
        sid += 1
    return {"language": "en", "engine": "fixture", "segments": segs, "timing": "word" if word_level else "segment",
            "duration": t + 2, "meta": {}}, t + 2


def make_video(path: Path, duration: float, w=640, h=360, fps=25, audio=True, face=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
           f"testsrc2=size={w}x{h}:rate={fps}:duration={duration:.2f}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=16000:duration={duration:.2f}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "64k"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True)
    return path


def write_fixture(dirpath: Path, word_level=True):
    dirpath = Path(dirpath)
    tr, dur = build_transcript(word_level=word_level)
    video = make_video(dirpath / "talk.mp4", dur)
    tpath = dirpath / "talk_transcript.json"
    tpath.write_text(json.dumps(tr), encoding="utf-8")
    return video, tpath, dur


NOTES = """Title: Habit formation overview
URL: https://example.org/habits
Publisher: Example Research Notes
Date: 2024-05-01
License: CC BY 4.0

Habits are behaviors that become automatic through repetition in a stable context. Researchers at University College London found that forming a new habit took a median of 66 days in one study. Environmental cues trigger habitual actions without conscious deliberation. A habit loop consists of a cue, a routine, and a reward. People who plan when and where they will act are more likely to follow through. Reducing friction, such as laying out running shoes, makes a behavior easier to repeat. Missing a single day did not materially affect the habit formation process in that study. Willpower alone is considered an unreliable driver of long-term behavior change. Stable contexts such as the same time and place strengthen the link between cue and action.
"""


def tone_tts():
    """A TTS test double: a tone whose length follows the word count. NOT speech."""
    from providers.tts.base import TTSProvider
    from core.media import ffmpeg

    class ToneTTS(TTSProvider):
        name, kind, natural = "tone_test_double", "offline", False
        languages = ("en",)

        def available(self):
            return True, "test double"

        def synthesize(self, text, out_path, language, voice=None, speed=None, pitch=None):
            dur = max(0.5, len(text.split()) * 0.36)
            ffmpeg(["-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=22050:duration={dur:.3f}", "-ac", "1",
                    "-c:a", "pcm_s16le", out_path])
            return out_path

    return ToneTTS()

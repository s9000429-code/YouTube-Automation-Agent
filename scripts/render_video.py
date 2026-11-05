"""Compose narration + per-section visuals (+ optional music, captions) into the final video."""
from __future__ import annotations

import shutil
from pathlib import Path

from core.captions import stage_fonts
from core.media import concat_line, ffmpeg, probe, quality_args
from core.safe_io import require_free_space, reserved_output

FORMATS = {"landscape": (1920, 1080), "shorts": (1080, 1920), "square": (1080, 1080)}
STYLE = {  # zoom strength per second (Ken Burns), fade seconds
    "educational": (0.010, 0.35), "motivational": (0.020, 0.25), "documentary": (0.014, 0.45),
    "storytelling": (0.016, 0.40), "facts": (0.018, 0.25), "technology": (0.012, 0.30), "science": (0.012, 0.35),
    "news": (0.006, 0.20), "tutorial": (0.004, 0.20), "entertainment": (0.020, 0.25), "comedy": (0.022, 0.20),
    "podcast": (0.008, 0.30), "debate": (0.008, 0.25)}


def section_spans(timeline: list[dict], script: dict, total: float, lead: float = 0.0) -> list[tuple[str, float, float]]:
    """(section_id, start, end) covering [0,total] with no gaps (each section runs until the next begins)."""
    first = {}
    for it in timeline:
        first.setdefault(it["section"], it["start"])
    ids = [s["id"] for s in script["sections"] if s["id"] in first]
    spans = []
    for k, sid in enumerate(ids):
        a = 0.0 if k == 0 else first[sid]
        b = first[ids[k + 1]] if k + 1 < len(ids) else total
        spans.append((sid, a, b))
    return spans


def render_section(asset: dict, dur: float, out: Path, w: int, h: int, fps: int, style: str, settings: dict,
                   profile: str | None) -> None:
    zoom, fade = STYLE.get(style, STYLE["educational"])
    frames = max(1, int(round(dur * fps)))
    fd = min(fade, dur / 3)
    if asset["kind"] == "image":
        big_w, big_h = int(w * 1.5) // 2 * 2, int(h * 1.5) // 2 * 2    # 1.5x oversample: smooth enough, ~45% cheaper than 2x
        vf = (f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,crop={big_w}:{big_h},"
              f"zoompan=z='min(1+{zoom}*on/{fps},1.25)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps},"
              f"fade=t=in:st=0:d={fd:.2f},fade=t=out:st={max(0, dur - fd):.2f}:d={fd:.2f},format=yuv420p")
        ffmpeg(["-loop", "1", "-framerate", str(fps), "-i", asset["path"], "-vf", vf, "-t", f"{dur:.3f}", "-an",
                "-c:v", "libx264", "-preset", settings["encoding"]["preset"], "-crf", str(settings["encoding"]["crf"]),
                "-pix_fmt", "yuv420p", "-r", str(fps), out])
    else:
        vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},"
              f"fade=t=in:st=0:d={fd:.2f},fade=t=out:st={max(0, dur - fd):.2f}:d={fd:.2f},format=yuv420p")
        ffmpeg(["-stream_loop", "-1", "-i", asset["path"], "-vf", vf, "-t", f"{dur:.3f}", "-an", "-c:v", "libx264",
                "-preset", settings["encoding"]["preset"], "-crf", str(settings["encoding"]["crf"]), "-pix_fmt", "yuv420p", out])


def render_video(spans, assets_by_section: dict, narration_wav: str | Path, out_path: str | Path, work_dir: str | Path,
                 settings: dict, fmt: str = "landscape", style: str = "educational", music: str | None = None,
                 music_volume: float = 0.12, ass_path: str | Path | None = None, profile: str | None = None,
                 progress=None, language: str = "en") -> Path:
    w, h = FORMATS[fmt]
    fps = settings["fps"]
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    total = spans[-1][2]
    require_free_space(work, int(total * 2_000_000), reserve_bytes=128 * 1024 * 1024)
    segs = []
    for k, (sid, a, b) in enumerate(spans):
        seg = work / f"seg_{k:03d}.mp4"
        render_section(assets_by_section[sid], b - a, seg, w, h, fps, style, settings, profile)
        segs.append(seg)
        if progress:
            progress((k + 1) / (len(spans) + 1))
    lst = work / "segments.txt"
    lst.write_text("".join(concat_line(s) for s in segs), encoding="utf-8")
    joined = work / "visual.mp4"
    ffmpeg(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined])
    cmd: list = ["-i", joined, "-i", narration_wav]
    amap = "[1:a]aresample=48000,loudnorm=I=-16:TP=-1.5:LRA=11[a]"
    if music and Path(music).is_file():
        cmd += ["-stream_loop", "-1", "-i", music]
        amap = (f"[1:a]aresample=48000,loudnorm=I=-16:TP=-1.5:LRA=11[v];[2:a]aresample=48000,volume={music_volume},"
                f"atrim=0:{total:.3f},afade=t=out:st={max(0, total - 1.5):.2f}:d=1.5[m];"
                f"[v][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]")
    vmap, vfilter = "0:v", []
    cwd = None
    if ass_path:
        local = work / "_burn.ass"
        shutil.copyfile(ass_path, local)
        fontsdir = stage_fonts(language, work)
        vfilter = ["[0:v]ass=_burn.ass" + (f":fontsdir={fontsdir}" if fontsdir else "") + "[vb]"]
        vmap, cwd = "[vb]", work
    fc = ";".join(vfilter + [amap])
    with reserved_output(out_path) as target:
        ffmpeg(cmd + ["-filter_complex", fc, "-map", vmap, "-map", "[a]", "-t", f"{total:.3f}"] +
               (["-c:v", "libx264", "-preset", settings["encoding"]["preset"], "-crf", str(settings["encoding"]["crf"]),
                 "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", settings["encoding"]["audio_bitrate"], "-movflags", "+faststart"]
                if ass_path else ["-c:v", "copy", "-c:a", "aac", "-b:a", settings["encoding"]["audio_bitrate"], "-movflags", "+faststart"])
               + [target], cwd=cwd, duration=total, progress=progress and (lambda f: progress(0.9 + 0.1 * f)))
        final = target
    for s in segs:
        s.unlink(missing_ok=True)
    for p in (lst, joined, work / "_burn.ass"):
        p.unlink(missing_ok=True)
    shutil.rmtree(work / "_fonts", ignore_errors=True)
    return final

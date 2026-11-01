"""Renders one vertical Short with FFmpeg: reframe, captions, hook text, fades, optional zoom/music."""
from __future__ import annotations

import shutil
from pathlib import Path

from core.captions import stage_fonts
from core.media import ffmpeg, quality_args
from core.safe_io import reserved_output, require_free_space

from .vertical_reframer import ReframePlan

EDIT_STYLES = {
    # zoom: slow push-in; fade: seconds in/out; music_gain: relative bed volume; grade: ffmpeg eq args
    "podcast":     {"zoom": False, "fade": 0.15, "music_gain": 0.10, "grade": None},
    "educational": {"zoom": False, "fade": 0.20, "music_gain": 0.08, "grade": "eq=contrast=1.04:saturation=1.05"},
    "motivational": {"zoom": True, "fade": 0.25, "music_gain": 0.14, "grade": "eq=contrast=1.08:saturation=1.12"},
    "storytelling": {"zoom": True, "fade": 0.30, "music_gain": 0.12, "grade": "eq=contrast=1.05:saturation=0.95"},
    "comedy":      {"zoom": False, "fade": 0.10, "music_gain": 0.10, "grade": "eq=saturation=1.15"},
    "documentary": {"zoom": True, "fade": 0.35, "music_gain": 0.10, "grade": "eq=contrast=1.06:saturation=0.9"},
    "news":        {"zoom": False, "fade": 0.10, "music_gain": 0.06, "grade": None},
    "tutorial":    {"zoom": False, "fade": 0.15, "music_gain": 0.06, "grade": None},
}


def edit_style(name: str) -> dict:
    return EDIT_STYLES.get(name, EDIT_STYLES["podcast"])


def build_command(src: Path, start: float, dur: float, plan: ReframePlan, ass_name: str, settings: dict,
                  style: dict, music: str | None, normalize: bool, zoom: bool | None, preview: bool,
                  profile: str | None, out: Path, fps: int, fontsdir: str | None = None) -> list:
    W, H = plan.out_w, plan.out_h
    chain = plan.filter_graph.replace("[vout]", "[vr]")
    vf = [chain]
    last = "[vr]"
    step = 0

    def add(f: str):
        nonlocal last, step
        step += 1
        vf.append(f"{last}{f}[v{step}]")
        last = f"[v{step}]"

    if style.get("grade") and not preview:
        add(style["grade"])
    use_zoom = style["zoom"] if zoom is None else zoom
    if use_zoom and not preview:
        add(f"zoompan=z='min(1+0.0006*on,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={fps}")
    fd = style["fade"]
    add(f"fade=t=in:st=0:d={fd:.2f},fade=t=out:st={max(0.0, dur - fd):.2f}:d={fd:.2f}")
    add(f"ass={ass_name}" + (f":fontsdir={fontsdir}" if fontsdir else ""))
    vf.append(f"{last}null[vfinal]")

    af = []
    a = "[0:a:0]"
    if normalize:
        af.append(f"{a}loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:st=0:d={fd:.2f},"
                  f"afade=t=out:st={max(0.0, dur - fd):.2f}:d={fd:.2f}[a0]")
    else:
        af.append(f"{a}afade=t=in:st=0:d={fd:.2f},afade=t=out:st={max(0.0, dur - fd):.2f}:d={fd:.2f}[a0]")
    amap = "[a0]"
    cmd: list = ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src]
    if music:
        cmd += ["-stream_loop", "-1", "-i", music]
        af.append(f"[1:a]volume={style['music_gain']},atrim=0:{dur:.3f},"
                  f"afade=t=out:st={max(0.0, dur - 0.8):.2f}:d=0.8[m]")
        af.append("[a0][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[amix]")
        amap = "[amix]"
    cmd += ["-filter_complex", ";".join(vf + af), "-map", "[vfinal]", "-map", amap, "-r", str(fps),
            "-t", f"{dur:.3f}"] + quality_args(settings, profile, preview) + [out]
    return cmd


def render_short(src: str | Path, start: float, end: float, out_path: str | Path, plan: ReframePlan,
                 ass_path: str | Path, settings: dict, edit_style_name: str = "podcast", music: str | None = None,
                 normalize: bool = True, zoom: bool | None = None, preview: bool = False,
                 profile: str | None = None, fps: int = 30, progress=None, final: bool = True,
                 language: str = "en") -> Path:
    """Render [start,end] of src. Finals never overwrite (name, name_v2, ...); previews are job-owned."""
    src, out_path = Path(src), Path(out_path)
    dur = end - start
    if dur <= 0:
        raise ValueError("clip has non-positive duration")
    require_free_space(out_path.parent, int(dur * (1_000_000 if preview else 1_500_000)), reserve_bytes=128 * 1024 * 1024)
    style = edit_style(edit_style_name)
    ass_path = Path(ass_path)
    work = ass_path.parent
    local_ass = work / f"_render_{ass_path.name}"
    shutil.copyfile(ass_path, local_ass)          # ffmpeg is run with cwd=work, so the filter needs no path escaping
    fontsdir = stage_fonts(language, work)        # Windows only: libass has no fontconfig there
    try:
        def go(target: Path):
            cmd = build_command(src, start, dur, plan, local_ass.name, settings, style, music, normalize, zoom,
                                preview, profile, target, fps, fontsdir)
            ffmpeg(cmd, duration=dur, progress=progress, cwd=work)

        if final:
            with reserved_output(out_path) as target:
                go(target)
                return target
        out_path.parent.mkdir(parents=True, exist_ok=True)
        go(out_path)
        return out_path
    finally:
        local_ass.unlink(missing_ok=True)

"""Landscape -> vertical reframing.

Modes
  blur_fill   : whole frame kept, centered over a blurred fill. Never crops content (safest; default when
                no single clear subject is found).
  center_crop : fixed centre 9:16 crop. Explicit opt-in only (it can cut off people/text).
  face_track  : ONE static crop per clip centred on the median face position (OpenCV Haar detector).
                It is not per-frame tracking. Used only when a single steady face is found; if people
                are spread across the frame or several faces appear, we fall back to blur_fill.
  auto        : face_track when justified, else blur_fill.
"""
from __future__ import annotations

import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.media import ffmpeg
from core.resources import has_module


@dataclass
class ReframePlan:
    mode: str
    out_w: int
    out_h: int
    filter_graph: str                 # consumes [0:v], produces [vout]
    crop: dict | None = None
    subject_bbox: tuple[float, float, float, float] | None = None   # normalised x0,y0,x1,y1 in OUTPUT frame
    video_band: tuple[float, float] | None = None                    # normalised y-range of real video in output
    notes: list[str] = field(default_factory=list)
    face_ratio: float = 0.0


def _even(x: float) -> int:
    v = int(round(x))
    return v - (v % 2)


def build_filter(mode: str, src_w: int, src_h: int, out_w: int, out_h: int, crop_x: int | None = None) -> tuple[str, dict | None, tuple | None]:
    """Return (filter_graph, crop_dict, video_band)."""
    target = out_w / out_h
    src_ar = src_w / src_h
    if abs(src_ar - target) / target < 0.02:
        return f"[0:v]scale={out_w}:{out_h},setsar=1[vout]", None, (0.0, 1.0)
    if mode == "blur_fill":
        small_w, small_h = _even(out_w / 4), _even(out_h / 4)
        g = (f"[0:v]split=2[bgsrc][fgsrc];"
             f"[bgsrc]scale={small_w}:{small_h}:force_original_aspect_ratio=increase,crop={small_w}:{small_h},"
             f"boxblur=10:2,scale={out_w}:{out_h},eq=brightness=-0.06[bg];"
             f"[fgsrc]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[fg];"
             f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[vout]")
        fg_h = min(out_h, out_w / src_ar) if src_ar >= target else out_h
        y0 = (out_h - fg_h) / 2 / out_h
        return g, None, (y0, 1 - y0)
    # crop modes
    if src_ar >= target:
        cw, ch = _even(src_h * target), _even(src_h)
        cx = _even(crop_x if crop_x is not None else (src_w - cw) / 2)
        cx = max(0, min(cx, src_w - cw))
        cy = 0
    else:   # source narrower than 9:16: crop height instead
        cw, ch = _even(src_w), _even(src_w / target)
        cx, cy = 0, _even((src_h - ch) / 2)
    g = f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={out_w}:{out_h},setsar=1[vout]"
    return g, {"x": cx, "y": cy, "w": cw, "h": ch}, (0.0, 1.0)


def _haar_path() -> str | None:
    if not has_module("cv2"):
        return None
    import cv2  # type: ignore
    p = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return str(p) if p.exists() else None


def detect_faces(video: str | Path, start: float, end: float, max_frames: int = 16) -> tuple[list[list[tuple[float, float, float, float]]], list[str]]:
    """Sample frames in [start,end] and return per-frame lists of normalised face boxes (x0,y0,x1,y1)."""
    haar = _haar_path()
    if not haar:
        return [], ["face detection unavailable (OpenCV or its Haar cascade missing)"]
    import cv2  # type: ignore

    dur = max(0.5, end - start)
    fps = min(1.0, max_frames / dur)
    tmp = Path(tempfile.mkdtemp(prefix="faces_"))
    try:
        ffmpeg(["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", video, "-vf", f"fps={fps:.4f},scale=480:-2",
                "-q:v", "5", tmp / "f_%03d.jpg"])
        det = cv2.CascadeClassifier(haar)
        frames = []
        for img_path in sorted(tmp.glob("f_*.jpg")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape[:2]
            faces = det.detectMultiScale(img, scaleFactor=1.15, minNeighbors=5, minSize=(int(w * 0.06), int(w * 0.06)))
            frames.append([(x / w, y / h, (x + fw) / w, (y + fh) / h) for (x, y, fw, fh) in faces])
        return frames, []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def decide_from_faces(frames: list[list[tuple]]) -> tuple[str, float | None, float | None, tuple | None, float, str]:
    """-> (mode, cx_norm, cy_norm, bbox, face_ratio, reason)"""
    if not frames:
        return "blur_fill", None, None, None, 0.0, "no frames sampled"
    with_face = [f for f in frames if f]
    ratio = len(with_face) / len(frames)
    multi = sum(1 for f in with_face if len(f) >= 2 and max(b[0] for b in f) - min(b[0] for b in f) > 0.2) / len(frames)
    if ratio < 0.4:
        return "blur_fill", None, None, None, ratio, f"faces in only {ratio:.0%} of sampled frames"
    if multi >= 0.3:
        return "blur_fill", None, None, None, ratio, "several people spread across the frame; keeping the full frame"
    main = [max(f, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) for f in with_face]
    cxs = [(b[0] + b[2]) / 2 for b in main]
    if max(cxs) - min(cxs) > 0.25:
        return "blur_fill", None, None, None, ratio, "speaker moves too much for a single static crop"
    cx = statistics.median(cxs)
    cy = statistics.median([(b[1] + b[3]) / 2 for b in main])
    bbox = (min(b[0] for b in main), min(b[1] for b in main), max(b[2] for b in main), max(b[3] for b in main))
    return "face_track", cx, cy, bbox, ratio, f"single steady face in {ratio:.0%} of frames"


def plan_reframe(video: str | Path, info: dict, start: float, end: float, mode: str = "auto",
                 out_w: int = 1080, out_h: int = 1920) -> ReframePlan:
    sw, sh = info["width"], info["height"]
    notes: list[str] = []
    if mode not in ("auto", "blur_fill", "center_crop", "face_track"):
        raise ValueError(f"unknown reframe mode {mode!r}")
    chosen, cx_norm, bbox, ratio = mode, None, None, 0.0
    if mode in ("auto", "face_track"):
        frames, warn = detect_faces(video, start, end)
        notes += warn
        dec, cx, cy, bb, ratio, why = decide_from_faces(frames)
        notes.append(f"reframe decision: {why}")
        if dec == "face_track":
            chosen, cx_norm, bbox = "face_track", cx, bb
        else:
            chosen = "blur_fill"
            if mode == "face_track":
                notes.append("face_track requested but not justified; used blur_fill so nobody is cropped out")
    target = out_w / out_h
    src_ar = sw / sh
    crop_x = None
    if chosen == "face_track":
        cw = _even(sh * target) if src_ar >= target else sw
        crop_x = int(cx_norm * sw - cw / 2)
    g, crop, band = build_filter("center_crop" if chosen == "face_track" else chosen, sw, sh, out_w, out_h, crop_x)
    out_bbox = None
    if chosen == "face_track" and crop and bbox:
        # map face bbox from source-normalised to crop-normalised coords
        x0 = (bbox[0] * sw - crop["x"]) / crop["w"]
        x1 = (bbox[2] * sw - crop["x"]) / crop["w"]
        out_bbox = (max(0.0, x0), bbox[1], min(1.0, x1), bbox[3])
    return ReframePlan(chosen, out_w, out_h, g, crop, out_bbox, band, notes, ratio)


def caption_y_fraction(plan: ReframePlan, default: float = 0.74) -> float:
    """Pick a caption height that avoids faces and (for blur_fill) sits on the blurred fill when possible."""
    y = default
    if plan.mode == "blur_fill" and plan.video_band and plan.video_band[0] > 0.2:
        y = min(0.88, plan.video_band[1] + 0.07)        # below the video, over the blur
    if plan.subject_bbox and plan.subject_bbox[3] > y - 0.06:
        y = min(0.88, plan.subject_bbox[3] + 0.12)       # below the face, never over it
    return y


def hook_y_fraction(plan: ReframePlan, default: float = 0.16) -> float:
    if plan.mode == "blur_fill" and plan.video_band and plan.video_band[0] > 0.2:
        return max(0.08, plan.video_band[0] - 0.08)
    if plan.subject_bbox and plan.subject_bbox[1] < default + 0.06:
        return max(0.05, plan.subject_bbox[1] - 0.06)
    return default

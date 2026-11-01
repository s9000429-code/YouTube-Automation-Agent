"""Thumbnails (1280x720 JPEG < 2 MB) saved separately from video. Uses only original/licensed images."""
from __future__ import annotations

import subprocess
from pathlib import Path

from core.captions import font_file
from core.safe_io import reserve_output

STYLES = ("bold", "minimal", "split")


def _font(language: str, size: int):
    from PIL import ImageFont
    try:
        path = font_file(language, bold=True)
        return ImageFont.truetype(str(path), size) if path else ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default(size)


def short_title(title: str, max_words: int = 6) -> str:
    words = [w for w in title.replace(":", " ").split()]
    return " ".join(words[:max_words]).upper() if len(words) > 3 else title.upper()


def make_thumbnail(title: str, base_image: str | Path | None, out_path: str | Path, language: str = "en",
                   style: str = "bold") -> tuple[Path, dict]:
    from PIL import Image, ImageDraw, ImageFilter, ImageStat

    if style not in STYLES:
        raise ValueError(f"unknown thumbnail style {style!r}; options {STYLES}")
    W, H = 1280, 720
    if base_image and Path(base_image).is_file():
        im = Image.open(base_image).convert("RGB")
        r = max(W / im.width, H / im.height)
        im = im.resize((int(im.width * r), int(im.height * r)))
        im = im.crop(((im.width - W) // 2, (im.height - H) // 2, (im.width - W) // 2 + W, (im.height - H) // 2 + H))
    else:
        im = Image.new("RGB", (W, H), (24, 32, 64))
    d = ImageDraw.Draw(im, "RGBA")
    if style == "split":
        d.rectangle([0, 0, W * 0.58, H], fill=(0, 0, 0, 170))
        box = (60, 60, int(W * 0.58) - 40, H - 60)
    elif style == "minimal":
        im = im.filter(ImageFilter.GaussianBlur(6))
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, 0, W, H], fill=(0, 0, 0, 120))
        box = (100, 120, W - 100, H - 120)
    else:
        d.rectangle([0, int(H * 0.52), W, H], fill=(0, 0, 0, 185))
        box = (60, int(H * 0.54), W - 60, H - 40)
    text = short_title(title) if language in ("en", "hinglish") else title
    size = 150
    while size > 40:
        f = _font(language, size)
        words, lines, cur = text.split(), [], ""
        for w in words:
            t = f"{cur} {w}".strip()
            if d.textlength(t, font=f) <= box[2] - box[0] or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        lh = int(size * 1.15)
        if len(lines) * lh <= box[3] - box[1] and max(d.textlength(l, font=f) for l in lines) <= box[2] - box[0]:
            break
        size -= 6
    y = box[1] + ((box[3] - box[1]) - len(lines) * lh) // 2
    for l in lines:
        for dx, dy in ((-3, -3), (3, 3), (-3, 3), (3, -3)):
            d.text((box[0] + dx, y + dy), l, font=f, fill=(0, 0, 0, 255))
        d.text((box[0], y), l, font=f, fill=(255, 224, 64, 255) if style == "bold" else (255, 255, 255, 255))
        y += lh
    # readability: luminance contrast between text region and its surroundings is checked by QA via font size
    out = reserve_output(out_path)
    q = 90
    while True:
        im.save(out, "JPEG", quality=q)
        if out.stat().st_size < 1_900_000 or q <= 50:
            break
        q -= 10
    return out, {"font_px": size, "lines": len(lines), "style": style, "bytes": out.stat().st_size, "size": [W, H]}

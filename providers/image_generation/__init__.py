"""Image generation providers. Shipped: local_cards (Pillow title/fact cards, original graphics).
Interface allows plugging a diffusion model or stock API later without touching the pipeline."""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from core.captions import font_file
from core.resources import has_module


class ImageProvider(ABC):
    name = "base"
    license = "unknown"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        ...

    @abstractmethod
    def generate(self, heading: str, body: str, out_path: str | Path, width: int, height: int,
                 seed: str = "", language: str = "en") -> Path:
        ...


PALETTES = [((18, 32, 66), (52, 94, 168)), ((30, 20, 54), (122, 62, 158)), ((10, 52, 54), (38, 140, 128)),
            ((54, 26, 20), (176, 84, 52)), ((24, 24, 28), (88, 96, 120)), ((20, 44, 28), (84, 150, 72))]


def _font_path(language: str) -> str | None:
    p = font_file(language, bold=True)
    return str(p) if p else None


class LocalCards(ImageProvider):
    name = "local_cards"
    license = "original_generated"

    def available(self):
        return (True, "ok") if has_module("PIL") else (False, "Pillow not installed")

    def generate(self, heading, body, out_path, width, height, seed="", language="en"):
        from PIL import Image, ImageDraw, ImageFont

        h = int(hashlib.sha256((seed or heading).encode()).hexdigest(), 16)
        c1, c2 = PALETTES[h % len(PALETTES)]
        img = Image.new("RGB", (width, height), c1)
        px = ImageDraw.Draw(img)
        for y in range(height):  # vertical gradient
            t = y / max(1, height - 1)
            px.line([(0, y), (width, y)], fill=tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3)))
        fpath = _font_path(language)

        def font(size):
            try:
                return ImageFont.truetype(fpath, size) if fpath else ImageFont.load_default(size)
            except OSError:
                return ImageFont.load_default(size)

        def wrap(text, f, max_w):
            lines, cur = [], ""
            for word in text.split():
                trial = f"{cur} {word}".strip()
                if px.textlength(trial, font=f) <= max_w or not cur:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = word
            return lines + ([cur] if cur else [])

        margin = int(width * 0.08)
        hf = font(int(min(width, height) * 0.075))
        bf = font(int(min(width, height) * 0.038))
        hl = wrap(heading, hf, width - 2 * margin)[:4]
        bl = wrap(body, bf, width - 2 * margin)[:7] if body else []
        hh = sum(int(hf.size * 1.25) for _ in hl)
        bh = sum(int(bf.size * 1.4) for _ in bl)
        y = (height - hh - bh - (int(height * 0.04) if bl else 0)) // 2
        for line in hl:
            px.text((margin + 3, y + 3), line, font=hf, fill=(0, 0, 0))
            px.text((margin, y), line, font=hf, fill=(255, 255, 255))
            y += int(hf.size * 1.25)
        y += int(height * 0.04)
        for line in bl:
            px.text((margin, y), line, font=bf, fill=(235, 238, 245))
            y += int(bf.size * 1.4)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, quality=92)
        return out


REGISTRY = {"local_cards": LocalCards}


def get_image_provider(name: str = "local_cards", config: dict | None = None) -> ImageProvider:
    if name not in REGISTRY:
        raise KeyError(f"unknown image provider {name!r}; options: {sorted(REGISTRY)}")
    return REGISTRY[name](config or {})

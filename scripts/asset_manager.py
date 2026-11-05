"""Visual assets per section, with licence tracking.

Sources, in order: (1) the user's own library in assets/ (image/video files, optional `<file>.license.json`
sidecar: {"license": "...", "source": "...", "tags": [...], "attribution": "..."}), (2) original generated
cards (providers.image_generation). Files without a sidecar licence are 'unknown' and are NOT used unless
`allow_unknown_assets` is set; when allowed they are flagged. Nothing is downloaded from the web here."""
from __future__ import annotations

import json
import re
from pathlib import Path

from providers.image_generation import ImageProvider

IMG = {".jpg", ".jpeg", ".png", ".webp"}
VID = {".mp4", ".mov", ".mkv", ".webm"}
OPEN_LICENSES = ("cc0", "public domain", "cc by", "cc-by", "cc by-sa", "own", "original_generated", "licensed", "pexels", "pixabay", "unsplash")
RESTRICTED = ("all rights reserved", "no derivatives", "cc by-nd", "cc-nd", "non-commercial", "cc by-nc", "cc-nc", "editorial")


def classify_license(lic: str | None) -> str:
    l = (lic or "").strip().lower()
    if not l or l == "unknown":
        return "unknown"
    if any(r in l for r in RESTRICTED):
        return "restricted"
    if any(o in l for o in OPEN_LICENSES):
        return "ok"
    return "unknown"


def scan_library(lib_dir: str | Path) -> list[dict]:
    lib = Path(lib_dir)
    out = []
    if not lib.is_dir():
        return out
    for f in sorted(lib.rglob("*")):
        if f.suffix.lower() not in IMG | VID or not f.is_file():
            continue
        side = f.with_name(f.name + ".license.json")
        meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        out.append({"path": str(f), "kind": "video" if f.suffix.lower() in VID else "image",
                    "license": meta.get("license", "unknown"), "source": meta.get("source", ""),
                    "attribution": meta.get("attribution", ""),
                    "tags": [t.lower() for t in meta.get("tags", [])] + re.findall(r"[a-z]{4,}", f.stem.lower()),
                    "license_status": classify_license(meta.get("license"))})
    return out


def match(section_text: str, hint: str, lib: list[dict], used: set[str]) -> dict | None:
    words = set(re.findall(r"[a-z]{4,}", (section_text + " " + hint).lower()))
    best, score = None, 0
    for a in lib:
        s = len(words & set(a["tags"])) - (0.5 if a["path"] in used else 0)
        if s > score:
            best, score = a, s
    return best


def assign_assets(script: dict, lib_dir: str | Path, out_dir: str | Path, image_provider: ImageProvider, w: int, h: int,
                  language: str, allow_unknown: bool = False) -> list[dict]:
    lib = scan_library(lib_dir)
    used: set[str] = set()
    result = []
    for k, sec in enumerate(script["sections"]):
        text = " ".join(s["text"] for s in sec["sentences"])
        a = match(text, sec.get("visual_hint", ""), lib, used)
        warnings = []
        if a and a["license_status"] == "restricted":
            warnings.append(f"asset {Path(a['path']).name} has a restrictive licence ({a['license']}); not used")
            a = None
        elif a and a["license_status"] == "unknown":
            if allow_unknown:
                warnings.append(f"asset {Path(a['path']).name} has UNKNOWN usage rights; used only because allow_unknown_assets is on")
            else:
                warnings.append(f"asset {Path(a['path']).name} has no licence info; skipped")
                a = None
        if a:
            used.add(a["path"])
            result.append({"section": sec["id"], **a, "warnings": warnings})
        else:
            card = Path(out_dir) / f"card_{k:02d}.png"
            heading = sec["title"] if sec["kind"] != "body" else (sec.get("visual_hint") or sec["title"]).title()
            body = "" if sec["kind"] in ("hook", "intro", "conclusion") else text[:140]
            image_provider.generate(heading if sec["kind"] != "hook" else script["topic"].title(), body, card, w, h,
                                    seed=sec["id"] + script["topic"], language=language)
            result.append({"section": sec["id"], "path": str(card), "kind": "image", "license": image_provider.license,
                           "source": f"generated:{image_provider.name}", "license_status": "ok", "attribution": "",
                           "tags": [], "warnings": warnings})
    return result

#!/usr/bin/env python3
"""One-time setup for the YouTube Automation Agent (made for Windows 10/11; Linux/macOS: it reports what to install).

  py -3 setup.py            interactive: asks before every download
  py -3 setup.py --check    only report what is installed / missing, change nothing
  py -3 setup.py --yes      accept the recommended choices without asking
  py -3 setup.py --model small --edge-tts --opencv

What it can install (Windows): Pillow, FFmpeg (into ./tools/bin), whisper.cpp (into ./tools/whisper) and a Whisper
model (into ./models). Nothing large is downloaded without your confirmation. It never touches files outside this folder
except pip packages. Downloads come from github.com, huggingface.co, gyan.dev -- these URLs are not verified by the author's
test environment, so if one fails the script tells you exactly which file to download manually and where to put it."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
MODELS = ROOT / "models"

FFMPEG_URLS = [
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip",
]
WHISPER_API = "https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
MODEL_MB = {"tiny": 75, "base": 142, "small": 466, "medium": 1500}
MIN_PY = (3, 10)


def is_windows() -> bool:
    return sys.platform == "win32"


# ---- pure helpers (unit-tested) -----------------------------------------------------------------
def pick_whisper_asset(assets: list[dict]) -> dict | None:
    """Choose the plain CPU Windows x64 build from a GitHub release's assets."""
    names = {a["name"]: a for a in assets}
    if "whisper-bin-x64.zip" in names:
        return names["whisper-bin-x64.zip"]
    for n, a in sorted(names.items()):
        low = n.lower()
        if low.endswith(".zip") and "x64" in low and "bin" in low and not any(k in low for k in ("cublas", "blas", "cuda", "vulkan")):
            return a
    return None


def choose_model(total_ram_mb: int, free_disk_mb: int, requested: str | None = None) -> str:
    """Multilingual model by RAM. A 4 GB PC gets 'base' (about 0.5 GB RAM); 'small' is better for Hindi/Kannada/Telugu
    but needs about 1 GB free RAM, so it is only the default when there is more memory."""
    if requested:
        if requested not in MODEL_MB:
            raise ValueError(f"unknown model {requested!r}; options: {sorted(MODEL_MB)}")
        return requested
    if total_ram_mb and total_ram_mb <= 4608:
        name = "base"
    elif total_ram_mb and total_ram_mb <= 16384:
        name = "small"
    else:
        name = "small"
    while name != "tiny" and MODEL_MB[name] * 2 > free_disk_mb:
        name = {"small": "base", "base": "tiny", "medium": "small"}[name]
    return name


def merge_settings(existing: dict, updates: dict) -> dict:
    out = json.loads(json.dumps(existing))
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_settings(out[k], v)
        else:
            out[k] = v
    return out


def find_in_zip(names: list[str], wanted: list[str]) -> dict[str, str]:
    """Map wanted basenames (lowercase) -> the zip member that has that basename."""
    found = {}
    for n in names:
        base = n.rsplit("/", 1)[-1].lower()
        if base in wanted and base not in found:
            found[base] = n
    return found


# ---- io helpers ----------------------------------------------------------------------------------
def say(msg: str = "") -> None:
    print(msg, flush=True)


def ask(question: str, yes: bool, default: bool = True) -> bool:
    if yes:
        say(f"{question} -> yes (--yes)")
        return True
    hint = "Y/n" if default else "y/N"
    try:
        a = input(f"{question} [{hint}] ").strip().lower()
    except EOFError:
        return default
    return default if not a else a.startswith("y")


def download(url: str, dest: Path, min_bytes: int = 1000) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "YouTubeAutomationAgent-setup/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, last = 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            pct = int(got * 100 / total) if total else -1
            if pct != last and (pct % 10 == 0 or pct == -1):
                say(f"   {got / 1e6:7.1f} MB" + (f" ({pct}%)" if total else ""))
                last = pct
    if got < min_bytes or (total and got != total):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"incomplete download from {url} ({got} of {total or '?'} bytes)")
    if dest.exists():
        dest.unlink()
    part.rename(dest)
    return dest


# ---- integrity ------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def parse_checksum(text: str | None, filename: str | None = None) -> str | None:
    """Extract a SHA-256 from '<hash>', '<hash>  name', 'name: <hash>' or 'sha256:<hash>' text.
    With `filename`, only the line naming that file counts."""
    if not text:
        return None
    for line in text.splitlines():
        m = re.search(r"(?<![0-9a-fA-F])([0-9a-fA-F]{64})(?![0-9a-fA-F])", line)
        if not m:
            continue
        if filename is None or filename in line:
            return m.group(1).lower()
    return None


def fetch_text(url: str, timeout: int = 20) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "YouTubeAutomationAgent-setup/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(200_000).decode("utf-8", "replace")
    except Exception:
        return None


def find_expected_checksum(url: str, filename: str) -> str | None:
    """Look for a checksum published next to the file (<url>.sha256, or a sibling checksums.sha256)."""
    own = parse_checksum(fetch_text(url + ".sha256"))
    if own:
        return own
    return parse_checksum(fetch_text(url.rsplit("/", 1)[0] + "/checksums.sha256"), filename)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def hf_expected_sha256(url: str) -> str | None:
    """Hugging Face serves the LFS SHA-256 in the X-Linked-ETag header of the first (redirect) response."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "YouTubeAutomationAgent-setup/1.0"})
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            resp = opener.open(req, timeout=20)
            headers = resp.headers
        except urllib.error.HTTPError as e:      # the 302 arrives as an HTTPError with the headers we want
            headers = e.headers
        tag = (headers.get("X-Linked-ETag") or headers.get("ETag") or "").strip('"')
        return tag.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", tag) else None
    except Exception:
        return None


def verify_download(path: Path, expected: str | None) -> str:
    """'verified' if the SHA-256 matches. 'unverified' if the source published none. Raises (and deletes the file)
    on a mismatch, so a corrupted or tampered download is never used."""
    if not expected:
        return "unverified (the download source published no checksum)"
    got = sha256_file(path)
    if got != expected.lower():
        path.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {path.name}: expected {expected[:12]}..., got {got[:12]}...")
    return f"verified sha256 {got[:12]}..."


def extract_selected(zip_path: Path, wanted: list[str], dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(zip_path) as z:
        found = find_in_zip(z.namelist(), wanted)
        for base, member in found.items():
            target = dest_dir / Path(member).name
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
    return out


def total_ram_mb() -> int:
    sys.path.insert(0, str(ROOT))
    from core.resources import total_ram_mb as t
    return t()


def free_disk_mb() -> int:
    return shutil.disk_usage(ROOT).free // (1024 * 1024)


def on_path(name: str) -> str | None:
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = str(TOOLS / "bin") + os.pathsep + str(TOOLS / "whisper") + os.pathsep + old
    try:
        return shutil.which(name)
    finally:
        os.environ["PATH"] = old


# ---- steps ----------------------------------------------------------------------------------------
def step_python() -> bool:
    ok = sys.version_info >= MIN_PY
    say(f"[{'ok' if ok else 'FAIL'}] Python {sys.version.split()[0]} (need {MIN_PY[0]}.{MIN_PY[1]}+)")
    return ok


def step_pip(check: bool, yes: bool, extras: list[str]) -> None:
    need = []
    for mod, pkg in (("PIL", "pillow"),):
        try:
            __import__(mod)
            say(f"[ok] {pkg}")
        except ImportError:
            need.append(pkg)
            say(f"[missing] {pkg}")
    for pkg in extras:
        mod = {"edge-tts": "edge_tts", "opencv-python-headless": "cv2"}[pkg]
        try:
            __import__(mod)
            say(f"[ok] {pkg}")
        except ImportError:
            need.append(pkg)
            say(f"[missing] {pkg}")
    if need and not check and ask(f"Install Python packages: {', '.join(need)}?", yes):
        subprocess.run([sys.executable, "-m", "pip", "install", *need], check=False)


def step_ffmpeg(check: bool, yes: bool) -> None:
    if on_path("ffmpeg") and on_path("ffprobe"):
        say(f"[ok] ffmpeg: {on_path('ffmpeg')}")
        return
    say("[missing] ffmpeg / ffprobe")
    if check:
        return
    if not is_windows():
        say("   install with your package manager, e.g. `sudo apt install ffmpeg` or `brew install ffmpeg`")
        return
    if not ask("Download FFmpeg (about 80-200 MB depending on source) into ./tools/bin?", yes):
        return
    for url in FFMPEG_URLS:
        try:
            say(f"   downloading {url}")
            z = download(url, TOOLS / "downloads" / "ffmpeg.zip", 5_000_000)
            say("   integrity: " + verify_download(z, find_expected_checksum(url, url.rsplit("/", 1)[-1])))
            got = extract_selected(z, ["ffmpeg.exe", "ffprobe.exe"], TOOLS / "bin")
            if {p.name for p in got} >= {"ffmpeg.exe", "ffprobe.exe"}:
                z.unlink(missing_ok=True)
                say("[ok] ffmpeg installed to tools/bin")
                return
            say("   archive did not contain ffmpeg.exe and ffprobe.exe; trying next source")
        except Exception as e:
            say(f"   failed: {e}")
    say("[FAIL] could not download FFmpeg. Manual: download an 'essentials' build from https://www.gyan.dev/ffmpeg/builds/ "
        "and copy ffmpeg.exe and ffprobe.exe into the tools\\bin folder here.")


def step_whisper(check: bool, yes: bool) -> str | None:
    exe = on_path("whisper-cli") or on_path("whisper-cpp")
    if exe:
        say(f"[ok] whisper.cpp: {exe}")
        return exe
    say("[missing] whisper.cpp (speech recognition for Mode B)")
    if check or not is_windows():
        if not is_windows():
            say("   build from https://github.com/ggml-org/whisper.cpp or use faster-whisper / --transcript")
        return None
    if not ask("Download whisper.cpp for Windows (about 10-60 MB) into ./tools/whisper?", yes):
        return None
    try:
        req = urllib.request.Request(WHISPER_API, headers={"User-Agent": "YouTubeAutomationAgent-setup/1.0",
                                                           "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            rel = json.loads(r.read())
        asset = pick_whisper_asset(rel.get("assets", []))
        if not asset:
            raise RuntimeError("no Windows x64 CPU build among the release assets")
        say(f"   release {rel.get('tag_name')}: {asset['name']}")
        z = download(asset["browser_download_url"], TOOLS / "downloads" / asset["name"], 100_000)
        say("   integrity: " + verify_download(z, parse_checksum(asset.get("digest"))))
        with zipfile.ZipFile(z) as zf:                       # whisper-cli.exe needs its DLLs next to it
            dest = TOOLS / "whisper"
            dest.mkdir(parents=True, exist_ok=True)
            members = [m for m in zf.namelist() if not m.endswith("/")]
            for m in members:
                if m.lower().endswith((".exe", ".dll")):
                    with zf.open(m) as s, open(dest / Path(m).name, "wb") as d:
                        shutil.copyfileobj(s, d)
        z.unlink(missing_ok=True)
        exe = on_path("whisper-cli")
        if not exe:
            raise RuntimeError("whisper-cli.exe not found in the downloaded archive")
        cp = subprocess.run([exe, "-h"], capture_output=True, text=True, timeout=30)
        if cp.returncode not in (0, 1) and not (cp.stdout or cp.stderr):
            raise RuntimeError("whisper-cli.exe does not start (install 'Microsoft Visual C++ Redistributable 2015-2022 x64')")
        say("[ok] whisper.cpp installed to tools/whisper")
        return exe
    except Exception as e:
        say(f"[FAIL] whisper.cpp: {e}\n   Manual: download whisper-bin-x64.zip from "
            "https://github.com/ggml-org/whisper.cpp/releases and unzip its .exe/.dll files into tools\\whisper")
        return None


def step_model(check: bool, yes: bool, requested: str | None) -> Path | None:
    name = choose_model(total_ram_mb(), free_disk_mb(), requested)
    path = MODELS / f"ggml-{name}.bin"
    existing = sorted(MODELS.glob("ggml-*.bin")) if MODELS.is_dir() else []
    if existing and not requested:
        say(f"[ok] whisper model: {existing[0].name}")
        return existing[0]
    if path.exists():
        say(f"[ok] whisper model: {path.name}")
        return path
    say(f"[missing] whisper model (recommended for this PC: '{name}', about {MODEL_MB[name]} MB)")
    if name == "base":
        say("   note: 'base' is the safe choice for 4 GB RAM. Accuracy for Hindi/Kannada/Telugu is modest; "
            "'small' (--model small) is better but needs about 1 GB free RAM -- close other programs.")
    if check or not ask(f"Download the '{name}' model ({MODEL_MB[name]} MB) into ./models?", yes):
        return None
    try:
        url = MODEL_URL.format(name=name)
        got = download(url, path, MODEL_MB[name] * 700_000)
        say("   integrity: " + verify_download(got, hf_expected_sha256(url)))
        return got
    except Exception as e:
        say(f"[FAIL] model download: {e}\n   Manual: download {MODEL_URL.format(name=name)} and save it as {path}")
        return None


def step_config(whisper_exe: str | None, model: Path | None, check: bool) -> None:
    if check or not (whisper_exe or model):
        return
    cfg = ROOT / "config" / "settings.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}
    if cfg.exists() and not (cfg.with_suffix(".json.bak")).exists():
        shutil.copyfile(cfg, cfg.with_suffix(".json.bak"))
    tx = {"engine": "whisper_cpp", "threads": max(1, min(4, (os.cpu_count() or 2) - 1))}
    if model:
        tx["model_path"] = str(model)
    if whisper_exe:
        tx["whisper_binary"] = whisper_exe
    cfg.write_text(json.dumps(merge_settings(existing, {"shorts": {"transcription": tx}}), indent=2), encoding="utf-8")
    say(f"[ok] updated {cfg.relative_to(ROOT)} (backup: settings.json.bak)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--model", choices=sorted(MODEL_MB))
    ap.add_argument("--edge-tts", action="store_true", help="also install edge-tts (online neural voices; needs internet at run time)")
    ap.add_argument("--opencv", action="store_true", help="also install OpenCV (face-aware reframing; about 50 MB)")
    a = ap.parse_args(argv)
    say(f"YouTube Automation Agent setup  ({sys.platform}, {total_ram_mb()} MB RAM, {free_disk_mb()} MB free disk)\n")
    if not step_python():
        say("Install Python 3.10 or newer from https://www.python.org/downloads/ (tick 'Add python.exe to PATH').")
        return 1
    extras = (["edge-tts"] if a.edge_tts else []) + (["opencv-python-headless"] if a.opencv else [])
    step_pip(a.check, a.yes, extras)
    step_ffmpeg(a.check, a.yes)
    exe = step_whisper(a.check, a.yes)
    model = step_model(a.check, a.yes, a.model)
    step_config(exe, model, a.check)
    say("\nFinal check:")
    subprocess.run([sys.executable, str(ROOT / "agent.py"), "doctor"], check=False)
    say("\nNext: py -3 agent.py video make \"your topic\"   or   py -3 agent.py shorts create your_video.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())

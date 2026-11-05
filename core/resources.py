"""Resource awareness: RAM, disk, CPU, tool discovery, model/preset recommendations."""
from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from .platform_utils import is_windows

# approx. disk (MB) and peak RAM (MB) for whisper.cpp ggml models (from upstream README figures)
WHISPER_MODELS = {
    "tiny": (75, 390), "base": (142, 500), "small": (466, 1000),
    "medium": (1500, 2600), "large": (2900, 4700),
}


def _win_mem() -> tuple[int, int]:
    """(total_mb, available_mb) on Windows via GlobalMemoryStatusEx."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return int(st.ullTotalPhys // (1024 * 1024)), int(st.ullAvailPhys // (1024 * 1024))


def total_ram_mb() -> int:
    if is_windows():
        try:
            return _win_mem()[0]
        except Exception:
            return 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def is_low_ram(threshold_mb: int = 4608) -> bool:
    """True for ~4 GB machines (total RAM at or below the threshold)."""
    total = total_ram_mb()
    return 0 < total <= threshold_mb


def add_local_tools_to_path(root: str | Path) -> list[str]:
    """Prepend <root>/tools/* (where setup.py puts ffmpeg / whisper.cpp) to PATH for this process."""
    added = []
    base = Path(root) / "tools"
    if base.is_dir():
        for d in [base / "bin", *sorted(p for p in base.iterdir() if p.is_dir())]:
            if d.is_dir() and str(d) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                added.append(str(d))
    return added


def mem_available_mb() -> int:
    if is_windows():
        try:
            return _win_mem()[1]
        except Exception:
            return 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def cpu_count() -> int:
    return os.cpu_count() or 1


def disk_free_mb(path: str | Path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free // (1024 * 1024)


def recommend_whisper_model(ram_mb: int, disk_mb: int, multilingual: bool = True) -> str | None:
    """Largest model that fits with headroom (RAM x0.5, disk x0.5). None if even tiny won't fit.
    Half the available RAM is left for the OS, FFmpeg and Python, which matters on 4 GB machines."""
    best = None
    for name, (d, r) in WHISPER_MODELS.items():
        if name == "large":
            continue
        if d <= disk_mb * 0.5 and r <= ram_mb * 0.5:
            best = name
    return best


def check_model_fits(model: str, ram_mb: int, disk_mb: int) -> tuple[bool, str]:
    if model not in WHISPER_MODELS:
        return False, f"unknown model {model!r}"
    d, r = WHISPER_MODELS[model]
    if r > ram_mb * 0.75:
        return False, f"model {model} needs ~{r} MB RAM; only {ram_mb} MB available"
    if d > disk_mb * 0.5:
        return False, f"model {model} needs ~{d} MB disk; only {disk_mb} MB free"
    return True, "ok"


def encoding_preset_for(ram_mb: int, cpus: int) -> str:
    if ram_mb < 2500 or cpus <= 4:
        return "veryfast"
    return "medium" if cpus >= 8 else "fast"


def find_tool(*names: str) -> str | None:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def environment_report(root: str | Path, settings: dict | None = None) -> dict:
    ram, disk = mem_available_mb(), disk_free_mb(root)
    add_local_tools_to_path(root)
    tools = {
        "ffmpeg": find_tool("ffmpeg"), "ffprobe": find_tool("ffprobe"),
        "espeak-ng": find_tool("espeak-ng", "espeak"),
        "whisper.cpp": find_tool("whisper-cli", "whisper-cpp", "whisper.cpp"),
        "fc-list": find_tool("fc-list"),
        "powershell": find_tool("powershell", "pwsh") if is_windows() else None,
    }
    mods = {m: has_module(m) for m in ("PIL", "cv2", "numpy", "faster_whisper", "edge_tts")}
    return {
        "platform": os.name, "ram_total_mb": total_ram_mb(), "low_ram_profile": is_low_ram(),
        "ram_available_mb": ram, "disk_free_mb": disk, "cpus": cpu_count(),
        "tools": tools, "python_modules": mods,
        "recommended_whisper_model": recommend_whisper_model(ram, disk),
        "recommended_encoding_preset": encoding_preset_for(ram, cpu_count()),
        "ready_mode_a": bool(tools["ffmpeg"] and tools["ffprobe"]),
        "ready_mode_b": bool(tools["ffmpeg"] and tools["ffprobe"]),
        "notes": [n for n in [
            None if (tools["espeak-ng"] or tools["powershell"]) else
            "no offline TTS found (install espeak-ng, or on Windows use the built-in 'sapi' voice)",
            None if tools["whisper.cpp"] or mods["faster_whisper"] else
            "no speech-recognition engine found: Mode B needs whisper.cpp, faster-whisper, or a supplied transcript (--transcript)",
            None if mods["PIL"] else "Pillow not installed: thumbnails and generated title cards unavailable",
            None if mods["cv2"] else "OpenCV not installed: face-aware reframing falls back to blur-fill",
        ] if n],
    }

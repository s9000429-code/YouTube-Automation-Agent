"""Windows / low-RAM / installer tests. Windows APIs are mocked: these prove our logic, NOT that it runs on a real
Windows PC (that has to be confirmed on a real machine)."""
import base64
import ctypes
import http.server
import importlib.util
import json
import os
import subprocess
import threading
import zipfile
from pathlib import Path, PureWindowsPath

import pytest

from core import captions, media, recovery_manager, resources
from core.media import MediaError, concat_audio, concat_line
from providers.tts import TTSUnavailable, get_tts, sapi
from providers.tts.espeak import EspeakTTS
from providers.tts.sapi import SapiTTS

ROOT = Path(__file__).resolve().parent.parent


def load_setup():
    spec = importlib.util.spec_from_file_location("yta_setup", ROOT / "setup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- Windows process check --------------------------------------------------------------------
class FakeKernel32:
    def __init__(self, handle=1234, exit_code=259, exit_ok=True):
        self.handle, self.exit_code, self.exit_ok = handle, exit_code, exit_ok
        self.closed, self.opened_with = [], None

    def OpenProcess(self, access, inherit, pid):
        self.opened_with = (access, inherit, pid)
        return self.handle

    def GetExitCodeProcess(self, handle, ref):
        ref._obj.value = self.exit_code                       # ctypes.byref(c_ulong) -> write through
        return 1 if self.exit_ok else 0

    def CloseHandle(self, handle):
        self.closed.append(handle)


def test_windows_pid_alive_running_process():
    k = FakeKernel32(exit_code=259)
    assert recovery_manager._pid_alive_windows(4321, k) is True
    assert k.opened_with == (0x1000, False, 4321) and k.closed == [1234]


def test_windows_pid_dead_when_exited_or_not_found():
    assert recovery_manager._pid_alive_windows(1, FakeKernel32(exit_code=0)) is False
    k = FakeKernel32(handle=0)
    assert recovery_manager._pid_alive_windows(1, k) is False and k.closed == []   # nothing to close
    assert recovery_manager._pid_alive_windows(1, FakeKernel32(exit_ok=False)) is True   # cannot tell -> assume alive


def test_pid_alive_uses_windows_path_not_os_kill(monkeypatch):
    k = FakeKernel32(exit_code=259)
    monkeypatch.setattr(recovery_manager, "is_windows", lambda: True)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: k, raising=False)
    monkeypatch.setattr(os, "kill", lambda *a: (_ for _ in ()).throw(AssertionError("os.kill must not be used on Windows")))
    assert recovery_manager._pid_alive(777) is True and k.opened_with[2] == 777
    assert recovery_manager._pid_alive(0) is False and recovery_manager._pid_alive(None) is False


def test_posix_pid_check_still_works():
    assert recovery_manager._pid_alive(os.getpid()) is True
    assert recovery_manager._pid_alive(2 ** 22 + 99) is False


# ---- RAM / low-memory profile ----------------------------------------------------------------
def test_windows_memory_paths_and_low_ram(monkeypatch):
    monkeypatch.setattr(resources, "is_windows", lambda: True)
    monkeypatch.setattr(resources, "_win_mem", lambda: (4096, 1700))
    assert resources.total_ram_mb() == 4096 and resources.mem_available_mb() == 1700
    assert resources.is_low_ram() is True
    monkeypatch.setattr(resources, "_win_mem", lambda: (16384, 9000))
    assert resources.is_low_ram() is False
    monkeypatch.setattr(resources, "_win_mem", lambda: (_ for _ in ()).throw(OSError("api")))
    assert resources.total_ram_mb() == 0 and resources.mem_available_mb() == 0 and resources.is_low_ram() is False


def test_4gb_machine_gets_small_models_only():
    assert resources.recommend_whisper_model(1700, 20000) == "base"      # ~1.7 GB free on a 4 GB PC
    assert resources.check_model_fits("small", 1300, 20000)[0] is False
    assert resources.check_model_fits("base", 1300, 20000)[0] is True


def test_add_local_tools_to_path(tmp_path, monkeypatch):
    (tmp_path / "tools" / "bin").mkdir(parents=True)
    (tmp_path / "tools" / "whisper").mkdir()
    monkeypatch.setenv("PATH", "/usr/bin")
    added = resources.add_local_tools_to_path(tmp_path)
    assert str(tmp_path / "tools" / "bin") in added and os.environ["PATH"].startswith(str(tmp_path / "tools"))
    assert str(tmp_path / "tools" / "whisper") in os.environ["PATH"]
    assert resources.add_local_tools_to_path(tmp_path) == []                 # idempotent
    assert resources.add_local_tools_to_path(tmp_path / "nothing") == []


def test_ffmpeg_thread_limit_applied(monkeypatch):
    seen = []
    monkeypatch.setattr(media, "run", lambda cmd, timeout=None, cwd=None: seen.append(list(cmd)))
    try:
        media.set_threads(2)
        media.ffmpeg(["-i", "a.mp4", "b.mp4"])
        assert "-threads" in seen[-1] and seen[-1][seen[-1].index("-threads") + 1] == "2"
        media.set_threads(0)
        media.ffmpeg(["-i", "a.mp4", "b.mp4"])
        assert "-threads" not in seen[-1]
    finally:
        media.set_threads(0)


# ---- SAPI provider ----------------------------------------------------------------------------
def test_sapi_encoding_and_rate():
    script = "Write-Output 'हिंदी ✓'"
    assert base64.b64decode(sapi.encode_command(script)).decode("utf-16-le") == script
    assert [sapi.rate_from_speed(s) for s in (None, 160, 200, 120, 400, 10)] == [0, 0, 2, -2, 10, -8]


def test_sapi_unavailable_off_windows_and_without_powershell(monkeypatch):
    assert SapiTTS().available()[0] is False and "Windows" in SapiTTS().available()[1]
    monkeypatch.setattr(sapi, "is_windows", lambda: True)
    monkeypatch.setattr(sapi.shutil, "which", lambda n: None)
    assert SapiTTS().available() == (False, "PowerShell not found")
    monkeypatch.setattr(sapi.shutil, "which", lambda n: r"C:\Windows\System32\powershell.exe")
    assert SapiTTS().available()[0] is True
    assert SapiTTS().natural is False                                        # never advertised as natural/cloned


def _fake_ps(monkeypatch, returncode=0, stderr="", write=True):
    captured = {}

    def fake(script, env=None, timeout=180):
        captured.update(script=script, env=env, text=Path(env["YTA_TEXT"]).read_text(encoding="utf-8"))
        if write:
            Path(env["YTA_OUT"]).write_bytes(b"RIFF" + b"\0" * 500)
        return subprocess.CompletedProcess([], returncode, "Microsoft Zira", stderr)

    monkeypatch.setattr(sapi, "is_windows", lambda: True)
    monkeypatch.setattr(sapi.shutil, "which", lambda n: "powershell.exe")
    monkeypatch.setattr(sapi, "_run_ps", fake)
    return captured


def test_sapi_synthesize_passes_text_voice_rate_and_cultures(monkeypatch, tmp_path):
    cap = _fake_ps(monkeypatch)
    out = SapiTTS().synthesize("यह एक परीक्षण है", tmp_path / "o.wav", "hi", speed=200)
    assert out.exists() and cap["text"] == "यह एक परीक्षण है"               # Unicode survives (UTF-8 temp file)
    assert cap["env"]["YTA_CULTURES"] == "hi" and cap["env"]["YTA_RATE"] == "2" and cap["env"]["YTA_VOICE"] == ""
    assert not Path(cap["env"]["YTA_TEXT"]).exists()                         # temp text file removed
    SapiTTS().synthesize("Namaste doston", tmp_path / "h.wav", "hinglish", voice="Microsoft Heera")
    assert cap["env"]["YTA_CULTURES"] == "en-IN,en" and cap["env"]["YTA_VOICE"] == "Microsoft Heera"
    SapiTTS().synthesize("Hello", tmp_path / "e.wav", "en")
    assert cap["env"]["YTA_CULTURES"].startswith("en-IN,en-US")


def test_sapi_missing_language_pack_gives_actionable_error(monkeypatch, tmp_path):
    _fake_ps(monkeypatch, returncode=3, stderr="NO_VOICE", write=False)
    with pytest.raises(TTSUnavailable) as e:
        SapiTTS().synthesize("ಕನ್ನಡ", tmp_path / "k.wav", "kn")
    assert "kn" in str(e.value) and "Speech" in str(e.value)
    _fake_ps(monkeypatch, returncode=1, stderr="boom", write=False)
    with pytest.raises(TTSUnavailable) as e:
        SapiTTS().synthesize("x", tmp_path / "k.wav", "en")
    assert "Windows speech failed" in str(e.value)


def test_auto_provider_selection(monkeypatch):
    def avail(ok):
        return lambda self: (ok, "mock")

    monkeypatch.setattr(EspeakTTS, "available", avail(True))
    monkeypatch.setattr(SapiTTS, "available", avail(True))
    assert get_tts("auto").name == "espeak"                                  # espeak first when both exist
    monkeypatch.setattr(EspeakTTS, "available", avail(False))
    assert get_tts("auto").name == "sapi"
    monkeypatch.setattr(SapiTTS, "available", avail(False))
    with pytest.raises(TTSUnavailable):
        get_tts("auto")
    with pytest.raises(KeyError):
        get_tts("nonsense")


def test_auto_never_picks_online_or_placeholder_provider():
    from providers.tts import AUTO_ORDER
    assert AUTO_ORDER == ["espeak", "sapi"]


def test_workflow_reports_clear_failure_when_no_tts(env, monkeypatch):
    from core.job_manager import StageFailed
    from workflows.topic_to_video import TopicWorkflow
    monkeypatch.setattr(EspeakTTS, "available", lambda self: (False, "no"))
    monkeypatch.setattr(SapiTTS, "available", lambda self: (False, "no"))
    _, s, _, jm = env
    wf = TopicWorkflow(jm, s, {})
    j = wf.create("How habits form", "en", duration_s=30)
    with pytest.raises(StageFailed) as e:
        wf.run(j)
    assert "TTS unavailable" in str(e.value) and jm.get_job(j)["status"] == "failed"


# ---- fonts -----------------------------------------------------------------------------------
@pytest.fixture()
def win_fonts(tmp_path, monkeypatch):
    d = tmp_path / "WinFonts"
    d.mkdir()
    for n in ("Nirmala.ttf", "NirmalaB.ttf", "segoeui.ttf", "segoeuib.ttf", "arial.ttf", "arialbd.ttf"):
        (d / n).write_bytes(b"FONT" + n.encode())
    monkeypatch.setattr(captions, "_is_windows", lambda: True)
    monkeypatch.setattr(captions, "windows_fonts_dir", lambda: d)
    return d


def test_windows_font_lookup_covers_indian_scripts(win_fonts):
    for lang in ("hi", "kn", "te"):
        assert captions.pick_font(lang) == ("Nirmala UI", None)
        assert captions.font_file(lang, True).name == "NirmalaB.ttf" and captions.font_file(lang, False).name == "Nirmala.ttf"
    assert captions.pick_font("en") == ("Segoe UI", None) and captions.pick_font("hinglish")[0] == "Segoe UI"
    assert captions.font_file("en", True).name == "segoeuib.ttf"


def test_windows_font_fallbacks_and_warnings(win_fonts):
    (win_fonts / "Nirmala.ttf").unlink()
    (win_fonts / "NirmalaB.ttf").unlink()
    fam, warn = captions.pick_font("kn")
    assert fam == "Arial" and warn and "kn" in warn                          # warns instead of silently drawing boxes
    assert captions.font_file("kn") is None
    for f in win_fonts.iterdir():
        f.unlink()
    assert captions.pick_font("en") == ("Arial", None) and captions.font_file("en") is None


def test_stage_fonts_copies_for_libass_only_on_windows(win_fonts, tmp_path, monkeypatch):
    work = tmp_path / "work"
    assert captions.stage_fonts("hi", work) == "_fonts"
    assert sorted(p.name for p in (work / "_fonts").iterdir()) == ["Nirmala.ttf", "NirmalaB.ttf"]
    (work / "_fonts").joinpath("Nirmala.ttf").read_bytes() == (win_fonts / "Nirmala.ttf").read_bytes()
    for f in win_fonts.iterdir():
        f.unlink()
    assert captions.stage_fonts("hi", tmp_path / "w2") is None
    monkeypatch.setattr(captions, "_is_windows", lambda: False)
    assert captions.stage_fonts("hi", tmp_path / "w3") is None and not (tmp_path / "w3").exists()


def test_ass_filter_gets_fontsdir_only_when_given():
    from core.config import load_settings
    from modules.long_form_to_shorts import shorts_editor, vertical_reframer
    s = load_settings(ROOT)
    g, _, band = vertical_reframer.build_filter("blur_fill", 1920, 1080, 1080, 1920)
    plan = vertical_reframer.ReframePlan("blur_fill", 1080, 1920, g, video_band=band)
    style = shorts_editor.edit_style("podcast")

    def fc(fontsdir):
        cmd = shorts_editor.build_command(Path("in.mp4"), 0, 10, plan, "c.ass", s, style, None, True, None, False,
                                          None, Path("o.mp4"), 30, fontsdir)
        return cmd[cmd.index("-filter_complex") + 1]

    assert "ass=c.ass:fontsdir=_fonts[" in fc("_fonts")
    assert "ass=c.ass[" in fc(None) and "fontsdir" not in fc(None)


# ---- FFmpeg concat lists ---------------------------------------------------------------------
def test_concat_line_escapes_and_normalises():
    assert concat_line(Path("/tmp/a b/c.wav")) == "file '/tmp/a b/c.wav'\n"
    assert concat_line(Path("/tmp/O'Brien/c.wav")) == "file '/tmp/O'\\''Brien/c.wav'\n"
    assert concat_line(PureWindowsPath(r"C:\Users\O'Brien\My Videos\seg 1.mp4")) == \
        "file 'C:/Users/O'\\''Brien/My Videos/seg 1.mp4'\n"                  # backslashes -> forward slashes


def test_concat_audio_works_in_folder_with_apostrophe_and_spaces(tmp_path):
    d = tmp_path / "O'Brien's videos"
    d.mkdir()
    for n in ("a", "b"):
        media.ffmpeg(["-f", "lavfi", "-i", "sine=frequency=300:sample_rate=22050:duration=0.5", "-ac", "1", d / f"{n}.wav"])
    out = concat_audio([d / "a.wav", d / "b.wav"], d / "out.wav", d / "list" / "l.txt")
    assert abs(media.probe(out)["duration"] - 1.0) < 0.1


# ---- setup.py helpers -------------------------------------------------------------------------
def test_pick_whisper_asset():
    su = load_setup()
    a = lambda *names: [{"name": n, "browser_download_url": f"u/{n}"} for n in names]
    assert su.pick_whisper_asset(a("whisper-bin-Win32.zip", "whisper-bin-x64.zip", "whisper-cublas-12.4.0-bin-x64.zip"))["name"] == "whisper-bin-x64.zip"
    assert su.pick_whisper_asset(a("whisper-cublas-bin-x64.zip", "whisper-v1-bin-x64.zip"))["name"] == "whisper-v1-bin-x64.zip"
    assert su.pick_whisper_asset(a("whisper-cublas-12.4.0-bin-x64.zip", "whisper-bin-Win32.zip", "source.tar.gz")) is None
    assert su.pick_whisper_asset([]) is None


def test_choose_model_by_ram_and_disk():
    su = load_setup()
    assert su.choose_model(4096, 50000) == "base"                           # the 4 GB Windows PC
    assert su.choose_model(8192, 50000) == "small" and su.choose_model(32000, 50000) == "small"
    assert su.choose_model(4096, 200) == "tiny"                             # nearly full disk
    assert su.choose_model(0, 50000) == "small"                             # RAM unknown -> not the largest
    assert su.choose_model(4096, 50000, "small") == "small"
    with pytest.raises(ValueError):
        su.choose_model(4096, 50000, "huge")


def test_merge_settings_keeps_user_values():
    su = load_setup()
    user = {"fps": 24, "shorts": {"caption_style": "minimal", "transcription": {"chunk_s": 120}}}
    out = su.merge_settings(user, {"shorts": {"transcription": {"engine": "whisper_cpp", "model_path": "m.bin"}}})
    assert out["fps"] == 24 and out["shorts"]["caption_style"] == "minimal"
    assert out["shorts"]["transcription"] == {"chunk_s": 120, "engine": "whisper_cpp", "model_path": "m.bin"}
    assert user["shorts"]["transcription"] == {"chunk_s": 120}               # input not mutated


def test_zip_helpers(tmp_path):
    su = load_setup()
    z = tmp_path / "ff.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("ffmpeg-7.1-essentials_build/bin/ffmpeg.exe", "EXE1")
        zf.writestr("ffmpeg-7.1-essentials_build/bin/ffprobe.exe", "EXE2")
        zf.writestr("ffmpeg-7.1-essentials_build/bin/ffplay.exe", "EXE3")
        zf.writestr("ffmpeg-7.1-essentials_build/README.txt", "x")
    assert set(su.find_in_zip(zipfile.ZipFile(z).namelist(), ["ffmpeg.exe", "ffprobe.exe"])) == {"ffmpeg.exe", "ffprobe.exe"}
    got = su.extract_selected(z, ["ffmpeg.exe", "ffprobe.exe"], tmp_path / "tools" / "bin")
    assert sorted(p.name for p in got) == ["ffmpeg.exe", "ffprobe.exe"]
    assert (tmp_path / "tools" / "bin" / "ffprobe.exe").read_text() == "EXE2" and not (tmp_path / "tools" / "bin" / "ffplay.exe").exists()


class _Server:
    """Local HTTP server: serves a body, optionally lying about Content-Length (truncated download)."""
    def __init__(self, body: bytes, claimed: int | None = None):
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(claimed if claimed is not None else len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}/file.bin"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def test_download_success_and_truncation_detected(tmp_path):
    su = load_setup()
    body = os.urandom(3000)
    s = _Server(body)
    try:
        p = su.download(s.url, tmp_path / "d" / "f.bin", min_bytes=1000)
        assert p.read_bytes() == body and not list((tmp_path / "d").glob("*.part"))
    finally:
        s.close()
    s = _Server(body, claimed=9000)                                          # server says 9000 but sends 3000
    try:
        with pytest.raises(Exception):
            su.download(s.url, tmp_path / "d" / "bad.bin", min_bytes=1000)
        assert not (tmp_path / "d" / "bad.bin").exists() and not list((tmp_path / "d").glob("bad*.part"))
    finally:
        s.close()
    s = _Server(b"tiny")
    try:
        with pytest.raises(RuntimeError):
            su.download(s.url, tmp_path / "d" / "t.bin", min_bytes=1000)      # suspiciously small
    finally:
        s.close()


def test_step_config_merges_backs_up_and_respects_check(tmp_path, monkeypatch):
    su = load_setup()
    monkeypatch.setattr(su, "ROOT", tmp_path)
    cfg = tmp_path / "config" / "settings.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"fps": 24, "shorts": {"caption_style": "minimal"}}))
    su.step_config("C:/t/whisper-cli.exe", Path("C:/m/ggml-base.bin"), check=True)
    assert json.loads(cfg.read_text()) == {"fps": 24, "shorts": {"caption_style": "minimal"}}   # --check changes nothing
    su.step_config("C:/t/whisper-cli.exe", Path("C:/m/ggml-base.bin"), check=False)
    data = json.loads(cfg.read_text())
    assert data["fps"] == 24 and data["shorts"]["caption_style"] == "minimal"
    tx = data["shorts"]["transcription"]
    assert tx["engine"] == "whisper_cpp" and tx["whisper_binary"].endswith("whisper-cli.exe") and tx["threads"] >= 1
    assert json.loads(cfg.with_suffix(".json.bak").read_text())["fps"] == 24
    su.step_config(None, None, check=False)                                  # nothing found -> untouched
    assert json.loads(cfg.read_text()) == data


def test_ask_respects_yes_and_default(monkeypatch):
    su = load_setup()
    assert su.ask("go?", yes=True) is True
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert su.ask("go?", yes=False, default=False) is False
    monkeypatch.setattr("builtins.input", lambda *_: "Y")
    assert su.ask("go?", yes=False, default=False) is True
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert su.ask("go?", yes=False) is False


def test_setup_check_mode_makes_no_downloads_or_files(tmp_path, monkeypatch, capsys):
    su = load_setup()
    monkeypatch.setattr(su, "ROOT", tmp_path)
    monkeypatch.setattr(su, "TOOLS", tmp_path / "tools")
    monkeypatch.setattr(su, "MODELS", tmp_path / "models")
    monkeypatch.setattr(su, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("download in --check")))
    monkeypatch.setattr(su, "on_path", lambda n: None)
    su.step_ffmpeg(True, False)
    assert su.step_whisper(True, False) is None and su.step_model(True, False, None) is None
    assert not (tmp_path / "tools").exists() and not (tmp_path / "models").exists()
    assert "[missing] ffmpeg" in capsys.readouterr().out


def _setup_in(tmp_path, monkeypatch, windows=True):
    su = load_setup()
    monkeypatch.setattr(su, "ROOT", tmp_path)
    monkeypatch.setattr(su, "TOOLS", tmp_path / "tools")
    monkeypatch.setattr(su, "MODELS", tmp_path / "models")
    monkeypatch.setattr(su, "is_windows", lambda: windows)
    monkeypatch.setattr(su, "fetch_text", lambda *a, **k: None)          # unit tests never touch the network
    monkeypatch.setattr(su, "hf_expected_sha256", lambda *a, **k: None)
    return su


def _make_zip(path: Path, members: dict):
    with zipfile.ZipFile(path, "w") as z:
        for n, c in members.items():
            z.writestr(n, c)
    return path


def test_declined_consent_downloads_nothing(tmp_path, monkeypatch):
    su = _setup_in(tmp_path, monkeypatch)
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded without consent"))
    monkeypatch.setattr(su, "download", boom)
    monkeypatch.setattr(su.urllib.request, "urlopen", boom)
    monkeypatch.setattr(su, "ask", lambda *a, **k: False)
    monkeypatch.setattr(su, "on_path", lambda n: None)
    su.step_ffmpeg(False, False)
    assert su.step_whisper(False, False) is None and su.step_model(False, False, "base") is None
    assert not (tmp_path / "tools").exists() and not (tmp_path / "models").exists()


def test_windows_ffmpeg_install_flow_with_fallback_source(tmp_path, monkeypatch):
    su = _setup_in(tmp_path, monkeypatch)
    monkeypatch.setattr(su, "ask", lambda *a, **k: True)
    monkeypatch.setattr(su, "on_path", lambda n: None)
    calls = []

    def fake_download(url, dest, min_bytes=1000):
        calls.append(url)
        if "gyan.dev" in url:
            raise RuntimeError("blocked")                                  # first source fails -> fallback is tried
        return _make_zip(dest, {"ff/bin/ffmpeg.exe": "E1", "ff/bin/ffprobe.exe": "E2"}) if dest.parent.mkdir(parents=True, exist_ok=True) is None else None

    monkeypatch.setattr(su, "download", fake_download)
    su.step_ffmpeg(False, False)
    assert len(calls) == 2 and "gyan.dev" in calls[0] and "BtbN" in calls[1]
    assert (tmp_path / "tools" / "bin" / "ffmpeg.exe").read_text() == "E1"
    assert not (tmp_path / "tools" / "downloads" / "ffmpeg.zip").exists()   # archive cleaned up


def test_windows_ffmpeg_all_sources_fail_prints_manual_steps(tmp_path, monkeypatch, capsys):
    su = _setup_in(tmp_path, monkeypatch)
    monkeypatch.setattr(su, "ask", lambda *a, **k: True)
    monkeypatch.setattr(su, "on_path", lambda n: None)
    monkeypatch.setattr(su, "download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    su.step_ffmpeg(False, False)
    out = capsys.readouterr().out
    assert "[FAIL]" in out and "gyan.dev" in out and "tools\\bin" in out


def test_windows_whisper_install_flow(tmp_path, monkeypatch):
    su = _setup_in(tmp_path, monkeypatch)
    monkeypatch.setattr(su, "ask", lambda *a, **k: True)
    release = {"tag_name": "v9.9.9", "assets": [
        {"name": "whisper-bin-x64.zip", "browser_download_url": "https://example.invalid/w.zip"},
        {"name": "whisper-cublas-bin-x64.zip", "browser_download_url": "https://example.invalid/c.zip"}]}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(release).encode()

    monkeypatch.setattr(su.urllib.request, "urlopen", lambda *a, **k: Resp())
    fetched = []

    def fake_download(url, dest, min_bytes=1000):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        return _make_zip(dest, {"Release/whisper-cli.exe": "EXE", "Release/whisper.dll": "DLL", "Release/ggml.dll": "DLL2",
                                "Release/readme.md": "x"})

    monkeypatch.setattr(su, "download", fake_download)
    monkeypatch.setattr(su, "on_path", lambda n: str(tmp_path / "tools" / "whisper" / "whisper-cli.exe")
                        if n == "whisper-cli" and (tmp_path / "tools" / "whisper" / "whisper-cli.exe").exists() else None)
    monkeypatch.setattr(su.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "usage", ""))
    exe = su.step_whisper(False, False)
    assert fetched == ["https://example.invalid/w.zip"]                       # plain CPU build, never the cublas one
    assert exe and sorted(p.name for p in (tmp_path / "tools" / "whisper").iterdir()) == ["ggml.dll", "whisper-cli.exe", "whisper.dll"]


def test_windows_whisper_failure_prints_manual_steps(tmp_path, monkeypatch, capsys):
    su = _setup_in(tmp_path, monkeypatch)
    monkeypatch.setattr(su, "ask", lambda *a, **k: True)
    monkeypatch.setattr(su, "on_path", lambda n: None)
    monkeypatch.setattr(su.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("no network")))
    assert su.step_whisper(False, False) is None
    out = capsys.readouterr().out
    assert "[FAIL] whisper.cpp" in out and "whisper-bin-x64.zip" in out and "tools\\whisper" in out


def test_model_step_downloads_only_chosen_model_and_reports_failure(tmp_path, monkeypatch, capsys):
    su = _setup_in(tmp_path, monkeypatch)
    monkeypatch.setattr(su, "ask", lambda *a, **k: True)
    monkeypatch.setattr(su, "total_ram_mb", lambda: 4096)
    seen = []
    monkeypatch.setattr(su, "download", lambda url, dest, min_bytes=0: seen.append((url, dest, min_bytes)) or dest)
    assert su.step_model(False, False, None).name == "ggml-base.bin"
    assert seen[0][0].endswith("/ggml-base.bin") and "huggingface.co/ggerganov/whisper.cpp" in seen[0][0] and seen[0][2] > 90_000_000
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models" / "ggml-base.bin").write_bytes(b"x")
    seen.clear()
    assert su.step_model(False, False, None).name == "ggml-base.bin" and not seen       # already present: no re-download
    monkeypatch.setattr(su, "download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net")))
    assert su.step_model(False, False, "tiny") is None
    assert "Manual: download" in capsys.readouterr().out


# ---- integrity checks (local server, no internet) --------------------------------------------
H64 = "a" * 64


def test_parse_checksum_formats():
    su = load_setup()
    h = "0123456789abcdef" * 4
    assert su.parse_checksum(h) == h and su.parse_checksum(h.upper() + "  file.zip\n") == h
    assert su.parse_checksum(f"sha256:{h}") == h
    multi = f"{'b' * 64}  other.zip\n{h} *ffmpeg.zip\n"
    assert su.parse_checksum(multi, "ffmpeg.zip") == h and su.parse_checksum(multi, "missing.zip") is None
    assert su.parse_checksum(multi) == "b" * 64                              # no filename -> first hash
    for bad in (None, "", "not a hash", "g" * 64, "a" * 63, "a" * 65):
        assert su.parse_checksum(bad) is None


def test_verify_download_match_mismatch_and_unavailable(tmp_path):
    su = load_setup()
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    good = su.sha256_file(f)
    assert su.verify_download(f, good).startswith("verified") and su.verify_download(f, good.upper()).startswith("verified")
    assert su.verify_download(f, None).startswith("unverified") and f.exists()       # honest: never reports 'verified'
    with pytest.raises(RuntimeError) as e:
        su.verify_download(f, H64)
    assert "mismatch" in str(e.value) and not f.exists()                              # bad file is deleted, not kept


class _Files(http.server.BaseHTTPRequestHandler):
    routes = {}

    def do_GET(self):
        body = self.routes.get(self.path)
        if body is None:
            self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_HEAD(self):
        if self.path == "/resolve/ggml-base.bin":
            self.send_response(302)
            self.send_header("X-Linked-ETag", f'"{H64}"')
            self.send_header("Location", "/cdn/ggml-base.bin")
            self.end_headers()
        elif self.path == "/resolve/no-etag.bin":
            self.send_response(302); self.send_header("Location", "/cdn/x"); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *a):
        pass


def _serve(routes):
    handler = type("H", (_Files,), {"routes": routes})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_find_expected_checksum_sources_and_hf_header():
    su = load_setup()
    h1, h2 = "1" * 64, "2" * 64
    srv, base = _serve({"/a/file.zip.sha256": f"{h1}  file.zip\n".encode(),
                        "/b/checksums.sha256": f"{h2}  other.zip\n{h1}  file.zip\n".encode()})
    try:
        assert su.find_expected_checksum(f"{base}/a/file.zip", "file.zip") == h1          # <url>.sha256
        assert su.find_expected_checksum(f"{base}/b/file.zip", "file.zip") == h1          # sibling checksums.sha256
        assert su.find_expected_checksum(f"{base}/b/unknown.zip", "unknown.zip") is None
        assert su.find_expected_checksum(f"{base}/nothing/file.zip", "file.zip") is None  # 404 -> unverified
        assert su.fetch_text(f"{base}/missing") is None
        assert su.hf_expected_sha256(f"{base}/resolve/ggml-base.bin") == H64              # header on the redirect
        assert su.hf_expected_sha256(f"{base}/resolve/no-etag.bin") is None
    finally:
        srv.shutdown()


def test_ffmpeg_install_rejects_wrong_checksum_and_tries_next_source(tmp_path, monkeypatch, capsys):
    su = _setup_in(tmp_path, monkeypatch)
    monkeypatch.setattr(su, "ask", lambda *a, **k: True)
    monkeypatch.setattr(su, "on_path", lambda n: None)

    def fake_download(url, dest, min_bytes=1000):
        dest.parent.mkdir(parents=True, exist_ok=True)
        return _make_zip(dest, {"b/ffmpeg.exe": "E1" if "gyan" in url else "GOOD", "b/ffprobe.exe": "E2"})

    monkeypatch.setattr(su, "download", fake_download)
    good = {"v": None}

    def expected(url, name):
        if "gyan" in url:
            return H64                                                       # wrong on purpose
        z = tmp_path / "ref.zip"
        _make_zip(z, {"b/ffmpeg.exe": "GOOD", "b/ffprobe.exe": "E2"})
        return None                                                          # BtbN source: no checksum published

    monkeypatch.setattr(su, "find_expected_checksum", expected)
    su.step_ffmpeg(False, False)
    out = capsys.readouterr().out
    assert "checksum mismatch" in out and "unverified" in out                # mismatch refused, fallback flagged unverified
    assert (tmp_path / "tools" / "bin" / "ffmpeg.exe").read_text() == "GOOD"

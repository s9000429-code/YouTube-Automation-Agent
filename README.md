# YouTube Automation Agent

Two independent workflows sharing one core (SQLite jobs, checkpoints, safe file handling):

- **Mode A**: `topic -> research -> script -> narration -> subtitles -> assets -> render -> thumbnail -> metadata -> QA`
- **Mode B**: `long video -> transcript -> highlights -> review/approve -> vertical Shorts + captions + metadata -> QA`

Nothing is uploaded or published. You upload.

---

## Windows 10 setup (4 GB RAM PC)

Needs about 1.5 GB free disk for tools plus space for your videos. Tested on Linux only; the Windows-specific code was tested with
mocks, see [Verification status](#verification-status) before relying on it.

### 1. Install Python 3.10 or newer (once)
Download from <https://www.python.org/downloads/windows/>. In the installer **tick "Add python.exe to PATH"**. Then open
*Command Prompt* and check:
```bat
py -3 --version
```

### 2. Unzip the project
Unzip to a short path without OneDrive/Desktop sync, for example `C:\yta`. Then:
```bat
cd C:\yta\YOUTUBE_AUTOMATION_AGENT
```

### 3. Run the installer (asks before every download)
```bat
setup_windows.bat
```
This is the same as `py -3 setup.py`. It will:
1. check Python, and offer `pip install pillow` (needed for thumbnails and title cards);
2. offer to download **FFmpeg** (80-200 MB) into `tools\bin`;
3. offer to download **whisper.cpp** (Windows CPU build) into `tools\whisper`;
4. offer to download a **Whisper speech model** into `models\` -- on a 4 GB PC it picks **`base`** (142 MB, ~0.5 GB RAM);
5. write the paths into `config\settings.json` (your old file is saved as `settings.json.bak`);
6. run `agent.py doctor` as a final check.

Options:
```bat
py -3 setup.py --check            :: only report what is installed or missing; changes nothing, downloads nothing
py -3 setup.py --yes              :: accept the recommended choices without asking
py -3 setup.py --model small      :: better Hindi/Kannada/Telugu accuracy, needs ~1 GB free RAM (close other programs)
py -3 setup.py --edge-tts         :: also install online neural voices (needs internet while narrating)
py -3 setup.py --opencv           :: also install OpenCV (face-aware reframing, ~50 MB)
```
Where the download source publishes a SHA-256, the installer checks it and refuses a mismatching file. Where none is published it
prints `unverified`. If any download fails it prints the exact manual step (see Troubleshooting).

### 4. Check the machine
```bat
yta.bat doctor
```
`yta.bat` is a short launcher for `py -3 agent.py`. `doctor` lists RAM, free disk, found tools, the recommended model, and anything missing.

### 5. First runs
```bat
:: Mode B: Shorts from a long video. Stops for your approval before the final render.
yta.bat shorts create "D:\videos\talk.mp4" --count 5 --language en
yta.bat shorts list JOB_ID                       :: see the chosen clips (ids, times, scores, warnings)
yta.bat shorts preview JOB_ID                    :: low-resolution previews in outputs\JOB_ID\previews
yta.bat shorts edit JOB_ID CLIP_ID --start 62 --end 98
yta.bat shorts approve JOB_ID all
yta.bat shorts render JOB_ID                     :: add --quality draft when creating the job on slow PCs

:: no speech model yet? use an existing transcript (.json/.srt/.vtt) instead
yta.bat shorts create "D:\videos\talk.mp4" --transcript "D:\videos\talk.srt"

:: Mode A: topic to video
yta.bat video make "How habits form" --duration 120 --language en
yta.bat video make "How habits form" --tts edge_online          :: online neural voice (after setup.py --edge-tts)

yta.bat jobs                 :: all jobs and their status
yta.bat status JOB_ID        :: stages, outputs
yta.bat resume JOB_ID        :: continue an interrupted or failed job from its last good stage
```
Results are in `outputs\JOB_ID\` (`shorts\`, `final_video\`, `thumbnails\`, `metadata\`, `subtitles\`, `reports\summary.json`).
Mode A research: by default Wikipedia (needs internet) plus any notes you list in `config\settings.json` under
`topic_to_video.research.manual.paths`.

### 4 GB RAM guidance
- The low-memory profile switches on automatically when the PC reports 4.5 GB or less: FFmpeg uses 2 threads, speech recognition runs on
  5-minute chunks with 2 threads. Nothing loads a whole video into memory.
- Windows itself uses about 2 GB. Close the browser and other programs before transcribing or rendering.
- Use the **`base`** model (default). `small` may work with everything else closed; `medium` will not.
- If renders are too slow, create jobs with `--quality draft` and review with `shorts preview` first.
- Real speed on a 4 GB PC has **not been measured**; expect full 1080x1920 renders to be slow.

### Voices on Windows (Mode A narration)
| Language | Default offline voice (`--tts auto` picks it) | Notes |
|---|---|---|
| English | Windows built-in voices (SAPI: e.g. David, Zira) | clearer than eSpeak but still basic synthetic speech, not natural, not voice cloning |
| Hindi | only if the Hindi speech pack is installed | Settings > Time & Language > Language > Hindi > Options > install **Speech** |
| Hinglish | an English voice reads the romanised text | pronunciation of Hindi words will be imperfect |
| Kannada, Telugu | usually no offline Windows voice | use `--tts edge_online` (internet) or supply your own TTS command (`config\providers.json` -> `command`) |

If no voice exists for the language the job stops with a message saying so; it never reads text with a wrong-language voice silently.

### Troubleshooting
| Symptom | Fix |
|---|---|
| `'py' is not recognized` | Reinstall Python and tick **Add python.exe to PATH**, or use `python` instead of `py -3`. |
| `required tool not found: ffmpeg` | Run `setup_windows.bat`. Manual: download an *essentials* build from <https://www.gyan.dev/ffmpeg/builds/>, copy `ffmpeg.exe` and `ffprobe.exe` into `tools\bin`. |
| whisper download failed | Manual: from <https://github.com/ggml-org/whisper.cpp/releases> download `whisper-bin-x64.zip`, copy its `.exe` and `.dll` files into `tools\whisper`. |
| model download failed | Manual: download `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin`, save as `models\ggml-base.bin`, re-run `py -3 setup.py`. |
| `whisper-cli.exe` does not start / missing DLL | Install *Microsoft Visual C++ Redistributable 2015-2022 (x64)* from Microsoft, then re-run setup. |
| antivirus or SmartScreen flags downloaded `.exe` | The files are the official FFmpeg / whisper.cpp builds; verify their source yourself and allow them, or install the tools another way and make sure they are on PATH. |
| `no speech-recognition engine found` | Finish step 3, or use `--transcript FILE`. |
| `no Windows voice installed for language 'hi'` | Install the language's Speech pack (table above) or use `--tts edge_online`. |
| captions show boxes for Hindi/Kannada/Telugu | Windows font *Nirmala UI* must exist in `C:\Windows\Fonts`; the job report warns when it cannot be found. |
| `Need ~N MB free` / insufficient storage | Free disk space or point `paths.outputs` in `config\settings.json` at a bigger drive. The job stops before copying or rendering. |
| out of memory / PC freezes | Close other programs, use the `base` or `tiny` model, `--quality draft`. |
| job shows `interrupted` or `failed` | Fix the cause, then `yta.bat resume JOB_ID`; finished stages are not repeated. |
| Hindi text prints as `?` in the console | The agent switches the console to UTF-8; if it still happens, run `chcp 65001` first. |
| antivirus locks files in `outputs` | Exclude the project folder or retry; the agent never overwrites finals, it writes `_v2`, `_v3`. |

---

## Linux / Termux quick start
Termux: `pkg install python ffmpeg espeak`, then `pip install pillow` (optional `opencv-python`).
```bash
python3 agent.py doctor
python3 agent.py video make "How habits form" --duration 120 --language en
python3 agent.py shorts create talk.mp4 --count 5 --transcript talk.srt   # or configure whisper.cpp
python3 agent.py shorts preview JOB | edit JOB ID --start 62 --end 98 | approve JOB all | render JOB
python3 agent.py jobs | status JOB | resume JOB
```
`python3 setup.py --check` reports missing tools on any platform (automatic downloads are Windows-only).

Outputs: `outputs/JOB_ID/{source,research,scripts,narration,transcripts,subtitles,assets,previews,final_video,shorts,thumbnails,metadata,reports}`.

## Safety guarantees (each has a test)
Originals never modified (processing copy + fingerprint re-check before render); finals never overwritten (`_v2`, `_v3`);
only the job's marked `tmp/` is cleaned; free space checked before copying/rendering; interrupted jobs are detected and resumed
from validated checkpoints; DB migrations back up first and roll back on failure; URL input needs `--i-have-rights`
and only accepts direct video files; voice cloning refuses without a valid consent record (no cloning engine ships);
the installer downloads nothing without your confirmation and verifies published checksums.

## Verification status
**Verified (automated tests, Linux):** both workflows end to end on synthetic media, all safety rules, database migrations,
recovery, QA checks, and the new Windows/low-RAM logic (process check, SAPI provider, font lookup, FFmpeg concat lists, installer helpers)
using mocked Windows APIs. See the test results in the project report (`python -m pytest -q`).

**Verified for real (one source):** the installer's FFmpeg step was run against the BtbN GitHub release: the zip downloaded,
its SHA-256 matched BtbN's published checksum, and it contained genuine x64 `ffmpeg.exe` and `ffprobe.exe`.

**NOT verified:** the gyan.dev FFmpeg URL, the whisper.cpp GitHub release lookup and asset names, and the Hugging Face model URL could
not be reached from the build environment. The installer discovers the whisper.cpp release at run time and verifies checksums where the
source provides one, but those three sources have never been exercised end to end. Nothing has run on a real Windows PC, on
real speech, with a real Windows voice, or on Android/Termux.

## Honest limitations
- **Speech recognition is not bundled.** Mode B needs whisper.cpp (+ a model; size is checked against RAM/disk first),
  faster-whisper, or `--transcript`. It was **tested with supplied transcripts only**, never on real speech audio.
- **Voices:** eSpeak and Windows SAPI are basic synthetic TTS (kn/te voice availability is unverified; Latin-script Hinglish via an
  English or Hindi voice is a guess). Online neural TTS (edge-tts) is optional and untested. Narration was **tested only with a tone
  test double**, never a real voice.
- **Mode A subtitles** are timed per sentence from measured audio and labelled `measured_sentence`/`interpolated` -- never "verified".
  QA always warns about this. Run narration through an ASR engine for word-accurate captions.
- **Script writing without an LLM is extractive**: an organised read of researched facts, not creative prose. Hook/intro/outro
  templates exist for en, hi, hinglish only. An optional LLM may reword but is rejected if it adds numbers/names.
- **Understanding/highlights are heuristic** (cue lexicons for en/hi/hinglish + statistics). Kannada/Telugu get only language-independent signals.
  No visual scene understanding. Scores are explainable, not proven. QA cannot prove a clip is not misleading: review before publishing.
- **Reframing:** blur-fill (default), centre crop, or ONE static face-centred crop per clip (OpenCV Haar). No per-frame tracking.
- **Assets:** your own library (`assets/`, licence sidecars `file.ext.license.json`) or generated cards. No stock-site download.
- **Performance on weak hardware is unmeasured.** Full-resolution renders are slow on weak CPUs; use `--quality draft` and previews.
- Not implemented: YouTube API publishing/scheduling, multi-channel, voice-clone backend, per-frame tracking.

## Tests
```bash
python3 -m pytest -q -p no:cacheprovider        # about 5 minutes; most time is real FFmpeg rendering
python3 -m pytest -q tests/test_windows.py      # Windows / low-RAM / installer tests only (seconds)
```
Tests that need Pillow report as *skipped* (with the reason) on an interpreter that lacks it.
Supported Python: 3.10+. Last results (Linux build environment): Python 3.13 with Pillow, **103 passed**; Python 3.11, **97 passed,
6 skipped** (the 6 need Pillow, which could not be installed there because the package index was unreachable). Python 3.10 and 3.12
were not run in full (3.12: core/windows/shorts/topic = 97 passed, 6 skipped). `pip` could not be tested in the build environment.

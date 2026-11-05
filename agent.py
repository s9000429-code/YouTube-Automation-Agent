#!/usr/bin/env python3
"""YouTube Automation Agent CLI.

  agent.py doctor
  agent.py video create "topic" [--language en --style educational --duration 120 --format landscape]
  agent.py video run JOB | agent.py video make "topic" ...
  agent.py shorts create FILE [--count 5 --transcript T.json ...]  -> analyze, review, approve, render
  agent.py jobs | status JOB | recover | resume JOB
Nothing is ever uploaded or published by this tool."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.config import load_settings, resolve_path
from core.database import open_database
from core.job_manager import JobManager, StageFailed
from core.logging_manager import setup_logging
from core.recovery_manager import recover, resume_plan
from core.resources import environment_report
from core.safe_io import SafetyError


def load_providers(root: Path) -> dict:
    p = root / "config" / "providers.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def boot(args):
    from core import media
    from core.resources import add_local_tools_to_path, is_low_ram
    add_local_tools_to_path(args.root)                      # ffmpeg / whisper.cpp installed by setup.py
    settings = load_settings(args.root)
    low = is_low_ram()
    media.set_threads(settings["encoding"].get("threads", 2 if low else 0))
    if low:                                                 # ~4 GB machines: smaller ASR chunks, fewer threads
        tx = settings["shorts"]["transcription"]
        tx["chunk_s"] = min(tx.get("chunk_s", 600), 300)
        tx.setdefault("threads", 2)
    pc = load_providers(Path(args.root))
    if pc.get("research"):
        settings["topic_to_video"]["research"] = {**settings["topic_to_video"]["research"], **pc["research"]}
    if pc.get("llm"):
        settings["topic_to_video"]["llm"] = pc["llm"]
    if pc.get("transcription"):
        settings["shorts"]["transcription"] = {**settings["shorts"]["transcription"], **pc["transcription"]}
    setup_logging(resolve_path(settings, "logs"), console=args.verbose)
    conn = open_database(resolve_path(settings, "database"), resolve_path(settings, "backups"))
    jm = JobManager(conn, settings)
    recovered = recover(jm)
    if recovered:
        print(f"note: marked interrupted job(s): {', '.join(recovered)} (resume with `agent.py resume JOB`)", file=sys.stderr)
    return settings, jm, pc


def pjson(o):
    print(json.dumps(o, indent=2, ensure_ascii=False, default=str))


def cmd_doctor(a):
    settings = load_settings(a.root)
    from providers.tts import list_tts
    pjson({"environment": environment_report(a.root, settings), "tts": list_tts(load_providers(Path(a.root)).get("tts", {}))})


def video_flow(a, settings, jm, pc, job_id=None):
    from workflows.topic_to_video import TopicWorkflow
    wf = TopicWorkflow(jm, settings, pc, progress_cb=(lambda s: print(f"[{s['percent']:5.1f}%] {s['message']}", file=sys.stderr)) if not a.quiet else None)
    return wf, job_id


def cmd_video(a):
    settings, jm, pc = boot(a)
    wf, _ = video_flow(a, settings, jm, pc)
    try:
        if a.action in ("create", "make"):
            opts = dict(style=a.style, duration_s=a.duration, format=a.format, tts=a.tts, voice=a.voice, speed=a.speed,
                        pitch=a.pitch, music=a.music, burn_subtitles=a.burn_subtitles or None, script_file=a.script_file,
                        thumbnail_style=a.thumbnail_style, allow_unknown_assets=a.allow_unknown_assets or None,
                        allow_placeholder_tts=a.allow_placeholder_tts or None)
            jid = wf.create(a.target, a.language, **opts)
            print(f"created job {jid}")
            if a.action == "create":
                return
        else:
            jid = a.target
        pjson(wf.run(jid, force=a.force))
    except StageFailed as e:
        print(f"STOPPED ({e.status}): {e}", file=sys.stderr)
        sys.exit(2)


def cmd_shorts(a):
    from workflows.long_form_to_shorts import ShortsWorkflow
    settings, jm, pc = boot(a)
    wf = ShortsWorkflow(jm, settings, (lambda s: print(f"[{s['percent']:5.1f}%] {s['message']}", file=sys.stderr)) if not a.quiet else None)
    try:
        if a.action == "create":
            opts = dict(count=(a.count if a.count == "auto" else int(a.count)) if a.count else None,
                        target_duration_s=a.target_duration, min_duration_s=a.min_duration, max_duration_s=a.max_duration,
                        caption_style=a.caption_style, editing_style=a.editing_style,
                        clip_selection="manual" if a.manual else None, reframe=a.reframe, zoom=True if a.zoom else None,
                        hook_text=False if a.no_hook else None, music=a.music, generate_metadata=False if a.no_metadata else None,
                        quality=a.quality, transcript_file=a.transcript, auto_approve=a.auto_approve or None, style=a.style)
            jid = wf.create(a.target, a.language, **opts)
            print(f"created job {jid}")
            pjson(wf.analyze(jid))
            if a.auto_approve:
                pjson(wf.render(jid))
            else:
                print("\nReview the clips above, then: shorts preview JOB | approve JOB [IDS|all] | edit | render JOB", file=sys.stderr)
        elif a.action == "analyze":
            pjson(wf.analyze(a.target, force=a.force))
        elif a.action == "list":
            pjson(wf.status(a.target))
        elif a.action == "preview":
            pjson(wf.preview(a.target))
        elif a.action == "approve":
            ids = "all" if not a.ids or a.ids == ["all"] else [int(i) for i in a.ids]
            print(f"approved {wf.approve(a.target, ids)} clip(s)")
        elif a.action == "reject":
            print(f"rejected {wf.reject(a.target, [int(i) for i in a.ids])} clip(s)")
        elif a.action == "edit":
            pjson(wf.edit_clip(a.target, int(a.ids[0]), float(a.start) if a.start else None, float(a.end) if a.end else None, not a.no_snap))
        elif a.action == "add-clip":
            pjson(wf.add_clip(a.target, float(a.start), float(a.end), not a.no_snap))
        elif a.action == "render":
            pjson(wf.render(a.target))
        elif a.action == "fetch-url":
            from modules.long_form_to_shorts.video_analyzer import download_direct_url
            print(download_direct_url(a.target, resolve_path(settings, "outputs") / "_downloads", a.i_have_rights))
    except StageFailed as e:
        print(f"STOPPED ({e.status}): {e}", file=sys.stderr)
        sys.exit(2)
    except (SafetyError, FileNotFoundError, ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_jobs(a):
    _, jm, _ = boot(a)
    for r in jm.list_jobs():
        print(f"{r['id']}  {r['job_type']:<20} {r['status']:<22} {r['stage'] or '-':<12} {r['input_ref'][:50]}")


def cmd_status(a):
    _, jm, _ = boot(a)
    j = jm.get_job(a.target)
    pjson({"job": {k: j[k] for k in j.keys() if k != "params_json"}, "resume_plan": resume_plan(jm, a.target),
           "outputs": [dict(o) for o in jm.outputs(a.target)]})


def cmd_resume(a):
    settings, jm, pc = boot(a)
    job = jm.get_job(a.target)
    try:
        if job["job_type"] == "topic_to_video":
            from workflows.topic_to_video import TopicWorkflow
            pjson(TopicWorkflow(jm, settings, pc).run(a.target))
        else:
            from workflows.long_form_to_shorts import ShortsWorkflow
            wf = ShortsWorkflow(jm, settings)
            pjson(wf.render(a.target) if wf.highlights(a.target, ("approved",)) else wf.analyze(a.target))
    except StageFailed as e:
        print(f"STOPPED ({e.status}): {e}", file=sys.stderr)
        sys.exit(2)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):                 # Windows consoles default to cp1252; Hindi/Kannada/Telugu need UTF-8
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="agent.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    v = sub.add_parser("video")
    v.add_argument("action", choices=["create", "run", "make"])
    v.add_argument("target", help="topic (create/make) or job id (run)")
    v.add_argument("--language", default=None)
    v.add_argument("--style", default=None)
    v.add_argument("--duration", type=int, default=None)
    v.add_argument("--format", default=None, choices=["landscape", "shorts", "square"])
    v.add_argument("--tts", default=None)
    v.add_argument("--voice", default=None)
    v.add_argument("--speed", type=int, default=None)
    v.add_argument("--pitch", type=int, default=None)
    v.add_argument("--music", default=None)
    v.add_argument("--script-file", default=None)
    v.add_argument("--burn-subtitles", action="store_true")
    v.add_argument("--thumbnail-style", default=None)
    v.add_argument("--allow-unknown-assets", action="store_true")
    v.add_argument("--allow-placeholder-tts", action="store_true")
    v.add_argument("--force", action="store_true")
    v.set_defaults(fn=cmd_video)
    s = sub.add_parser("shorts")
    s.add_argument("action", choices=["create", "analyze", "list", "preview", "approve", "reject", "edit", "add-clip", "render", "fetch-url"])
    s.add_argument("target", help="video file (create), job id, or URL (fetch-url)")
    s.add_argument("ids", nargs="*")
    s.add_argument("--language", default=None)
    s.add_argument("--count", default=None, help="3, 5, 10 or auto")
    s.add_argument("--target-duration", type=float, default=None)
    s.add_argument("--min-duration", type=float, default=None)
    s.add_argument("--max-duration", type=float, default=None)
    s.add_argument("--caption-style", default=None, choices=["classic", "bold_pop", "minimal", "karaoke"])
    s.add_argument("--editing-style", default=None)
    s.add_argument("--style", default=None, help="content style used for clip scoring")
    s.add_argument("--reframe", default=None, choices=["auto", "blur_fill", "center_crop", "face_track"])
    s.add_argument("--quality", default=None, choices=["draft", "standard", "high"])
    s.add_argument("--transcript", default=None, help="existing transcript (.json/.srt/.vtt) to skip speech recognition")
    s.add_argument("--music", default=None)
    s.add_argument("--manual", action="store_true")
    s.add_argument("--zoom", action="store_true")
    s.add_argument("--no-hook", action="store_true")
    s.add_argument("--no-metadata", action="store_true")
    s.add_argument("--auto-approve", action="store_true")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--no-snap", action="store_true")
    s.add_argument("--force", action="store_true")
    s.add_argument("--i-have-rights", action="store_true")
    s.set_defaults(fn=cmd_shorts)
    sub.add_parser("jobs").set_defaults(fn=cmd_jobs)
    for name, fn in (("status", cmd_status), ("resume", cmd_resume)):
        x = sub.add_parser(name)
        x.add_argument("target")
        x.set_defaults(fn=fn)
    sub.add_parser("recover").set_defaults(fn=lambda a: boot(a))
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

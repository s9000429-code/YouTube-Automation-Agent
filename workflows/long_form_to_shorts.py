"""Mode B orchestration: long video -> analysed -> (review/approve) -> rendered, QA'd Shorts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from core.job_manager import JobManager, StageFailed, now_iso
from core.logging_manager import get_logger
from core.progress_tracker import ProgressTracker
from core.safe_io import (SafetyError, atomic_write_json, cleanup_job_tmp, make_job_tmp, make_processing_copy,
                          quick_fingerprint, require_free_space, sha256_file)
from modules.long_form_to_shorts import (audio_extractor, caption_generator, clip_selector, content_understanding,
                                         highlight_detector, shorts_editor, shorts_metadata, shorts_qa,
                                         transcription_engine, vertical_reframer, video_analyzer)
from modules.long_form_to_shorts.clip_selector import Clip
from modules.long_form_to_shorts.text_utils import build_sentences, fmt_hms
from providers.llm import get_llm
from providers.transcription import EngineUnavailable, Transcript, choose_engine

log = get_logger("shorts")

ANALYZE_STAGES = ["ingest", "extract_audio", "transcribe", "understand", "highlights", "select"]
RENDER_STAGES = ["render", "qa"]
SHORT_LIMITS_NOTE = ("Heuristic scoring and boundary checks cannot prove a clip is not misleading; review each clip "
                     "before publishing.")


class ShortsWorkflow:
    def __init__(self, jm: JobManager, settings: dict, progress_cb: Callable[[dict], None] | None = None):
        self.jm, self.settings, self.cb = jm, settings, progress_cb

    # ------------------------------------------------------------------ create
    def create(self, source: str | Path, language: str | None = None, **opts) -> str:
        """opts: count, target_duration_s, min_duration_s, max_duration_s, caption_style, editing_style,
        clip_selection ('auto'|'manual'), reframe, zoom, hook_text, music (path|None), generate_metadata,
        quality, transcript_file, auto_approve, style (content style for scoring), confirm_rights."""
        src = Path(source)
        if str(source).startswith(("http://", "https://")):
            raise SafetyError("pass a local file; for direct video URLs use `agent.py shorts fetch-url` "
                              "(requires --i-have-rights)")
        if not src.is_file():
            raise FileNotFoundError(f"source video not found: {src}")
        params = {"source": str(src.resolve()), "language": language or "auto", **{k: v for k, v in opts.items() if v is not None}}
        fmt = f"{self.settings['shorts']['width']}x{self.settings['shorts']['height']}"
        lang = None if language in (None, "auto") else language
        return self.jm.create_job("long_form_to_shorts", str(src), lang, fmt, params)

    # --------------------------------------------------------------- settings
    def cfg(self, job_id: str) -> dict:
        base = dict(self.settings["shorts"])
        p = self.jm.params(job_id)
        m = {"count": "count", "target_duration_s": "target_duration_s", "min_duration_s": "min_duration_s",
             "max_duration_s": "max_duration_s", "caption_style": "caption_style", "editing_style": "editing_style",
             "clip_selection": "clip_selection", "reframe": "reframe", "zoom": "zoom", "hook_text": "hook_text",
             "generate_metadata": "generate_metadata", "quality": "quality", "normalize_audio": "normalize_audio"}
        for k, v in m.items():
            if k in p:
                base[k] = p[k]
        base["music_path"] = p.get("music")
        base["transcript_file"] = p.get("transcript_file")
        base["auto_approve"] = bool(p.get("auto_approve"))
        base["style"] = p.get("style") or base["editing_style"]
        return base

    def _jobs(self, job_id):
        return self.jm.job_dir(job_id)

    def _tracker(self, job_id, stages):
        return ProgressTracker(self.jm, job_id, stages, self.cb)

    # ---------------------------------------------------------------- analyze
    def analyze(self, job_id: str, force: bool = False) -> dict:
        """Stages ingest..select. Ends in review_required (or transcription_required / failed)."""
        jm, job = self.jm, self.jm.get_job(job_id)
        cfg = self.cfg(job_id)
        jdir = self._jobs(job_id)
        make_job_tmp(jdir)
        handler = jm.job_log(job_id)
        tracker = self._tracker(job_id, ANALYZE_STAGES + RENDER_STAGES)
        try:
            def stage(name, fn):
                tracker.start(name)
                ran = jm.run_stage(job_id, name, fn, force=force)
                tracker.finish(name, skipped=not ran)

            stage("ingest", lambda: self._ingest(job_id))
            stage("extract_audio", lambda: self._extract_audio(job_id, tracker))
            stage("transcribe", lambda: self._transcribe(job_id, cfg, tracker))
            stage("understand", lambda: self._understand(job_id, cfg))
            stage("highlights", lambda: self._highlights(job_id, cfg))
            stage("select", lambda: self._select(job_id, cfg))
            if cfg["auto_approve"] and cfg["clip_selection"] == "auto":
                self.approve(job_id, "all")
            jm.set_status(job_id, "review_required", stage="select")
            return self.status(job_id)
        except StageFailed:
            raise
        finally:
            jm.end_job_log(handler)

    def _ingest(self, job_id: str) -> list[str]:
        p = self.jm.params(job_id)
        res = video_analyzer.analyze_video(p["source"])
        if res["errors"]:
            raise StageFailed("; ".join(res["errors"]), "failed")
        info = res["info"]
        cfg = self.cfg(job_id)
        jdir = self._jobs(job_id)
        need = video_analyzer.estimate_disk_need(info, 10 if cfg["count"] == "auto" else int(cfg["count"]),
                                                 cfg["max_duration_s"])
        require_free_space(jdir, need, reserve_bytes=self.settings["storage"]["reserve_mb"] * 1024 * 1024)
        copy, fp = make_processing_copy(p["source"], jdir / "source")
        self.jm.update_params(job_id, processing_copy=str(copy), source_fingerprint=fp, source_info=info,
                              ingest_warnings=res["warnings"])
        atomic_write_json(jdir / "reports" / "source_info.json", {"info": info, "warnings": res["warnings"], "fingerprint": fp})
        return [str(copy)]

    def _extract_audio(self, job_id: str, tracker) -> list[str]:
        p = self.jm.params(job_id)
        jdir = self._jobs(job_id)
        wav = audio_extractor.extract_audio(p["processing_copy"], jdir / "tmp" / "audio_16k.wav",
                                            duration_hint=p["source_info"]["duration"],
                                            progress=lambda f: tracker.update(f, "extracting audio"))
        return [str(wav)]

    def _transcribe(self, job_id: str, cfg: dict, tracker) -> list[str]:
        p = self.jm.params(job_id)
        jdir = self._jobs(job_id)
        tx = cfg["transcription"] if "transcription" in cfg else self.settings["shorts"]["transcription"]
        try:
            engine = choose_engine(tx, cfg.get("transcript_file"))
        except EngineUnavailable as e:
            raise StageFailed(str(e), "transcription_required")
        lang = p.get("language")
        t = transcription_engine.transcribe_audio(engine, jdir / "tmp" / "audio_16k.wav", p["source_info"]["duration"],
                                                  lang, jdir / "tmp", tx.get("chunk_s", 600),
                                                  progress=lambda f: tracker.update(f, "transcribing"))
        problems = t.validate(p["source_info"]["duration"])
        if t.segments and any("after media end" in x for x in problems) and cfg.get("transcript_file"):
            raise StageFailed("supplied transcript runs past the end of the video: " + "; ".join(problems[:3]), "failed")
        if not t.segments:
            raise StageFailed("transcription produced no speech", "failed")
        files = transcription_engine.save_transcript(t, jdir / "transcripts")
        self.jm.conn.execute(
            "INSERT INTO transcripts(job_id,engine,language,path,timing,verified_timing,meta_json,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (job_id, t.engine, t.language, files["json"], t.timing, 1, json.dumps({"problems": problems}), now_iso()))
        self.jm.update_params(job_id, transcript_language=t.language, transcript_timing=t.timing,
                              transcript_engine=t.engine, transcript_problems=problems)
        return [files["json"], files["srt"]]

    def _load(self, job_id: str):
        jdir = self._jobs(job_id)
        t = Transcript.load(jdir / "transcripts" / "transcript.json")
        return t, build_sentences(t)

    def _understand(self, job_id: str, cfg: dict) -> list[str]:
        p = self.jm.params(job_id)
        t, sents = self._load(job_id)
        cmap = content_understanding.understand(sents, t.language, p["source_info"]["duration"])
        path = atomic_write_json(self._jobs(job_id) / "transcripts" / "content_map.json", cmap.to_dict())
        return [str(path)]

    def _highlights(self, job_id: str, cfg: dict) -> list[str]:
        p = self.jm.params(job_id)
        t, sents = self._load(job_id)
        cmap = content_understanding.understand(sents, t.language, p["source_info"]["duration"])
        cands = highlight_detector.generate_candidates(sents, cmap, cfg["min_duration_s"], cfg["max_duration_s"],
                                                       cfg["style"])
        self.jm.conn.execute("DELETE FROM highlights WHERE job_id=? AND status='candidate'", (job_id,))
        top = cands[:80]
        for c in top:
            self.jm.conn.execute(
                "INSERT INTO highlights(job_id,start_s,end_s,score,scores_json,text,status,warnings_json) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (job_id, c.start, c.end, c.score, json.dumps({**c.scores, "_i": c.i, "_j": c.j, "_reasons": c.reasons}),
                 c.text, "candidate", json.dumps(c.warnings)))
        path = atomic_write_json(self._jobs(job_id) / "reports" / "candidates.json",
                                 [c.to_dict() for c in top[:40]])
        self.jm.update_params(job_id, candidate_count=len(cands))
        return [str(path)]

    def _select(self, job_id: str, cfg: dict) -> list[str]:
        p = self.jm.params(job_id)
        if cfg["clip_selection"] == "manual":
            self.jm.update_params(job_id, selection_notes=["manual mode: add clips with `shorts add-clip` then approve"])
            return []
        t, sents = self._load(job_id)
        rows = self.jm.conn.execute("SELECT * FROM highlights WHERE job_id=? AND status='candidate' ORDER BY score DESC",
                                    (job_id,)).fetchall()
        cands = []
        for r in rows:
            sc = json.loads(r["scores_json"])
            c = highlight_detector.Candidate(sc["_i"], sc["_j"], r["start_s"], r["end_s"], r["score"],
                                             {k: v for k, v in sc.items() if not k.startswith("_")}, r["text"],
                                             sc.get("_reasons", []), json.loads(r["warnings_json"] or "[]"))
            cands.append(c)
        dur = p["source_info"]["duration"]
        clips, notes = clip_selector.select_clips(cands, cfg["count"], dur, cfg["target_duration_s"])
        self.jm.conn.execute("DELETE FROM highlights WHERE job_id=? AND status IN ('selected')", (job_id,))
        for k, clip in enumerate(clips, 1):
            clip_selector.refine_boundaries(sents, clip, dur)
            self.jm.conn.execute(
                "INSERT INTO highlights(job_id,start_s,end_s,score,scores_json,text,status,order_idx,warnings_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (job_id, clip.start, clip.end, clip.score, json.dumps({**clip.scores, "_reasons": clip.reasons}),
                 clip.text, "selected", k, json.dumps(clip.warnings)))
        self.jm.update_params(job_id, selection_notes=notes)
        path = atomic_write_json(self._jobs(job_id) / "reports" / "selected.json",
                                 [{"start": c.start, "end": c.end, "score": round(c.score, 3), "text": c.text[:200],
                                   "warnings": c.warnings} for c in clips])
        return [str(path)]

    # ------------------------------------------------------- review operations
    def highlights(self, job_id: str, statuses: tuple[str, ...] = ("selected", "approved", "rejected", "rendered")) -> list[dict]:
        q = ",".join("?" * len(statuses))
        rows = self.jm.conn.execute(
            f"SELECT * FROM highlights WHERE job_id=? AND status IN ({q}) ORDER BY start_s", (job_id, *statuses)).fetchall()
        return [{"id": r["id"], "start": r["start_s"], "end": r["end_s"], "duration": r["end_s"] - r["start_s"],
                 "score": r["score"], "status": r["status"], "text": r["text"], "user_edited": bool(r["user_edited"]),
                 "warnings": json.loads(r["warnings_json"] or "[]"), "hms": f"{fmt_hms(r['start_s'])}-{fmt_hms(r['end_s'])}"}
                for r in rows]

    def approve(self, job_id: str, ids: list[int] | str = "all") -> int:
        if ids == "all":
            cur = self.jm.conn.execute("UPDATE highlights SET status='approved' WHERE job_id=? AND status='selected'", (job_id,))
        else:
            q = ",".join("?" * len(ids))
            cur = self.jm.conn.execute(
                f"UPDATE highlights SET status='approved' WHERE job_id=? AND id IN ({q}) AND status IN ('selected','rejected')",
                (job_id, *ids))
        return cur.rowcount

    def reject(self, job_id: str, ids: list[int]) -> int:
        q = ",".join("?" * len(ids))
        return self.jm.conn.execute(
            f"UPDATE highlights SET status='rejected' WHERE job_id=? AND id IN ({q})", (job_id, *ids)).rowcount

    def edit_clip(self, job_id: str, hid: int, start: float | None = None, end: float | None = None,
                  snap: bool = True) -> dict:
        row = self.jm.conn.execute("SELECT * FROM highlights WHERE id=? AND job_id=?", (hid, job_id)).fetchone()
        if not row:
            raise KeyError(f"no highlight {hid} in job {job_id}")
        s, e = (start if start is not None else row["start_s"]), (end if end is not None else row["end_s"])
        p = self.jm.params(job_id)
        cfg = self.cfg(job_id)
        t, sents = self._load(job_id)
        notes = []
        if snap:
            i, j, notes = clip_selector.snap_to_sentences(sents, s, e)
            if notes:
                s, e = max(0.0, sents[i].start - min(0.2, sents[i].gap_before * 0.5)), min(
                    p["source_info"]["duration"], sents[j].end + min(0.35, sents[j].gap_after * 0.6))
        text = " ".join(x.text for x in sents if x.end > s and x.start < e)
        clip = Clip(s, e, row["score"] or 0.0, text)
        checks = clip_selector.validate_clip(clip, p["source_info"]["duration"], cfg["min_duration_s"],
                                             cfg["max_duration_s"], sents)
        hard = [c for c in checks if not c["ok"] and c["level"] == "fail"]
        if any(c["check"] in ("start_before_end", "within_media", "duration_in_limits") for c in hard):
            raise ValueError("invalid clip: " + "; ".join(f"{c['check']} ({c['detail']})" for c in hard))
        self.jm.conn.execute("UPDATE highlights SET start_s=?, end_s=?, text=?, user_edited=1, status='selected' WHERE id=?",
                             (s, e, text, hid))
        return {"id": hid, "start": s, "end": e, "notes": notes, "checks_failed": [c["check"] for c in hard]}

    def add_clip(self, job_id: str, start: float, end: float, snap: bool = True) -> dict:
        """Manual clip creation (works in manual selection mode)."""
        order = self.jm.conn.execute("SELECT COALESCE(MAX(order_idx),0)+1 AS n FROM highlights WHERE job_id=?", (job_id,)).fetchone()["n"]
        cur = self.jm.conn.execute(
            "INSERT INTO highlights(job_id,start_s,end_s,score,scores_json,text,status,order_idx,user_edited,warnings_json) "
            "VALUES (?,?,?,?,?,?,?,?,1,'[]')", (job_id, start, end, None, "{}", "", "selected", order))
        return self.edit_clip(job_id, cur.lastrowid, start, end, snap)

    # ---------------------------------------------------------------- preview
    def preview(self, job_id: str) -> list[str]:
        """Low-res previews of selected/approved clips so the user can decide before final rendering."""
        p, cfg = self.jm.params(job_id), self.cfg(job_id)
        jdir = self._jobs(job_id)
        t, sents = self._load(job_id)
        out = []
        for h in self.highlights(job_id, ("selected", "approved")):
            res = self._render_clip(job_id, h, sents, t, p, cfg, preview=True)
            out.append(res["path"])
        return out

    # ----------------------------------------------------------------- render
    def _render_clip(self, job_id: str, h: dict, sents, t: Transcript, p: dict, cfg: dict, preview: bool,
                     meta: dict | None = None) -> dict:
        jdir = self._jobs(job_id)
        lang = t.language or "en"
        W, H, fps = cfg["width"], cfg["height"], cfg["fps"]
        if preview:
            W, H = 360, 640
        n = self.jm.conn.execute("SELECT order_idx FROM highlights WHERE id=?", (h["id"],)).fetchone()["order_idx"] or h["id"]
        stem = f"short_{n:02d}"
        info = p["source_info"]
        plan = vertical_reframer.plan_reframe(p["processing_copy"], info, h["start"], h["end"], cfg["reframe"], W, H)
        cc = caption_generator.clip_cues(t, h["start"], h["end"])
        hook = (meta or {}).get("hook_text") if cfg["hook_text"] else None
        files = caption_generator.write_caption_files(
            cc, jdir / "subtitles", stem + ("_preview" if preview else ""), W, H, cfg["caption_style"], lang,
            vertical_reframer.caption_y_fraction(plan), hook, vertical_reframer.hook_y_fraction(plan))
        if preview:
            out = jdir / "previews" / f"{stem}_preview.mp4"
        else:
            out = jdir / "shorts" / f"{stem}.mp4"
        music = cfg["music_path"] if cfg.get("music_path") and Path(cfg["music_path"]).is_file() else None
        path = shorts_editor.render_short(p["processing_copy"], h["start"], h["end"], out, plan, files["ass"],
                                          self.settings, cfg["editing_style"], music, cfg["normalize_audio"],
                                          cfg["zoom"] if cfg["zoom"] else None, preview, cfg["quality"] if not preview else "draft",
                                          fps, final=not preview, language=lang)
        return {"path": str(path), "plan": plan, "captions": cc, "files": files, "stem": stem}

    def render(self, job_id: str, force: bool = False) -> dict:
        jm, cfg, p = self.jm, self.cfg(job_id), self.jm.params(job_id)
        approved = self.highlights(job_id, ("approved",))
        if not approved:
            raise StageFailed("nothing approved to render: run `shorts approve` first (or use --auto-approve)",
                              "review_required")
        handler = jm.job_log(job_id)
        tracker = self._tracker(job_id, ANALYZE_STAGES + RENDER_STAGES)
        jdir = self._jobs(job_id)
        try:
            self._verify_source_untouched(job_id)
            t, sents = self._load(job_id)
            llm = get_llm(self.settings["topic_to_video"].get("llm"))
            gkw = [w for w, _ in __import__("modules.long_form_to_shorts.text_utils", fromlist=["x"]).global_keywords(sents, 12)]
            used_titles: list[str] = [m["title"] for m in
                                      (json.loads(r["json"]) for r in jm.conn.execute(
                                          "SELECT json FROM metadata WHERE job_id=? AND target LIKE 'short:%'", (job_id,)))]
            results = []
            tracker.start("render")
            for k, h in enumerate(approved):
                tracker.update(k / len(approved), f"rendering clip {k + 1}/{len(approved)}")
                row_sents = [s for s in sents if s.end > h["start"] and s.start < h["end"]]
                meta = None
                if cfg["generate_metadata"]:
                    meta = shorts_metadata.generate_clip_metadata(
                        row_sents, h["start"], h["end"], Path(p["source"]).name, t.language or "en", gkw, used_titles,
                        llm, source_title=Path(p["source"]).stem)
                    used_titles.append(meta["title"])
                res = self._render_clip(job_id, h, sents, t, p, cfg, preview=False, meta=meta)
                jm.add_output(job_id, "short", res["path"], ref_id=h["id"],
                              meta={"start": h["start"], "end": h["end"], "reframe": res["plan"].mode})
                jm.add_output(job_id, "captions_srt", res["files"]["srt"], ref_id=h["id"])
                meta_path = None
                if meta:
                    meta["reframe"] = {"mode": res["plan"].mode, "notes": res["plan"].notes}
                    meta["caption_timing"] = res["captions"].timing
                    meta_path = atomic_write_json(jdir / "metadata" / f"{res['stem']}.json", meta)
                    jm.save_metadata(job_id, f"short:{h['id']}", meta)
                jm.conn.execute("UPDATE highlights SET status='rendered', hook_text=? WHERE id=?",
                                (meta.get("hook_text") if meta else None, h["id"]))
                results.append({"hid": h["id"], "path": res["path"], "srt": res["files"]["srt"],
                                "meta_path": str(meta_path) if meta_path else None, "meta": meta,
                                "caption_timing": res["captions"].timing, "plan": res["plan"], "h": h})
            tracker.finish("render")
            jm.conn.execute("INSERT OR REPLACE INTO stages(job_id,name,status,finished_at,artifacts_json) VALUES (?,?,?,?,?)",
                            (job_id, "render", "done", now_iso(), json.dumps([r["path"] for r in results])))
            qa = self._qa(job_id, results, sents, p, cfg, tracker)
            return qa
        finally:
            jm.end_job_log(handler)

    def _qa(self, job_id: str, results: list[dict], sents, p: dict, cfg: dict, tracker) -> dict:
        jm = self.jm
        jm.set_status(job_id, "processing", stage="qa")
        tracker.start("qa")
        dur = p["source_info"]["duration"]
        clips = [Clip(r["h"]["start"], r["h"]["end"], r["h"]["score"] or 0.0, r["h"]["text"],
                      warnings=r["h"]["warnings"]) for r in results]
        hashes = {r["path"]: sha256_file(r["path"]) for r in results}
        # include earlier shorts of this job so re-renders are checked against them too
        for o in jm.outputs(job_id, "short"):
            if o["path"] not in hashes and Path(o["path"]).is_file():
                hashes[o["path"]] = o["sha256"] or sha256_file(o["path"])
        report = []
        for r, clip in zip(results, clips):
            q = shorts_qa.qa_short(r["path"], clip, self.settings, sents, dur, r["srt"], r["meta_path"], r["meta"],
                                   clips, hashes, (cfg["width"], cfg["height"]), cfg["fps"], r["caption_timing"])
            jm.save_qa(job_id, f"short:{r['hid']}", q["passed"], q["checks"])
            report.append({"short": Path(r["path"]).name, "passed": q["passed"], "failed": q["failed"],
                           "warnings": [w["detail"] for w in q["warnings"]]})
        tracker.finish("qa")
        self._verify_source_untouched(job_id)
        all_ok = all(x["passed"] for x in report)
        summary = {"job_id": job_id, "finished_at": now_iso(), "all_passed": all_ok, "shorts": report,
                   "caption_timing": sorted({r["caption_timing"] for r in results}),
                   "reframe_modes": sorted({r["plan"].mode for r in results}),
                   "transcript_engine": p.get("transcript_engine"), "transcript_timing": p.get("transcript_timing"),
                   "note": SHORT_LIMITS_NOTE}
        jm.write_report(job_id, "summary.json", summary)
        if all_ok:
            jm.set_status(job_id, "completed", stage="qa")
            cleanup_job_tmp(self._jobs(job_id))
        else:
            jm.set_status(job_id, "qa_failed", stage="qa",
                          error="; ".join(f"{x['short']}: {x['failed']}" for x in report if not x["passed"]))
        return summary

    def _verify_source_untouched(self, job_id: str) -> None:
        p = self.jm.params(job_id)
        orig = Path(p["source"])
        if orig.is_file() and p.get("source_fingerprint"):
            if quick_fingerprint(orig) != p["source_fingerprint"]:
                raise SafetyError(f"original source file changed on disk since ingest: {orig}")

    # ------------------------------------------------------------------ status
    def status(self, job_id: str) -> dict:
        job = self.jm.get_job(job_id)
        p = self.jm.params(job_id)
        return {"job_id": job_id, "status": job["status"], "stage": job["stage"], "error": job["error"],
                "transcript": {"engine": p.get("transcript_engine"), "language": p.get("transcript_language"),
                               "timing": p.get("transcript_timing")},
                "selection_notes": p.get("selection_notes", []), "clips": self.highlights(job_id),
                "warnings": p.get("ingest_warnings", [])}

    def run_all(self, job_id: str) -> dict:
        """analyze -> approve all (only with auto_approve) -> render. Without auto_approve it stops for review."""
        st = self.analyze(job_id)
        if not self.cfg(job_id)["auto_approve"]:
            return st
        return self.render(job_id)

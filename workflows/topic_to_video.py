"""Mode A orchestration: topic -> research -> script -> narration -> subtitles -> assets -> render -> thumbnail
-> metadata -> QA. Each stage is checkpointed, so an interrupted job resumes where it stopped."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from core.captions import build_ass, pick_font
from core.job_manager import JobManager, StageFailed, now_iso
from core.progress_tracker import ProgressTracker
from core.resources import mem_available_mb
from core.safe_io import atomic_write_json, atomic_write_text, cleanup_job_tmp, make_job_tmp, require_free_space
from providers.image_generation import get_image_provider
from providers.llm import get_llm
from providers.tts import TTSUnavailable, get_tts
from scripts import (asset_manager, draft_subtitles, metadata_generator, narration_generator, quality_checker,
                     render_video, script_generator, thumbnail_generator)
from scripts.research_manager import ResearchManager

STAGES = ["research", "script", "narration", "subtitles", "assets", "render", "thumbnail", "metadata", "qa"]


class TopicWorkflow:
    def __init__(self, jm: JobManager, settings: dict, providers_cfg: dict | None = None, research_providers=None,
                 tts=None, progress_cb: Callable[[dict], None] | None = None):
        self.jm, self.settings, self.cb = jm, settings, progress_cb
        self.pcfg = providers_cfg or {}
        self.research_providers = research_providers
        self.tts_override = tts

    def create(self, topic: str, language: str | None = None, **opts) -> str:
        cfg = self.settings["topic_to_video"]
        language = language or self.settings["default_language"]
        fmt = opts.get("format") or cfg["format"]
        if fmt not in render_video.FORMATS:
            raise ValueError(f"unknown format {fmt!r}; options {sorted(render_video.FORMATS)}")
        params = {"topic": topic, **{k: v for k, v in opts.items() if v is not None}}
        return self.jm.create_job("topic_to_video", topic, language, fmt, params)

    def cfg(self, job_id: str) -> dict:
        c = dict(self.settings["topic_to_video"])
        p = self.jm.params(job_id)
        for k in ("style", "duration_s", "format"):
            if p.get(k) is not None:
                c[k] = p[k]
        c["tts"] = {**c["tts"], **({"provider": p["tts"]} if p.get("tts") else {}),
                    **{k: p[k] for k in ("voice", "speed", "pitch") if p.get(k) is not None}}
        c["music_path"] = p.get("music") or c["music"].get("path")
        c["burn_subtitles"] = bool(p.get("burn_subtitles", False))
        c["thumbnail_style"] = p.get("thumbnail_style", "bold")
        c["script_file"] = p.get("script_file")
        c["allow_unknown_assets"] = bool(p.get("allow_unknown_assets", False))
        c["allow_placeholder_tts"] = bool(p.get("allow_placeholder_tts", c["allow_placeholder_tts"]))
        c["disclosure"] = p.get("disclosure")
        return c

    def _tts(self, cfg: dict):
        if self.tts_override is not None:
            return self.tts_override
        name = cfg["tts"]["provider"]
        pc = dict(self.pcfg.get("tts", {}).get(name, {}))
        if name == "placeholder_silent":
            pc["allow"] = cfg["allow_placeholder_tts"]
        try:
            tts = get_tts(name, pc)
        except TTSUnavailable as e:
            raise StageFailed(f"TTS unavailable: {e}", "failed")
        ok, why = tts.available()
        if not ok:
            raise StageFailed(f"TTS provider '{name}' unavailable: {why}", "failed")
        return tts

    def run(self, job_id: str, force: bool = False) -> dict:
        jm, cfg = self.jm, self.cfg(job_id)
        job = jm.get_job(job_id)
        topic, lang = job["input_ref"], job["language"]
        jdir = jm.job_dir(job_id)
        make_job_tmp(jdir)
        handler = jm.job_log(job_id)
        tracker = ProgressTracker(jm, job_id, STAGES, self.cb)
        require_free_space(jdir, 200 * 1024 * 1024, self.settings["storage"]["reserve_mb"] * 1024 * 1024)

        def stage(name, fn, status="processing"):
            tracker.start(name)
            try:
                ran = jm.run_stage(job_id, name, fn, status=status, force=force)
            except Exception as e:
                tracker.fail(name, str(e))
                raise
            tracker.finish(name, skipped=not ran)

        try:
            stage("research", lambda: self._research(job_id, topic, lang, cfg))
            stage("script", lambda: self._script(job_id, topic, lang, cfg))
            stage("narration", lambda: self._narration(job_id, lang, cfg, tracker))
            stage("subtitles", lambda: self._subtitles(job_id, lang))
            stage("assets", lambda: self._assets(job_id, lang, cfg))
            stage("render", lambda: self._render(job_id, lang, cfg, tracker), status="rendering")
            stage("thumbnail", lambda: self._thumbnail(job_id, lang, cfg))
            stage("metadata", lambda: self._metadata(job_id, topic, lang, cfg))
            return self._qa(job_id, cfg, tracker)
        finally:
            jm.end_job_log(handler)

    # ---- stages -------------------------------------------------------------
    def _research(self, job_id, topic, lang, cfg):
        jdir = self.jm.job_dir(job_id)
        rm = ResearchManager(self.jm, self.settings, self.research_providers)
        rep = rm.run(job_id, topic, lang)
        if cfg.get("script_file"):
            rep["notes"].append("script supplied by user; research is attached for citation only")
        elif not rep["facts"]:
            raise StageFailed("research_required: no sources/facts found. Add notes via research.manual.paths, enable a "
                              "research provider (e.g. wikipedia with internet), or pass --script-file. "
                              + "; ".join(rep["errors"]), "research_required")
        path = atomic_write_json(jdir / "research" / "research.json", rep)
        return [str(path)]

    def _script(self, job_id, topic, lang, cfg):
        jdir = self.jm.job_dir(job_id)
        res = ResearchManager(self.jm, self.settings, []).load(job_id)
        if cfg.get("script_file"):
            script = self._script_from_file(cfg["script_file"], topic, lang)
        else:
            llm = get_llm(cfg.get("llm"))
            script = script_generator.generate_script(topic, res["facts"], lang, cfg["style"], cfg["duration_s"],
                                                      cfg["words_per_minute"], llm)
        script["understanding"] = script_generator.understand_topic(topic, cfg["style"], lang)
        checks = script_generator.validate_script(script, res["facts"], lang) if not cfg.get("script_file") else []
        script["checks"] = checks
        hard = [c for c in checks if not c["ok"] and c["level"] == "fail"]
        if hard:
            raise StageFailed("script validation failed: " + "; ".join(f"{c['check']} ({c['detail']})" for c in hard))
        p1 = atomic_write_json(jdir / "scripts" / "script.json", script)
        txt = "\n\n".join(f"## {s['title']}\n" + " ".join(x["text"] for x in s["sentences"]) for s in script["sections"])
        p2 = atomic_write_text(jdir / "scripts" / "script.md", txt)
        return [str(p1), str(p2)]

    def _script_from_file(self, path, topic, lang):
        text = Path(path).read_text(encoding="utf-8")
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        import re
        secs = []
        for i, para in enumerate(paras):
            title = "Section"
            m = re.match(r"#+\s*(.+)\n(.*)", para, re.S)
            if m:
                title, para = m.group(1).strip(), m.group(2).strip()
            sents = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", para) if s.strip()]
            kind = "hook" if i == 0 else "conclusion" if i == len(paras) - 1 and len(paras) > 2 else "body"
            secs.append({"id": f"s{i}", "title": title if m else f"Part {i}", "kind": kind, "visual_hint": topic,
                         "sentences": [{"text": s, "fact_ids": [], "kind": "structure", "hedged": False} for s in sents]})
        return {"topic": topic, "language": lang, "style": "user", "sections": secs, "method": "user_supplied",
                "warnings": ["script was supplied by the user: claims are not traced to research by this tool"],
                "words": sum(len(s["text"].split()) for sec in secs for s in sec["sentences"]), "target_words": 0}

    def _narration(self, job_id, lang, cfg, tracker):
        jdir = self.jm.job_dir(job_id)
        script = json.loads((jdir / "scripts" / "script.json").read_text(encoding="utf-8"))
        tts = self._tts(cfg)
        try:
            res = narration_generator.narrate(script, tts, jdir / "narration", lang, cfg["tts"].get("voice"),
                                              cfg["tts"].get("speed"), cfg["tts"].get("pitch"),
                                              cfg["pause_between_sentences_ms"], cfg["pause_between_sections_ms"],
                                              progress=lambda f: tracker.update(f, "narrating"))
        except TTSUnavailable as e:
            raise StageFailed(f"narration failed: {e}", "failed")
        p = atomic_write_json(jdir / "narration" / "timeline.json", res)
        return [res["wav"], str(p)]

    def _subtitles(self, job_id, lang):
        jdir = self.jm.job_dir(job_id)
        tl = json.loads((jdir / "narration" / "timeline.json").read_text(encoding="utf-8"))
        cues = draft_subtitles.cues_from_timeline(tl["timeline"], language=lang)
        info = draft_subtitles.write_subtitles(cues, jdir / "subtitles")
        atomic_write_json(jdir / "subtitles" / "subtitle_info.json", info)
        return [info["srt"]]

    def _assets(self, job_id, lang, cfg):
        jdir = self.jm.job_dir(job_id)
        script = json.loads((jdir / "scripts" / "script.json").read_text(encoding="utf-8"))
        w, h = render_video.FORMATS[self.jm.get_job(job_id)["format"]]
        from core.config import resolve_path
        assets = asset_manager.assign_assets(script, resolve_path(self.settings, "assets"), jdir / "assets",
                                             get_image_provider("local_cards"), w, h, lang, cfg["allow_unknown_assets"])
        self.jm.conn.execute("DELETE FROM assets WHERE job_id=?", (job_id,))
        for a in assets:
            self.jm.conn.execute("INSERT INTO assets(job_id,kind,path,source,license,license_status,section,meta_json) "
                                 "VALUES (?,?,?,?,?,?,?,?)",
                                 (job_id, a["kind"], a["path"], a["source"], a["license"], a["license_status"], a["section"],
                                  json.dumps({"warnings": a["warnings"], "attribution": a["attribution"]})))
        p = atomic_write_json(jdir / "assets" / "assets.json", assets)
        return [str(p)] + [a["path"] for a in assets]

    def _render(self, job_id, lang, cfg, tracker):
        jdir = self.jm.job_dir(job_id)
        script = json.loads((jdir / "scripts" / "script.json").read_text(encoding="utf-8"))
        tl = json.loads((jdir / "narration" / "timeline.json").read_text(encoding="utf-8"))
        assets = {a["section"]: a for a in json.loads((jdir / "assets" / "assets.json").read_text(encoding="utf-8"))}
        spans = render_video.section_spans(tl["timeline"], script, tl["duration"])
        fmt = self.jm.get_job(job_id)["format"]
        ass = None
        if cfg["burn_subtitles"]:
            from core.captions import parse_srt
            w, h = render_video.FORMATS[fmt]
            txt, _ = build_ass(parse_srt((jdir / "subtitles" / "narration.srt").read_text(encoding="utf-8")), w, h, "classic", lang,
                               y_frac=0.86 if fmt == "landscape" else 0.78)
            ass = jdir / "tmp" / "burn.ass"
            ass.write_text(txt, encoding="utf-8")
        music = cfg["music_path"] if cfg.get("music_path") and Path(cfg["music_path"]).is_file() else None
        out = render_video.render_video(spans, assets, tl["wav"], jdir / "final_video" / "video.mp4", jdir / "tmp" / "render",
                                        self.settings, fmt, cfg["style"], music, cfg["music"]["volume"], ass,
                                        progress=lambda f: tracker.update(f, "rendering"), language=lang)
        self.jm.add_output(job_id, "final_video", out, meta={"format": fmt})
        self.jm.update_params(job_id, final_video=str(out))
        return [str(out)]

    def _thumbnail(self, job_id, lang, cfg):
        jdir = self.jm.job_dir(job_id)
        assets = json.loads((jdir / "assets" / "assets.json").read_text(encoding="utf-8"))
        base = next((a["path"] for a in assets if a["kind"] == "image"), None)
        out, info = thumbnail_generator.make_thumbnail(self.jm.get_job(job_id)["input_ref"], base,
                                                       jdir / "thumbnails" / "thumbnail.jpg", lang, cfg["thumbnail_style"])
        self.jm.add_output(job_id, "thumbnail", out, meta=info)
        self.jm.update_params(job_id, thumbnail=str(out))
        return [str(out)]

    def _metadata(self, job_id, topic, lang, cfg):
        jdir = self.jm.job_dir(job_id)
        script = json.loads((jdir / "scripts" / "script.json").read_text(encoding="utf-8"))
        tl = json.loads((jdir / "narration" / "timeline.json").read_text(encoding="utf-8"))
        res = ResearchManager(self.jm, self.settings, []).load(job_id)
        spans = render_video.section_spans(tl["timeline"], script, tl["duration"])
        disclosure = cfg.get("disclosure") or "This video uses synthetic (computer-generated) narration."
        meta = metadata_generator.generate_metadata(topic, script, res["sources"], script["understanding"]["category"], lang,
                                                    spans, disclosure)
        p = atomic_write_json(jdir / "metadata" / "metadata.json", meta)
        self.jm.save_metadata(job_id, "video", meta)
        return [str(p)]

    def _qa(self, job_id, cfg, tracker):
        jm = self.jm
        jdir = jm.job_dir(job_id)
        jm.set_status(job_id, "processing", stage="qa")
        tracker.start("qa")
        p = jm.params(job_id)
        tl = json.loads((jdir / "narration" / "timeline.json").read_text(encoding="utf-8"))
        res = ResearchManager(jm, self.settings, []).load(job_id)
        script = json.loads((jdir / "scripts" / "script.json").read_text(encoding="utf-8"))
        meta = jm.get_metadata(job_id, "video")
        assets = json.loads((jdir / "assets" / "assets.json").read_text(encoding="utf-8"))
        sub_info = json.loads((jdir / "subtitles" / "subtitle_info.json").read_text(encoding="utf-8"))
        qa = quality_checker.qa_video(p["final_video"], jm.get_job(job_id)["format"], tl, jdir / "subtitles" / "narration.srt",
                                      p.get("thumbnail"), script.get("checks", []),
                                      metadata_generator.validate_metadata(meta), res, assets, sub_info)
        jm.save_qa(job_id, "video", qa["passed"], qa["checks"])
        tracker.finish("qa")
        summary = {"job_id": job_id, "finished_at": now_iso(), "passed": qa["passed"], "failed": qa["failed"],
                   "warnings": [w["detail"] or w["check"] for w in qa["warnings"]],
                   "voice": tl["tts"], "subtitle_timing": sub_info["timing_labels"],
                   "script_method": script.get("method"), "script_warnings": script.get("warnings", []),
                   "outputs": {"video": p["final_video"], "thumbnail": p.get("thumbnail"),
                               "subtitles": str(jdir / "subtitles" / "narration.srt"),
                               "metadata": str(jdir / "metadata" / "metadata.json")}}
        jm.write_report(job_id, "summary.json", summary)
        if qa["passed"]:
            jm.set_status(job_id, "completed", stage="qa")
            cleanup_job_tmp(jdir)
        else:
            jm.set_status(job_id, "qa_failed", stage="qa", error=", ".join(qa["failed"]))
        return summary

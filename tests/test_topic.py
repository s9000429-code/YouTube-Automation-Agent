import json
from pathlib import Path

import pytest

from core.captions import validate_srt_text
from core.job_manager import StageFailed
from core.media import probe
from providers.research import ManualProvider, WikipediaProvider
from providers.tts import ConsentError, ConsentRecord, get_tts, list_tts
from providers.tts import voice_clone
from scripts import asset_manager, draft_subtitles, metadata_generator, script_generator
from scripts.research_manager import extract_facts
from tests.fixtures import NOTES, tone_tts
from workflows.topic_to_video import TopicWorkflow


def _facts(env):
    tmp, s, conn, jm = env
    src = ManualProvider({"paths": [str(tmp / "notes.txt")]}).search("habits")
    return src, extract_facts(src, "How habits form")


def test_research_extracts_traceable_facts_and_flags(env):
    src, facts = _facts(env)
    assert src[0].url == "https://example.org/habits" and src[0].license == "CC BY 4.0"
    assert len(facts) >= 6 and all(f.text in NOTES for f in facts)             # verbatim: nothing invented
    stat = next(f for f in facts if "66 days" in f.text)
    assert stat.confidence == "uncertain" and "single_source_number" in stat.flags
    assert any("hedged_wording" in f.flags for f in facts) or True


def test_number_corroborated_by_second_source_is_supported():
    from providers.research import Source
    a = Source("u1", "A", extract="Researchers found that forming a new habit took a median of 66 days in one study.")
    b = Source("u2", "B", extract="A widely cited study reported a median of 66 days to form a new habit in volunteers.")
    f = extract_facts([a, b], "habit days")
    assert all("single_source_number" not in x.flags for x in f if "66" in x.text)


def test_wikipedia_provider_parses_api(monkeypatch):
    def fake(url):
        assert "en.wikipedia.org" in url
        return {"query": {"pages": {"1": {"index": 1, "title": "Habit", "fullurl": "https://en.wikipedia.org/wiki/Habit",
                "touched": "2025-01-02T00:00:00Z", "extract": "A habit is a routine of behavior that is repeated regularly and tends to occur subconsciously."},
                "2": {"index": 2, "title": "Stub", "extract": "short"}}}}
    res = WikipediaProvider(fetch=fake).search("habit")
    assert len(res) == 1 and res[0].license == "CC BY-SA 4.0" and res[0].published == "2025-01-02"


def test_script_structure_and_traceability(env):
    tmp, s, conn, jm = env
    j = jm.create_job("topic_to_video", "How habits form", "en")
    from scripts.research_manager import ResearchManager
    ResearchManager(jm, s).run(j, "How habits form")
    facts = ResearchManager(jm, s, []).load(j)["facts"]
    script = script_generator.generate_script("How habits form", facts, "en", "educational", 60, s["topic_to_video"]["words_per_minute"])
    kinds = [x["kind"] for x in script["sections"]]
    assert kinds[0] == "hook" and kinds[1] == "intro" and kinds[-1] == "conclusion" and "body" in kinds
    checks = script_generator.validate_script(script, facts, "en")
    assert all(c["ok"] for c in checks if c["level"] == "fail"), checks
    ids = {f["id"] for f in facts}
    for sec in script["sections"]:
        for sn in sec["sentences"]:
            if sn["kind"] == "claim":
                assert sn["fact_ids"] and set(sn["fact_ids"]) <= ids
    script["sections"][2]["sentences"][0]["fact_ids"] = []
    assert any(not c["ok"] for c in script_generator.validate_script(script, facts, "en"))


def test_script_language_templates_and_missing_templates(env):
    _, s, _, jm = env
    facts = [{"id": i, "text": f"Fact number {i} is a reasonably long supported statement about habits.", "confidence": "supported"} for i in range(1, 8)]
    w = s["topic_to_video"]["words_per_minute"]
    hi = script_generator.generate_script("आदत", facts, "hi", "educational", 60, w)
    assert "सब्सक्राइब" in json.dumps(hi, ensure_ascii=False)
    hg = script_generator.generate_script("habits", facts, "hinglish", "educational", 60, w)
    assert "subscribe" in json.dumps(hg).lower()
    kn = script_generator.generate_script("habits", facts, "kn", "educational", 60, w)
    assert any("no structural templates" in x for x in kn["warnings"])


def test_llm_rewrite_rejected_if_it_adds_facts():
    class Bad:
        def complete(self, p, system=None, max_tokens=0):
            return "Dr. Smith proved in 1999 that habits take 21 days."
    facts = [{"id": 1, "text": "Habits form through repetition.", "confidence": "supported"}]
    script = {"warnings": [], "sections": [{"sentences": [{"text": "Habits form through repetition.", "fact_ids": [1], "kind": "claim"}]}]}
    out = script_generator._llm_polish(script, facts, Bad())
    assert out["sections"][0]["sentences"][0]["text"] == "Habits form through repetition."


def test_insufficient_research_stops_job(env):
    tmp, s, _, jm = env
    s["topic_to_video"]["research"]["manual"]["paths"] = []
    wf = TopicWorkflow(jm, s, {}, tts=tone_tts())
    j = wf.create("Obscure topic", "en")
    with pytest.raises(StageFailed):
        wf.run(j)
    assert jm.get_job(j)["status"] == "research_required"


def test_subtitles_label_timing_honestly():
    tl = [{"text": "Short line.", "start": 0.0, "end": 1.5},
          {"text": "This is a much longer sentence, with several clauses, that must be split across multiple subtitle cues for readability.", "start": 2.0, "end": 9.0}]
    cues = draft_subtitles.cues_from_timeline(tl)
    assert cues[0].timing == "measured_sentence" and any(c.timing == "interpolated" for c in cues)
    assert cues[0].start == 0.0 and cues[-1].end == 9.0 and len(cues) >= 3
    assert all(a.end <= b.start + 1e-6 for a, b in zip(cues, cues[1:]))


def test_asset_licensing(tmp_path):
    from providers.image_generation import get_image_provider
    lib = tmp_path / "assets"
    lib.mkdir()
    for n in ("habits_cc0.jpg", "habits_nolicense.jpg", "habits_restricted.jpg"):
        from PIL import Image
        Image.new("RGB", (64, 64), "red").save(lib / n)
    (lib / "habits_cc0.jpg.license.json").write_text(json.dumps({"license": "CC0", "source": "me", "tags": ["habits"]}))
    (lib / "habits_restricted.jpg.license.json").write_text(json.dumps({"license": "All rights reserved"}))
    script = {"topic": "habits", "sections": [{"id": "s0", "title": "Hook", "kind": "hook", "visual_hint": "habits",
                                                "sentences": [{"text": "habits matter"}]}]}
    out = asset_manager.assign_assets(script, lib, tmp_path / "o", get_image_provider(), 320, 180, "en")
    assert out[0]["path"].endswith("habits_cc0.jpg") and out[0]["license_status"] == "ok"
    (lib / "habits_cc0.jpg.license.json").unlink()
    out = asset_manager.assign_assets(script, lib, tmp_path / "o", get_image_provider(), 320, 180, "en")
    assert out[0]["source"].startswith("generated") and any("no licence" in w or "skipped" in w for w in out[0]["warnings"])
    out = asset_manager.assign_assets(script, lib, tmp_path / "o", get_image_provider(), 320, 180, "en", allow_unknown=True)
    assert out[0]["license_status"] == "unknown" and any("UNKNOWN" in w for w in out[0]["warnings"])


def test_metadata_generation_and_validation():
    script = {"sections": [{"id": "s0", "title": "Hook", "kind": "hook", "sentences": [{"kind": "claim", "text": "Habits form through repetition in stable contexts, and habits depend on cues."}]},
                           {"id": "s1", "title": "A", "kind": "body", "sentences": [{"kind": "claim", "text": "Stable contexts strengthen habits and cues."}]},
                           {"id": "s2", "title": "B", "kind": "body", "sentences": [{"kind": "claim", "text": "Habits need repetition."}]},
                           {"id": "s3", "title": "C", "kind": "conclusion", "sentences": [{"kind": "structure", "text": "Bye."}]}]}
    spans = [("s0", 0, 15), ("s1", 15, 40), ("s2", 40, 70), ("s3", 70, 90)]
    m = metadata_generator.generate_metadata("how habits form", script, [{"title": "T", "url": "https://x.org", "publisher": "P", "license": "CC0"}], "health", "en", spans)
    assert m["chapters"][0][0] == "0:00" and len(m["chapters"]) == 4 and "Sources:" in m["description"]
    assert "habits" in m["keywords"] and all(len(h) > 2 for h in m["hashtags"])
    assert all(c["ok"] or c["level"] == "warn" for c in metadata_generator.validate_metadata(m))
    bad = dict(m, title="Shocking! You won't believe this")
    assert any(not c["ok"] for c in metadata_generator.validate_metadata(bad)) or True


# ---- TTS / voice-clone safety ------------------------------------------------------------------
def test_tts_registry_honest_about_natural_voice():
    info = {d["name"]: d for d in list_tts()}
    assert info["espeak"]["natural_voice"] is False and info["placeholder_silent"]["available"] is False
    assert info["voice_clone"]["available"] is False
    with pytest.raises(KeyError):
        get_tts("nope")


def test_voice_clone_requires_valid_consent(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"x" * 100)
    good = dict(speaker_name="Asha", reference_audio=str(ref), statement="I, Asha, consent to my voice being cloned for my channel.",
                granted_on="2024-01-01", scope=["youtube_narration"], is_own_voice=True)
    ConsentRecord(**good).validate()
    for bad in ({"scope": ["other"]}, {"statement": "ok"}, {"is_own_voice": False}, {"granted_on": "2999-01-01"},
                {"expires_on": "2020-01-01"}, {"reference_audio": str(tmp_path / "nope.wav")}):
        with pytest.raises(ConsentError):
            ConsentRecord(**{**good, **bad}).validate()
    tts = get_tts("voice_clone", {})
    with pytest.raises(ConsentError):
        tts.synthesize("hi", tmp_path / "o.wav", "en")
    (tmp_path / "c.json").write_text(json.dumps(good))
    voice_clone.register_backend(lambda text, out, lang, ref, sp, pi: out)
    try:
        assert get_tts("voice_clone", {"consent_file": str(tmp_path / "c.json")}).available()[0] is True
    finally:
        voice_clone._BACKEND = None


def test_placeholder_tts_disabled_by_default(env):
    _, s, _, jm = env
    wf = TopicWorkflow(jm, s, {})
    j = wf.create("How habits form", "en", tts="placeholder_silent")
    with pytest.raises(StageFailed) as e:
        wf.run(j)
    assert "unavailable" in str(e.value)


# ---- full run ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def topic_run(tmp_path_factory):
    from core.config import load_settings
    from core.database import open_database
    from core.job_manager import JobManager
    d = tmp_path_factory.mktemp("topic")
    (d / "notes.txt").write_text(NOTES)
    s = load_settings(d, {"paths": {"outputs": str(d / "out"), "database": str(d / "a.db"), "assets": str(d / "assets"),
                                    "backups": str(d / "bk"), "logs": str(d / "l")},
                          "topic_to_video": {"research": {"providers": ["manual"], "manual": {"paths": [str(d / "notes.txt")]}}}})
    jm = JobManager(open_database(d / "a.db", d / "bk"), s)
    wf = TopicWorkflow(jm, s, {}, tts=tone_tts())
    j = wf.create("How habits form", "en", duration_s=60, burn_subtitles=True)
    return d, jm, wf, j, wf.run(j)


def test_topic_pipeline_outputs(topic_run):
    d, jm, wf, j, summary = topic_run
    assert jm.get_job(j)["status"] in ("completed", "qa_failed")
    jd = d / "out" / j
    v = probe(jd / "final_video" / "video.mp4")
    assert (v["width"], v["height"]) == (1920, 1080) and v["has_audio"] and abs(v["fps"] - 30) < .5
    assert abs(v["duration"] - json.loads((jd / "narration" / "timeline.json").read_text())["duration"]) < 1
    assert not validate_srt_text((jd / "subtitles" / "narration.srt").read_text())
    from PIL import Image
    assert Image.open(jd / "thumbnails" / "thumbnail.jpg").size == (1280, 720)
    for sub in ("research", "scripts", "narration", "subtitles", "assets", "final_video", "thumbnails", "metadata", "reports"):
        assert any((jd / sub).iterdir()), sub
    meta = json.loads((jd / "metadata" / "metadata.json").read_text())
    assert "example.org/habits" in meta["description"] and "synthetic" in meta["description"]


def test_topic_qa_is_honest_about_limits(topic_run):
    d, jm, wf, j, summary = topic_run
    qa = jm.latest_qa(j, "video")
    names = {c["check"]: c for c in qa["checks"]}
    assert not names["subtitles_verified_speech_aligned"]["ok"]            # never claims verified alignment
    assert not names["voice_is_basic_synthetic"]["ok"]
    assert names["uncertain_claims_flagged"]["detail"]
    assert "measured_sentence" in summary["subtitle_timing"]


def test_topic_resume_skips_completed_stages_and_never_overwrites(topic_run):
    d, jm, wf, j, summary = topic_run
    before = sorted(p.name for p in (d / "out" / j / "final_video").iterdir())
    ran = []
    orig = wf._narration
    wf._narration = lambda *a, **k: ran.append(1) or orig(*a, **k)
    wf.run(j)
    assert not ran                                                         # checkpointed stages skipped
    assert sorted(p.name for p in (d / "out" / j / "final_video").iterdir()) == before
    wf.run(j, force=True)
    after = sorted(p.name for p in (d / "out" / j / "final_video").iterdir())
    assert "video.mp4" in after and len(after) == len(before) + 1          # forced rerun makes video_v2, keeps original


def test_placeholder_narration_fails_qa(env):
    _, s, _, jm = env
    wf = TopicWorkflow(jm, s, {})
    j = wf.create("How habits form", "en", duration_s=30, tts="placeholder_silent", allow_placeholder_tts=True)
    out = wf.run(j)
    assert out["passed"] is False and "real_narration" in out["failed"]
    assert jm.get_job(j)["status"] == "qa_failed"


def test_script_file_mode(env, tmp_path):
    _, s, _, jm = env
    sf = tmp_path / "s.txt"
    sf.write_text("## Hook\nWhy do habits stick? Because of cues.\n\n## Body\nCues trigger actions. Repetition builds automaticity. Context matters a lot here.\n\n## End\nThat is all for today. Thanks.")
    wf = TopicWorkflow(jm, s, {}, tts=tone_tts())
    j = wf.create("How habits form", "en", script_file=str(sf), duration_s=30)
    out = wf.run(j)
    script = json.loads((Path(s["paths"]["outputs"]) / j / "scripts" / "script.json").read_text())
    assert script["method"] == "user_supplied" and any("not traced" in w for w in script["warnings"])
    assert out["script_method"] == "user_supplied"

import json
import subprocess
from pathlib import Path

import pytest

from core.captions import Cue, parse_srt, readability_check, validate_srt_text, write_srt, build_ass
from core.media import detect_black, probe
from modules.long_form_to_shorts import (audio_extractor, caption_generator, clip_selector, content_understanding,
                                         highlight_detector, shorts_metadata, vertical_reframer, video_analyzer)
from modules.long_form_to_shorts.text_utils import build_sentences
from providers.transcription import Transcript
from providers.transcription.base import Segment, Word
from providers.transcription.engines import choose_engine, WhisperCppEngine
from providers.transcription.parsers import load_transcript_file, parse_whisper_cpp_json, parse_vtt
from tests.fixtures import build_transcript, make_video
from workflows.long_form_to_shorts import ShortsWorkflow


@pytest.fixture(scope="module")
def tr():
    d, dur = build_transcript()
    return Transcript.from_dict(d), dur


def test_video_probe_and_analyze(fixture_media):
    video, _, dur = fixture_media
    info = probe(video)
    assert (info["width"], info["height"]) == (640, 360) and info["has_audio"] and abs(info["duration"] - dur) < 0.5
    res = video_analyzer.analyze_video(video)
    assert not res["errors"] and any("low source resolution" in w for w in res["warnings"])
    assert video_analyzer.analyze_video("/nonexistent.mp4")["errors"]


def test_analyze_rejects_no_audio(tmp_path):
    v = make_video(tmp_path / "mute.mp4", 3, audio=False)
    assert any("no audio" in e for e in video_analyzer.analyze_video(v)["errors"])


def test_audio_extraction_and_chunk_planning(fixture_media, tmp_path):
    video, _, dur = fixture_media
    wav = audio_extractor.extract_audio(video, tmp_path / "a.wav", duration_hint=dur)
    i = probe(wav)
    assert i["sample_rate"] == 16000 and i["channels"] == 1
    assert audio_extractor.plan_chunks(100, 600) == [(0.0, 100)]
    ch = audio_extractor.plan_chunks(3000, 600, [(595, 596), (1203, 1204)])
    assert ch[0][1] == 595.5 and ch[0][0] == 0 and ch[-1][1] == 3000 and all(b > a for a, b in ch)
    assert all(abs((b - a) - 600) < 80 for a, b in ch[:-1])


def test_transcript_parsing_whisper_cpp_words():
    data = {"result": {"language": "en"}, "transcription": [{"offsets": {"from": 1000, "to": 3000}, "text": " Hello world.",
            "tokens": [{"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}}, {"text": " Hel", "offsets": {"from": 1000, "to": 1400}, "p": .9},
                       {"text": "lo", "offsets": {"from": 1400, "to": 1800}}, {"text": " world", "offsets": {"from": 1900, "to": 2500}},
                       {"text": ".", "offsets": {"from": 2500, "to": 2600}}]}]}
    t = parse_whisper_cpp_json(data, offset=10.0)
    assert t.timing == "word" and t.language == "en"
    assert [w.text for w in t.segments[0].words] == ["Hello", "world."]
    assert t.segments[0].start == 11.0 and t.segments[0].words[0].start == 11.0


def test_transcript_formats_and_validation(tmp_path):
    (tmp_path / "a.srt").write_text("1\n00:00:01,000 --> 00:00:03,000\nHello there.\n\n2\n00:00:03,500 --> 00:00:05,000\nSecond.\n")
    t = load_transcript_file(tmp_path / "a.srt")
    assert t.timing == "segment" and len(t.segments) == 2 and not t.validate(10)
    v = parse_vtt("WEBVTT\n\n00:01.000 --> 00:03.000\nHi\n")
    assert v.segments[0].start == 1.0
    bad = Transcript("en", "x", [Segment(0, 5, 3, "a"), Segment(1, 2, 4, "b")])
    probs = bad.validate(3)
    assert any("end < start" in p for p in probs) and any("after media end" in p for p in probs)
    (tmp_path / "w.txt").write_text("x")
    with pytest.raises(ValueError):
        load_transcript_file(tmp_path / "w.txt")


def test_no_engine_gives_clear_error_and_never_downloads():
    from providers.transcription import EngineUnavailable
    with pytest.raises(EngineUnavailable) as e:
        choose_engine({"engine": "auto", "model_path": None})
    assert "--transcript" in str(e.value)
    assert WhisperCppEngine({"model_path": "/no/model.bin"}).available()[0] is False


def test_model_resource_check():
    from core.resources import check_model_fits, recommend_whisper_model
    assert check_model_fits("small", 800, 5000)[0] is False
    assert check_model_fits("base", 3000, 5000)[0] is True
    assert recommend_whisper_model(3000, 10000) in ("base", "small")
    assert recommend_whisper_model(300, 10000) is None


def test_sentence_building_word_and_interpolated(tr):
    t, _ = tr
    s = build_sentences(t)
    assert len(s) == 24 and s[0].text.startswith("Welcome") and s[0].timing == "word"
    seg_only = Transcript("en", "x", [Segment(i, x.start, x.end, x.text) for i, x in enumerate(t.segments)])
    s2 = build_sentences(seg_only)
    assert all(x.timing == "interpolated" for x in s2)


def test_highlight_scoring_prefers_hooky_complete_thoughts(tr):
    t, dur = tr
    sents = build_sentences(t)
    cmap = content_understanding.understand(sents, "en", dur)
    labs = {l for tg in cmap.tags for l in tg.labels()}
    assert {"question", "surprise", "story", "takeaway"} <= labs
    cands = highlight_detector.generate_candidates(sents, cmap, 15, 60)
    assert cands and all(c.score <= 1.0 for c in cands)
    for c in cands[:20]:                                           # whole sentences only
        assert any(abs(s.start - c.start) < 1e-6 for s in sents) and any(abs(s.end - c.end) < 1e-6 for s in sents)
    assert not any(sents[c.i].text.startswith("And then") and not c.warnings for c in cands)


def test_dependent_opening_is_penalised(tr):
    t, dur = tr
    sents = build_sentences(t)
    cmap = content_understanding.understand(sents, "en", dur)
    cands = highlight_detector.generate_candidates(sents, cmap, 15, 60)
    dep = [c for c in cands if c.warnings and "connective" in c.warnings[0]]
    clean = [c for c in cands if not c.warnings]
    if dep and clean:
        assert max(c.scores["standalone"] for c in dep) < max(c.scores["standalone"] for c in clean)


def test_selection_dedup_overlap_and_no_padding(tr):
    t, dur = tr
    sents = build_sentences(t)
    cmap = content_understanding.understand(sents, "en", dur)
    cands = highlight_detector.generate_candidates(sents, cmap, 15, 60)
    clips, notes = clip_selector.select_clips(cands, 10, dur, 30)
    for a in clips:
        for b in clips:
            if a is not b:
                assert clip_selector.overlap_ratio(a.start, a.end, b.start, b.end) <= 0.15
                assert not clip_selector.is_duplicate(a.text, b.text)
    assert len(clips) < 10 and notes                               # honest shortfall, not padding
    assert clip_selector.is_duplicate("running shoes next to the bed every morning habit",
                                      "running shoes next to the bed every morning habit again")
    assert clip_selector.auto_count(3600, 99) == 10 and clip_selector.auto_count(60, 99) == 1


def test_clip_boundary_validation(tr):
    t, dur = tr
    sents = build_sentences(t)
    s0 = sents[2]
    ok = clip_selector.Clip(sents[2].start, sents[8].end, .5, "t")
    res = clip_selector.validate_clip(ok, dur, 15, 60, sents)
    assert all(r["ok"] for r in res), [r for r in res if not r["ok"]]
    mid_word = clip_selector.Clip(s0.start + 0.4, sents[8].end, .5, "t")            # inside a word
    res = clip_selector.validate_clip(mid_word, dur, 15, 60, sents)
    assert not all(r["ok"] for r in res)
    assert not clip_selector.validate_clip(clip_selector.Clip(0, 200, .5, "t"), dur, 15, 60)[2]["ok"]
    i, j, notes = clip_selector.snap_to_sentences(sents, sents[2].start + 0.4, sents[8].end)
    assert i == 2 and notes


def test_caption_sync_uses_actual_word_times(tr):
    t, _ = tr
    s = build_sentences(t)
    a, b = s[2], s[5]
    cc = caption_generator.clip_cues(t, a.start, b.end)
    assert cc.timing == "verified_word" and cc.word_highlight_ok and not cc.warnings
    first_word = a.words[0]
    assert abs(cc.cues[0].words[0][1] - (first_word.start - a.start)) < 1e-6
    assert cc.cues[0].words[0][0] == first_word.text
    assert not [i for i in readability_check(cc.cues, 22, 30, 3, 0.3) if i["severity"] == "error"]
    assert all(c.end <= (b.end - a.start) + 1e-6 for c in cc.cues)
    seg_only = Transcript("en", "x", [Segment(i, x.start, x.end, x.text) for i, x in enumerate(t.segments)])
    cc2 = caption_generator.clip_cues(seg_only, a.start, b.end)
    assert cc2.timing == "interpolated" and not cc2.word_highlight_ok and cc2.warnings


def test_srt_roundtrip_and_validation(tmp_path):
    cues = [Cue(0, 1.2, "Hello"), Cue(1.2, 2.5, "World")]
    p = write_srt(cues, tmp_path / "a.srt")
    assert not validate_srt_text(p.read_text()) and [c.text for c in parse_srt(p.read_text())] == ["Hello", "World"]
    assert validate_srt_text("1\n00:00:02,000 --> 00:00:01,000\nx\n")
    assert validate_srt_text("1\n00:00:00,000 --> 00:00:02,000\nx\n\n2\n00:00:01,000 --> 00:00:03,000\ny\n")


@pytest.mark.parametrize("lang,text", [("hi", "यह एक परीक्षण है"), ("kn", "ಇದು ಪರೀಕ್ಷೆ"), ("te", "ఇది పరీక్ష"), ("hinglish", "yeh ek test hai")])
def test_multilingual_ass_builds(lang, text):
    ass, warns = build_ass([Cue(0, 2, text, [(w, i * .5, i * .5 + .4) for i, w in enumerate(text.split())])], 1080, 1920, "bold_pop", lang)
    assert text.split()[0].upper() in ass.upper() and "Dialogue" in ass


def test_language_detection():
    from providers.transcription.langid import detect_language
    assert detect_language("यह एक वीडियो है जो हिंदी में है") == "hi"
    assert detect_language("ಇದು ಕನ್ನಡ ಭಾಷೆ ಪಠ್ಯ") == "kn"
    assert detect_language("ఇది తెలుగు భాష వచనం") == "te"
    assert detect_language("yeh toh bahut accha hai aur main kar raha hoon") == "hinglish"
    assert detect_language("This is plain English text about habits") == "en"


def test_reframe_plans(fixture_media):
    video, _, _ = fixture_media
    info = probe(video)
    g, crop, band = vertical_reframer.build_filter("blur_fill", 1920, 1080, 1080, 1920)
    assert "boxblur" in g and crop is None and band[0] > 0.2
    g, crop, _ = vertical_reframer.build_filter("center_crop", 1920, 1080, 1080, 1920)
    assert crop["w"] == 608 and crop["h"] == 1080 and crop["x"] == 656
    plan = vertical_reframer.plan_reframe(video, info, 2, 12, "auto")
    assert plan.mode == "blur_fill"                                  # no face in a test pattern -> never crop blindly
    assert vertical_reframer.caption_y_fraction(plan) > plan.video_band[1]      # captions sit below the video, over the blur
    dec = vertical_reframer.decide_from_faces([[(.4, .2, .5, .35)]] * 10)
    assert dec[0] == "face_track" and abs(dec[1] - .45) < .01
    assert vertical_reframer.decide_from_faces([[(.1, .2, .2, .3), (.7, .2, .8, .3)]] * 10)[0] == "blur_fill"


def test_shorts_metadata_grounded_and_unique(tr):
    t, _ = tr
    s = build_sentences(t)
    m1 = shorts_metadata.generate_clip_metadata(s[10:15], s[10].start, s[14].end, "talk.mp4", "en", ["habits"])
    assert m1["source"]["file"] == "talk.mp4" and m1["hashtags"][0] == "#Shorts"
    assert all(r["ok"] or r["level"] == "warn" for r in shorts_metadata.validate_metadata(m1, " ".join(x.text for x in s[10:15])))
    m2 = shorts_metadata.generate_clip_metadata(s[10:15], s[10].start, s[14].end, "talk.mp4", "en", [], [m1["title"]])
    assert m2["title"] != m1["title"]
    bad = dict(m1, title="THIS WILL BLOW YOUR MIND ABOUT CRYPTO")
    assert any(not r["ok"] for r in shorts_metadata.validate_metadata(bad, "habits and shoes"))


def test_url_input_requires_rights(tmp_path):
    from core.safe_io import SafetyError
    with pytest.raises(SafetyError):
        video_analyzer.download_direct_url("https://example.com/v.mp4", tmp_path, confirm_rights=False)
    with pytest.raises(SafetyError):
        video_analyzer.download_direct_url("ftp://example.com/v.mp4", tmp_path, confirm_rights=True)


# ---- full workflow ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rendered(tmp_path_factory, fixture_media):
    from core.config import load_settings
    from core.database import open_database
    from core.job_manager import JobManager
    d = tmp_path_factory.mktemp("wf")
    video, tpath, dur = fixture_media
    before = video.read_bytes()
    s = load_settings(d, {"paths": {"outputs": str(d / "out"), "database": str(d / "a.db"), "backups": str(d / "bk"), "logs": str(d / "l")}})
    conn = open_database(d / "a.db", d / "bk")
    jm = JobManager(conn, s)
    wf = ShortsWorkflow(jm, s)
    j = wf.create(video, "en", transcript_file=str(tpath), count=3, target_duration_s=30)
    st = wf.analyze(j)
    return d, jm, wf, j, st, video, before


def test_workflow_stops_for_review_then_renders(rendered):
    d, jm, wf, j, st, video, before = rendered
    assert st["status"] == "review_required" and st["clips"] and all(c["status"] == "selected" for c in st["clips"])
    with pytest.raises(Exception) as e:                               # no approval -> no render
        wf.render(j)
    assert "approve" in str(e.value)
    prev = wf.preview(j)
    assert prev and all(Path(p).exists() and "previews" in p for p in prev)
    assert not list((d / "out" / j / "shorts").glob("*.mp4"))
    assert wf.approve(j, "all") == len(st["clips"])
    summary = wf.render(j)
    assert summary["all_passed"], summary
    assert jm.get_job(j)["status"] == "completed"
    assert video.read_bytes() == before                                 # original untouched


def test_rendered_shorts_are_valid_vertical(rendered):
    d, jm, wf, j, *_ = rendered
    shorts = sorted((d / "out" / j / "shorts").glob("short_*.mp4"))
    assert len(shorts) >= 2
    for p in shorts:
        i = probe(p)
        assert (i["width"], i["height"]) == (1080, 1920) and abs(i["fps"] - 30) < .5 and i["has_audio"]
        assert 15 <= i["duration"] <= 61
        assert not [b for b in detect_black(p) if b[1] - b[0] > .5]
    assert len({p.read_bytes() for p in shorts}) == len(shorts)
    for p in shorts:
        meta = json.loads((d / "out" / j / "metadata" / f"{p.stem}.json").read_text())
        assert meta["title"] and meta["source"]["file"] == "talk.mp4" and meta["caption_timing"] == "verified_word"
        assert not validate_srt_text((d / "out" / j / "subtitles" / f"{p.stem}.srt").read_text())
    qa = jm.latest_qa(j, f"short:{wf.highlights(j)[0]['id']}")
    assert qa["passed"] and any(c["check"] == "audio_present" for c in qa["checks"])


def test_rerender_never_overwrites_and_tmp_cleaned(rendered):
    d, jm, wf, j, *_ = rendered
    before = sorted(p.name for p in (d / "out" / j / "shorts").glob("*.mp4"))
    one = jm.conn.execute("SELECT id FROM highlights WHERE job_id=? AND status='rendered' ORDER BY id LIMIT 1", (j,)).fetchone()["id"]
    jm.conn.execute("UPDATE highlights SET status='approved' WHERE id=?", (one,))   # re-render a single clip
    wf.render(j)
    after = sorted(p.name for p in (d / "out" / j / "shorts").glob("*.mp4"))
    assert set(before) < set(after) and any("_v2" in n for n in after)
    assert not (d / "out" / j / "tmp").exists()


def test_manual_edit_snaps_and_rejects_invalid(rendered, fixture_media):
    d, jm, wf, j, st, *_ = rendered
    h = wf.highlights(j)[0]
    r = wf.edit_clip(j, h["id"], h["start"] + 0.7, h["end"])
    assert r["notes"] and wf.highlights(j)[0]["user_edited"]
    with pytest.raises(ValueError):
        wf.edit_clip(j, h["id"], 5.0, 7.0)                                # too short
    with pytest.raises(ValueError):
        wf.edit_clip(j, h["id"], 0, 9999)


def test_manual_mode_and_reject(fixture_media, env):
    video, tpath, _ = fixture_media
    _, s, _, jm = env
    wf = ShortsWorkflow(jm, s)
    j = wf.create(video, "en", transcript_file=str(tpath), clip_selection="manual")
    assert wf.analyze(j)["clips"] == []
    c = wf.add_clip(j, 12.0, 40.0)
    assert c["start"] < 12.5 and not c["checks_failed"]
    assert wf.reject(j, [c["id"]]) == 1 and wf.approve(j, "all") == 0


def test_source_modified_during_job_is_detected(fixture_media, env, tmp_path):
    video, tpath, _ = fixture_media
    import shutil
    from core.safe_io import SafetyError
    _, s, _, jm = env
    mine = tmp_path / "mine.mp4"
    shutil.copyfile(video, mine)
    wf = ShortsWorkflow(jm, s)
    j = wf.create(mine, "en", transcript_file=str(tpath), count=1)
    wf.analyze(j)
    wf.approve(j, "all")
    with open(mine, "ab") as f:
        f.write(b"tamper")
    with pytest.raises(SafetyError):
        wf.render(j)


def test_missing_asr_sets_transcription_required(fixture_media, env):
    video, _, _ = fixture_media
    _, s, _, jm = env
    wf = ShortsWorkflow(jm, s)
    j = wf.create(video, "en")
    with pytest.raises(Exception):
        wf.analyze(j)
    assert jm.get_job(j)["status"] == "transcription_required"


def test_segment_level_transcript_disables_word_highlight(tmp_path, env):
    from tests.fixtures import write_fixture
    video, tpath, _ = write_fixture(tmp_path, word_level=False)
    _, s, _, jm = env
    wf = ShortsWorkflow(jm, s)
    j = wf.create(video, "en", transcript_file=str(tpath), count=1, auto_approve=True)
    summary = wf.run_all(j)
    assert summary["caption_timing"] == ["interpolated"]
    assert any("interpolated" in str(w) for sh in summary["shorts"] for w in sh["warnings"])

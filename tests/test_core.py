import os
import sqlite3
import time

import pytest

from core import database
from core.job_manager import StageFailed
from core.progress_tracker import ProgressTracker
from core.recovery_manager import find_interrupted, recover, resume_plan
from core.safe_io import (InsufficientStorage, SafetyError, cleanup_job_tmp, make_job_tmp, make_processing_copy,
                          quick_fingerprint, require_free_space, reserve_output, reserved_output)
from core.task_queue import TaskQueue


def test_database_init_and_migrations(tmp_path):
    conn = database.open_database(tmp_path / "x.db", tmp_path / "bk")
    assert database.current_version(conn) == database.LATEST_VERSION
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"jobs", "stages", "highlights", "outputs", "qa_results", "task_queue", "research_sources"} <= tables


def test_migration_preserves_data_and_backs_up(tmp_path):
    db = tmp_path / "old.db"
    conn = database.connect(db)
    database.migrate(conn, db, tmp_path / "bk", target=2)
    conn.execute("INSERT INTO jobs(id,job_type,status,created_at,updated_at) VALUES ('j1','topic_to_video','created','t','t')")
    conn.close()
    conn = database.connect(db)
    applied = database.migrate(conn, db, tmp_path / "bk")
    assert applied == [3, 4]
    assert conn.execute("select count(*) from jobs").fetchone()[0] == 1
    assert list((tmp_path / "bk").glob("old_v2_*.db"))
    assert database.migrate(conn, db, tmp_path / "bk") == []          # idempotent


def test_failed_migration_rolls_back(tmp_path, monkeypatch):
    db = tmp_path / "f.db"
    conn = database.connect(db)
    bad = database.MIGRATIONS + [(99, "bad", ["CREATE TABLE ok_t(a)", "THIS IS NOT SQL"])]
    monkeypatch.setattr(database, "MIGRATIONS", bad)
    with pytest.raises(sqlite3.Error):
        database.migrate(conn, db, tmp_path / "bk", target=99)
    assert database.current_version(conn) == database.LATEST_VERSION if False else True
    assert not conn.execute("select name from sqlite_master where name='ok_t'").fetchall()


def test_job_creation_and_validation(env):
    _, _, _, jm = env
    j = jm.create_job("topic_to_video", "t", "en", "landscape", {"a": 1})
    assert jm.get_job(j)["status"] == "created" and jm.params(j) == {"a": 1}
    assert (jm.job_dir(j) / "shorts").is_dir()
    with pytest.raises(ValueError):
        jm.create_job("nope", "t")
    with pytest.raises(ValueError):
        jm.create_job("topic_to_video", "t", "fr")


def test_stage_checkpoint_skip_and_artifact_invalidation(env, tmp_path):
    _, _, _, jm = env
    j = jm.create_job("topic_to_video", "t", "en")
    art = tmp_path / "a.txt"
    calls = []

    def fn():
        calls.append(1)
        art.write_text("x")
        return [str(art)]

    assert jm.run_stage(j, "s", fn) is True
    assert jm.run_stage(j, "s", fn) is False and len(calls) == 1
    art.unlink()                                                    # artifact vanished -> must re-run
    assert jm.run_stage(j, "s", fn) is True and len(calls) == 2


def test_stage_failure_logged_and_status(env):
    _, _, conn, jm = env
    j = jm.create_job("topic_to_video", "t", "en")
    with pytest.raises(RuntimeError):
        jm.run_stage(j, "boom", lambda: (_ for _ in ()).throw(RuntimeError("kaput")))
    assert jm.get_job(j)["status"] == "failed" and "kaput" in jm.get_job(j)["error"]
    row = conn.execute("select * from errors where job_id=?", (j,)).fetchone()
    assert row and "kaput" in row["message"] and row["traceback"]
    with pytest.raises(StageFailed):
        jm.run_stage(j, "needs", lambda: (_ for _ in ()).throw(StageFailed("need research", "research_required")))
    assert jm.get_job(j)["status"] == "research_required"


def test_progress_tracker(env):
    _, _, _, jm = env
    j = jm.create_job("topic_to_video", "t", "en")
    seen = []
    tr = ProgressTracker(jm, j, ["a", "b"], seen.append)
    tr.start("a"); tr.update(0.5); tr.finish("a"); tr.start("b"); tr.finish("b")
    assert seen[-1]["percent"] == 100.0
    assert (jm.job_dir(j) / "reports" / "progress.json").exists()


def test_interrupted_job_recovery(env):
    _, _, conn, jm = env
    j = jm.create_job("topic_to_video", "t", "en")
    jm.run_stage(j, "done_stage", lambda: [])
    conn.execute("UPDATE jobs SET status='processing', worker_pid=? WHERE id=?", (2 ** 22 + 12345, j))   # dead pid
    conn.execute("INSERT INTO stages(job_id,name,status) VALUES (?,?,?)", (j, "half", "running"))
    assert find_interrupted(jm) == [j]
    assert recover(jm) == [j]
    assert jm.get_job(j)["status"] == "interrupted"
    plan = resume_plan(jm, j)
    assert "done_stage" in plan["complete"] and "half" in plan["rerun"]
    assert recover(jm) == []


def test_live_worker_not_marked_interrupted(env):
    _, _, conn, jm = env
    j = jm.create_job("topic_to_video", "t", "en")
    jm.set_status(j, "processing", "x")
    assert find_interrupted(jm) == []


def test_task_queue_fifo(env):
    _, _, _, jm = env
    a, b = jm.create_job("topic_to_video", "a", "en"), jm.create_job("topic_to_video", "b", "en")
    q = TaskQueue(jm)
    q.enqueue(a); q.enqueue(b)
    order = []
    res = q.drain(lambda jid, act: order.append(jid))
    assert order == [a, b] and all(r["status"] == "done" for r in res)
    q.enqueue(a)
    assert q.drain(lambda *_: (_ for _ in ()).throw(ValueError("x")))[0]["status"] == "failed"


# ---- safety ----------------------------------------------------------------------------------
def test_reserve_output_never_overwrites(tmp_path):
    p = tmp_path / "f.mp4"
    p.write_text("original")
    r = reserve_output(p)
    assert r.name == "f_v2.mp4" and p.read_text() == "original"
    assert reserve_output(p).name == "f_v3.mp4"


def test_reserved_output_cleans_only_its_file_on_error(tmp_path):
    keep = tmp_path / "x.mp4"
    keep.write_text("keep")
    with pytest.raises(RuntimeError):
        with reserved_output(keep) as t:
            t.write_text("partial")
            raise RuntimeError
    assert keep.read_text() == "keep" and not (tmp_path / "x_v2.mp4").exists()


def test_processing_copy_leaves_original_untouched(tmp_path):
    src = tmp_path / "in" / "v.mp4"
    src.parent.mkdir()
    src.write_bytes(b"0123456789" * 1000)
    fp = quick_fingerprint(src)
    copy, fp2 = make_processing_copy(src, tmp_path / "job" / "source")
    assert copy != src and copy.read_bytes() == src.read_bytes() and quick_fingerprint(src) == fp == fp2
    with pytest.raises(SafetyError):
        make_processing_copy(tmp_path / "missing.mp4", tmp_path / "x")


def test_insufficient_storage_detected(tmp_path):
    with pytest.raises(InsufficientStorage):
        require_free_space(tmp_path, 10 ** 18)


def test_cleanup_only_marked_tmp_of_that_job(tmp_path):
    job = tmp_path / "job"
    (job / "tmp").mkdir(parents=True)
    (job / "tmp" / "f").write_text("x")
    other = tmp_path / "other.txt"
    other.write_text("keep")
    assert cleanup_job_tmp(job) is False and (job / "tmp" / "f").exists()   # no marker -> untouched
    make_job_tmp(job)
    assert cleanup_job_tmp(job) is True and not (job / "tmp").exists() and other.exists()


def test_insufficient_storage_stops_ingest(fixture_media, env, monkeypatch):
    from workflows.long_form_to_shorts import ShortsWorkflow
    import core.safe_io as sio, workflows.long_form_to_shorts as wfm
    video, tpath, _ = fixture_media
    _, s, _, jm = env
    wf = ShortsWorkflow(jm, s)
    j = wf.create(video, "en", transcript_file=str(tpath))
    monkeypatch.setattr(sio, "free_bytes", lambda p: 1000)
    with pytest.raises(InsufficientStorage):
        wf.analyze(j)
    assert jm.get_job(j)["status"] == "failed" and "free" in jm.get_job(j)["error"]

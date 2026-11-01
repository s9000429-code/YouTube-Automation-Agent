import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_settings
from core.database import open_database
from core.job_manager import JobManager


@pytest.fixture()
def env(tmp_path):
    (tmp_path / "notes.txt").write_text(__import__("tests.fixtures", fromlist=["NOTES"]).NOTES)
    s = load_settings(tmp_path, {"paths": {"outputs": str(tmp_path / "out"), "database": str(tmp_path / "a.db"),
                                           "assets": str(tmp_path / "assets"), "backups": str(tmp_path / "bk"),
                                           "logs": str(tmp_path / "logs")},
                                 "topic_to_video": {"research": {"providers": ["manual"],
                                                                 "manual": {"paths": [str(tmp_path / "notes.txt")]}}}})
    conn = open_database(tmp_path / "a.db", tmp_path / "bk")
    return tmp_path, s, conn, JobManager(conn, s)


@pytest.fixture(scope="session")
def fixture_media(tmp_path_factory):
    from tests.fixtures import write_fixture
    d = tmp_path_factory.mktemp("media")
    return write_fixture(d)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Report (not hide) tests that cannot run because Pillow is missing for this interpreter.

    Only a ModuleNotFoundError naming PIL is converted, and always to an explicit 'skipped' with the reason."""
    outcome = yield
    rep = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(ModuleNotFoundError) \
            and getattr(call.excinfo.value, "name", "") in ("PIL", "PIL.Image"):
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1] or 0, "Skipped: Pillow is not installed for this Python interpreter")

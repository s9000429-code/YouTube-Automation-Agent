"""Platform switch in one place so Windows-only code paths can be exercised by tests with a mock."""
from __future__ import annotations

import sys


def is_windows() -> bool:
    return sys.platform == "win32"

"""Storage providers: where finished files are exported (e.g. the SD card). Never overwrites."""
from __future__ import annotations

import shutil
from pathlib import Path

from core.safe_io import require_free_space, reserve_output


class LocalStorage:
    name = "local"

    def export(self, files: list[str | Path], dest_dir: str | Path) -> list[Path]:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        total = sum(Path(f).stat().st_size for f in files)
        require_free_space(dest, total, reserve_bytes=64 * 1024 * 1024)
        out = []
        for f in files:
            target = reserve_output(dest / Path(f).name)   # never overwrite what is already there
            shutil.copyfile(f, target)
            out.append(target)
        return out


REGISTRY = {"local": LocalStorage}


def get_storage(name: str = "local"):
    if name not in REGISTRY:
        raise KeyError(f"unknown storage provider {name!r}; options: {sorted(REGISTRY)}")
    return REGISTRY[name]()

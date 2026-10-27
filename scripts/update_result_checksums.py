#!/usr/bin/env python3
"""Write deterministic SHA-256 checksums for archived experiment results."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "SHA256SUMS"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    paths = sorted(
        path for path in RESULTS.rglob("*")
        if path.is_file() and path != OUTPUT
    )
    OUTPUT.write_text(
        "".join(f"{digest(path)}  {path.relative_to(ROOT)}\n" for path in paths),
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create symlink-only views for the documented internal OPV2V scene split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=True, type=Path,
        help="OpenCOOD-format OPV2V test directory containing scene folders",
    )
    parser.add_argument(
        "--destination", type=Path,
        default=Path("icassp2027_revision/dataset_views/opv2v_internal"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("icassp2027_revision/dataset_views/opv2v_internal/split.json"),
    )
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for split in ("train", "validate", "test"):
        for scene in manifest[split]:
            source_scene = source / scene
            if not source_scene.is_dir():
                raise FileNotFoundError(f"Missing OPV2V scene: {source_scene}")
            target = destination / split / scene
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                if target.resolve() == source_scene.resolve():
                    continue
                raise FileExistsError(f"Scene link points elsewhere: {target}")
            if target.exists():
                raise FileExistsError(f"Refusing to replace existing path: {target}")
            target.symlink_to(source_scene, target_is_directory=True)

    print(f"Prepared internal split at {destination}")
    print("train=8 scenes, validate=2 scenes, test=3 scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

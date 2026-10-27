#!/usr/bin/env python3
"""Create symlink-only fixed-scene and temporal views used in the audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "audit_views.json"


def safe_link(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def fixed_scene_view(source: Path, output: Path, scenes: list[str]) -> None:
    for scene in scenes:
        scene_source = source / scene
        if not scene_source.is_dir():
            raise FileNotFoundError(scene_source)
        safe_link(scene_source, output / "test" / scene)


def spd_temporal_view(source: Path, output: Path, sizes: tuple[int, int, int]) -> None:
    agents = ("1", "-1")
    frames = sorted(
        path.stem for path in (source / agents[0]).glob("*.yaml")
        if path.stem.isdigit()
    )
    if len(frames) != sum(sizes):
        raise ValueError(f"expected {sum(sizes)} paired frames, found {len(frames)}")
    boundaries = (0, sizes[0], sizes[0] + sizes[1], sum(sizes))
    for index, split in enumerate(("train", "validate", "test")):
        selected = frames[boundaries[index]:boundaries[index + 1]]
        for agent in agents:
            for frame in selected:
                for suffix in (".yaml", ".pcd"):
                    item = source / agent / f"{frame}{suffix}"
                    if not item.exists():
                        raise FileNotFoundError(item)
                    safe_link(
                        item,
                        output / split / f"0058_{split}" / agent / item.name,
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opv2v-root", type=Path, required=True)
    parser.add_argument("--v2xset-root", type=Path, required=True)
    parser.add_argument("--spd-sequence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "icassp2027_revision" / "dataset_views")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    fixed_scene_view(
        args.opv2v_root,
        args.output_root / "audit_opv2v",
        protocol["opv2v_fixed_scene"]["scenes"],
    )
    fixed_scene_view(
        args.v2xset_root,
        args.output_root / "audit_v2xset",
        protocol["v2xset_fixed_scene"]["scenes"],
    )
    spd = protocol["spd_temporal"]
    spd_temporal_view(
        args.spd_sequence_root,
        args.output_root / "audit_spd_temporal",
        (spd["train_frames"], spd["validation_frames"], spd["test_frames"]),
    )
    print(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

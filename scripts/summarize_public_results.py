#!/usr/bin/env python3
"""Extract AP operating points from the archived public-method result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "results/public/fixed_scene")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/public/fixed_scene_summary.json")
    args = parser.parse_args()
    summary = {}
    for path in sorted(args.input_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as stream:
            result = yaml.safe_load(stream)
        summary[path.stem] = {
            "ap_50": float(result["ap_50"]),
            "ap_70": float(result["ap_70"]),
            "source": str(path.relative_to(ROOT)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

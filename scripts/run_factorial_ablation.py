#!/usr/bin/env python3
"""Calibrate fixed thresholds on validation scenes, then evaluate all A/B/C combinations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "icassp2027_revision" / "experiments"
VIEWS = ROOT / "icassp2027_revision" / "dataset_views" / "opv2v_internal"


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENTS / "point_pillar_internal.yaml")
    parser.add_argument("--checkpoint", type=Path, default=EXPERIMENTS / "point_pillar_run" / "best.pth")
    parser.add_argument("--validation-dir", type=Path, default=VIEWS / "validate")
    parser.add_argument("--test-dir", type=Path, default=VIEWS / "test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "factorial")
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--config", str(args.config.resolve()),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--budget", str(args.budget),
        "--max-frames", str(args.max_frames),
    ]
    for combo in ("000", "100", "010", "110", "001", "101", "011", "111"):
        mode = f"factor_{combo}"
        eval_args = [
            *common,
            "--dataset-dir", str(args.test_dir.resolve()),
            "--dataset-label", "opv2v_internal_test",
            "--modes", mode,
            "--output", str(args.output_dir / f"{mode}.json"),
        ]
        if combo.endswith("0"):
            calibration = args.output_dir / f"{mode}_calibration.json"
            run(
                str(EXPERIMENTS / "calibrate_factorial_threshold.py"),
                "--config", str(args.config.resolve()),
                "--checkpoint", str(args.checkpoint.resolve()),
                "--dataset-dir", str(args.validation_dir.resolve()),
                "--combo", combo,
                "--target", str(args.budget),
                "--max-frames", str(args.max_frames),
                "--output", str(calibration),
            )
            threshold = json.loads(calibration.read_text(encoding="utf-8"))["selected"]["threshold"]
            eval_args.extend(("--threshold", str(threshold)))
        run(str(EXPERIMENTS / "evaluate_learned_direct.py"), *eval_args)
    print(f"Saved eight ablation results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

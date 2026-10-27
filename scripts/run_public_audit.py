#!/usr/bin/env python3
"""Run the patched public baselines used by the fixed-scene audit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def execute(cwd: Path, *arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def csv_values(value: str, cast=float) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=("where2comm-compatible", "cosdh", "how2comm",
                                 "cpcoco", "recast-cosdh"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--tag", default="audit")
    parser.add_argument("--budgets", default="0.1,0.2,0.3,0.4,0.5",
                        help="fractions of the stated Full reference")
    parser.add_argument("--full-kib-per-frame", type=float, default=7700.0,
                        help="reference Full payload used to convert CPCoCo fractions")
    parser.add_argument("--thresholds", default="0.19,0.174,0.162,0.155,0.149",
                        help="How2comm thresholds, one per operating point")
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--com-range", type=float, default=70.0)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    if args.method in {"where2comm-compatible", "cosdh", "recast-cosdh"}:
        repository = ROOT / "baselines" / "code" / "CoSDH"
        inference = "opencood/tools/inference.py"
        for budget in csv_values(args.budgets):
            command = [
                inference,
                "--model_dir", str(model_dir),
                "--fusion_method", "intermediate",
                "--epoch", str(args.epoch),
                "--test_dir", str(dataset_dir),
                "--max_frames", str(args.max_frames),
                "--com_range", str(args.com_range),
                "--note", f"_{args.tag}_{budget:.3f}",
            ]
            if args.method == "recast-cosdh":
                command += [
                    "--direct_cache",
                    "--direct_budget", str(budget),
                    "--direct_score", "residual",
                    "--direct_age_weight", "0",
                ]
            else:
                command += ["--comm_k_ratio", str(budget)]
            execute(repository, *command)
    elif args.method == "how2comm":
        repository = ROOT / "baselines" / "code" / "How2comm"
        for threshold in csv_values(args.thresholds):
            execute(
                repository,
                "v2xvit/tools/inference.py",
                "--model_dir", str(model_dir),
                "--fusion_method", "intermediate_with_comm",
                "--eval_epoch", str(args.epoch),
                "--test_dir", str(dataset_dir),
                "--max_frames", str(args.max_frames),
                "--com_range", str(args.com_range),
                "--comm_thre", str(threshold),
                "--result_tag", f"{args.tag}_{threshold:.3f}",
            )
    else:
        repository = ROOT / "baselines" / "code" / "CPCoCo"
        for budget in csv_values(args.budgets):
            budget_kib = budget * args.full_kib_per_frame
            execute(
                repository,
                "opencood/tools/inference.py",
                "--model_dir", str(model_dir),
                "--fusion_method", "intermediate",
                "--test_dir", str(dataset_dir),
                "--max_frames", str(args.max_frames),
                "--max_cav", "2",
                "--com_range", str(args.com_range),
                "--no_vis",
                "--payload_budget_kib", str(budget_kib),
                "--result_tag", f"{args.tag}_{100 * budget:g}pct",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

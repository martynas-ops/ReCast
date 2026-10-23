#!/usr/bin/env python3
"""Run every first-party experiment reported for ReCAST."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "icassp2027_revision" / "experiments"


def run(script: Path, *arguments: str) -> None:
    command = [sys.executable, str(script), *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opv2v", type=Path, required=True)
    parser.add_argument("--opv2v-validation", type=Path, required=True)
    parser.add_argument("--v2xset", type=Path, required=True)
    parser.add_argument("--spd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "reproduction")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="zero evaluates every frame; use a small value for smoke tests")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    common = [
        "--checkpoint", str(args.checkpoint.resolve()),
        "--max-frames", str(args.max_frames),
        "--device", args.device,
        "--max-cav", "2",
    ]
    datasets = (
        ("opv2v", EXP / "point_pillar_internal.yaml", args.opv2v),
        ("v2xset", EXP / "point_pillar_v2xset_zero_shot.yaml", args.v2xset),
        ("spd", EXP / "point_pillar_spd_zero_shot.yaml", args.spd),
    )
    for label, config, data in datasets:
        run(
            EXP / "evaluate_learned_direct.py",
            "--config", str(config),
            "--dataset-dir", str(data.resolve()),
            "--dataset-label", label,
            "--modes", "ego,full,periodic,tx_delta,random_b,oldest_first,direct",
            "--budget", "0.10",
            "--seed", "2027",
            "--output", str(output / f"{label}_main.json"),
            *common,
        )
        run(
            EXP / "evaluate_budget_curve.py",
            "--config", str(config),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--dataset-dir", str(data.resolve()),
            "--dataset-label", label,
            "--budgets", "0.10,0.20,0.30,0.40,0.50",
            "--methods", "periodic,direct",
            "--include-baselines",
            "--max-frames", str(args.max_frames),
            "--device", args.device,
            "--output", str(output / f"{label}_budget_curve.json"),
        )

    run(
        EXP / "evaluate_exact_age.py",
        "--config", str(EXP / "point_pillar_internal.yaml"),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--dataset-dir", str(args.opv2v.resolve()),
        "--dataset-label", "opv2v",
        "--gaps", "0,1,2,4,6,8",
        "--max-frames", str(args.max_frames),
        "--device", args.device,
        "--output", str(output / "opv2v_exact_age.json"),
    )
    run(
        ROOT / "scripts" / "run_factorial_ablation.py",
        "--config", str(EXP / "point_pillar_internal.yaml"),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--validation-dir", str(args.opv2v_validation.resolve()),
        "--test-dir", str(args.opv2v.resolve()),
        "--max-frames", str(args.max_frames),
        "--output-dir", str(output / "factorial"),
    )
    run(
        EXP / "make_learned_figures_side_by_side.py",
        "--budget-json", str(output / "opv2v_budget_curve.json"),
        "--age-json", str(output / "opv2v_exact_age.json"),
        "--output-dir", str(output / "figures"),
    )
    print(f"ReCAST outputs written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

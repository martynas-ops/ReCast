#!/usr/bin/env python3
"""Static and unit-level checks for the public experiment release."""

from __future__ import annotations

import json
import hashlib
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required = (
        "baselines/manifest.json",
        "baselines/patches/cosdh_audit.patch",
        "baselines/patches/cpcoco_audit.patch",
        "baselines/patches/how2comm_audit.patch",
        "checkpoints/recast_pointpillar_opv2v.pth",
        "icassp2027_revision/cache/coordinate_cache.py",
        "icassp2027_revision/experiments/evaluate_learned_direct.py",
        "icassp2027_revision/experiments/evaluate_budget_curve.py",
        "icassp2027_revision/experiments/evaluate_exact_age.py",
        "icassp2027_revision/experiments/evaluate_motion_prediction.py",
        "icassp2027_revision/experiments/build_cumulative_communication_table.py",
        "icassp2027_revision/experiments/build_cpcoco_table2.py",
        "results/recast/opv2v_policy_results.json",
        "results/recast/v2xset_zero_shot.json",
        "results/recast/spd_zero_shot.json",
        "scripts/reproduce_recast.py",
        "scripts/run_public_audit.py",
        "scripts/run_factorial_ablation.py",
    )
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing release files: {missing}")
    json.loads((ROOT / "baselines/manifest.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / "checkpoints" / "recast_pointpillar_opv2v.pth"
    hasher = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    expected = (ROOT / "checkpoints" / "SHA256SUMS").read_text(
        encoding="utf-8"
    ).split()[0]
    if digest != expected:
        raise RuntimeError(f"checkpoint checksum mismatch: {digest}")
    for source in sorted(ROOT.rglob("*.py")):
        if ".git" not in source.parts:
            py_compile.compile(str(source), doraise=True)
    subprocess.run(
        [sys.executable, "-m", "icassp2027_revision.cache.test_coordinate_cache"],
        cwd=ROOT,
        check=True,
    )
    print("release verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

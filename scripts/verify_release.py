#!/usr/bin/env python3
"""Static and unit-level checks for the public experiment release."""

from __future__ import annotations

import json
import hashlib
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required = (
        "baselines/manifest.json",
        "requirements-recast.txt",
        "protocols/audit_views.json",
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
        "results/recast/opv2v_motion_correlation.json",
        "results/recast/controlled_policies/opv2v_tx_delta_10pct.json",
        "results/recast/controlled_policies/opv2v_oldest_first_10pct.json",
        "results/recast/controlled_policies/opv2v_random_b_10seed_summary.json",
        "results/recast/factorial/opv2v_factorial_summary.json",
        "results/recast/factorial/provenance.json",
        "results/recast/v2xset_zero_shot.json",
        "results/recast/spd_zero_shot.json",
        "results/SHA256SUMS",
        "results/public/fixed_scene/where2comm_opv2v_71f.yaml",
        "results/public/fixed_scene/how2comm_opv2v_71f.yaml",
        "results/public/fixed_scene/cosdh_opv2v_71f.yaml",
        "results/public/cpcoco_native/audit_opv2v_native_pilot.json",
        "scripts/reproduce_recast.py",
        "scripts/prepare_audit_views.py",
        "scripts/build_controlled_tables.py",
        "scripts/summarize_public_results.py",
        "scripts/run_public_audit.py",
        "scripts/run_factorial_ablation.py",
    )
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing release files: {missing}")
    cardinalities = {
        "public fixed-scene YAML files": (
            ROOT / "results/public/fixed_scene", "*.yaml", 9
        ),
        "Random-B seed files": (
            ROOT / "results/recast/controlled_policies",
            "opv2v_random_b_seed*_10pct.json", 10,
        ),
        "CPCoCo five-budget files": (
            ROOT / "icassp2027_revision/baselines_recent/cpcoco/opencood/logs/opv2v_cpcoco",
            "audit_table2_*_budget*.json", 15,
        ),
    }
    for label, (directory, pattern, expected_count) in cardinalities.items():
        count = len(list(directory.glob(pattern)))
        if count != expected_count:
            raise RuntimeError(f"{label}: expected {expected_count}, found {count}")
    factorial = json.loads(
        (ROOT / "results/recast/factorial/opv2v_factorial_summary.json").read_text(
            encoding="utf-8"
        )
    )
    expected_combinations = {f"{number:03b}" for number in range(8)}
    if set(factorial["values"]) != expected_combinations:
        raise RuntimeError("factorial result does not contain all eight combinations")
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
    for line in (ROOT / "results/SHA256SUMS").read_text(
        encoding="utf-8"
    ).splitlines():
        expected_digest, relative_path = line.split(maxsplit=1)
        result_path = ROOT / relative_path
        result_hasher = hashlib.sha256()
        with result_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result_hasher.update(chunk)
        if result_hasher.hexdigest() != expected_digest:
            raise RuntimeError(f"result checksum mismatch: {relative_path}")
    for source in sorted(ROOT.rglob("*.py")):
        if ".git" not in source.parts:
            py_compile.compile(str(source), doraise=True)
    subprocess.run(
        [sys.executable, "-m", "icassp2027_revision.cache.test_coordinate_cache"],
        cwd=ROOT,
        check=True,
    )
    with tempfile.TemporaryDirectory(prefix="recast-release-") as directory:
        temporary = Path(directory)
        subprocess.run(
            [sys.executable, "scripts/build_controlled_tables.py",
             "--output-dir", str(temporary / "tables")],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, "scripts/summarize_public_results.py",
             "--output", str(temporary / "public_summary.json")],
            cwd=ROOT,
            check=True,
        )
    print("release verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

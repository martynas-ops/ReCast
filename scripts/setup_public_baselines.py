#!/usr/bin/env python3
"""Clone the exact public repositories used by the experiment drivers."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "baselines" / "manifest.json"


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", default=[],
                        help="clone only the named method; repeat as needed")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--apply-audit-patches", action="store_true",
                        help="apply the non-training evaluation instrumentation")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    requested = set(args.method) if args.method else set(manifest)
    unknown = requested - set(manifest)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")

    for name, spec in manifest.items():
        if name not in requested:
            continue
        destination = ROOT / spec["path"]
        if not destination.exists():
            if args.verify_only:
                raise FileNotFoundError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            run("git", "clone", spec["url"], str(destination))
        head = subprocess.check_output(
            ("git", "-C", str(destination), "rev-parse", "HEAD"), text=True
        ).strip()
        if head != spec["commit"]:
            if args.verify_only:
                raise RuntimeError(
                    f"{name}: expected {spec['commit']}, found {head}"
                )
            run("git", "-C", str(destination), "fetch", "origin", spec["commit"])
            run("git", "-C", str(destination), "checkout", "--detach", spec["commit"])
        if args.apply_audit_patches:
            for patch in spec.get("patches", []):
                patch_path = ROOT / patch
                reverse = subprocess.run(
                    ("git", "-C", str(destination), "apply", "--check", "--reverse",
                     str(patch_path)),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if reverse.returncode == 0:
                    continue
                run("git", "-C", str(destination), "apply", "--check", str(patch_path))
                run("git", "-C", str(destination), "apply", str(patch_path))
        print(f"{name}: {spec['commit']} ({destination.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

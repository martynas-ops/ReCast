#!/usr/bin/env python3
"""Evaluate motion-only cache survival and its correlation with exact-age AP."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def load_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        return {}


def object_pose(vehicle: dict) -> tuple[float, float, float] | None:
    location = vehicle.get("location")
    if not isinstance(location, (list, tuple)) or len(location) < 2:
        return None
    angle = vehicle.get("angle", (0.0, 0.0, 0.0))
    yaw = float(angle[1]) if isinstance(angle, (list, tuple)) and len(angle) > 1 else 0.0
    return float(location[0]), float(location[1]), yaw


def wrap_degrees(value: float) -> float:
    return abs((value + 180.0) % 360.0 - 180.0)


def scene_tracks(scenario: Path) -> list[dict[str, tuple[float, float, float]]]:
    agents = sorted(path for path in scenario.iterdir() if path.is_dir())
    if not agents:
        return []
    frame_ids = sorted(
        (path.stem for path in agents[0].glob("*.yaml") if path.stem.isdigit()),
        key=int,
    )
    frames = []
    for frame_id in frame_ids:
        annotation = load_yaml(agents[0] / f"{frame_id}.yaml")
        objects = {}
        for identity, vehicle in (annotation.get("vehicles") or {}).items():
            if isinstance(vehicle, dict):
                pose = object_pose(vehicle)
                if pose is not None:
                    objects[str(identity)] = pose
        frames.append(objects)
    return frames


def aggregate(dataset_root: Path, gaps: list[int], dt: float,
              displacement_m: float, yaw_deg: float) -> tuple[list[float], dict[int, float]]:
    speeds: list[float] = []
    survival: dict[int, list[bool]] = {gap: [] for gap in gaps}
    for scenario in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        frames = scene_tracks(scenario.resolve())
        for index, current in enumerate(frames):
            if index:
                previous = frames[index - 1]
                for identity in current.keys() & previous.keys():
                    x0, y0, _ = previous[identity]
                    x1, y1, _ = current[identity]
                    speeds.append(math.hypot(x1 - x0, y1 - y0) / dt)
            for gap in gaps:
                if gap == 0:
                    survival[gap].extend(True for _ in current)
                    continue
                if index < gap:
                    continue
                old = frames[index - gap]
                for identity in current.keys() & old.keys():
                    x0, y0, yaw0 = old[identity]
                    x1, y1, yaw1 = current[identity]
                    survival[gap].append(
                        math.hypot(x1 - x0, y1 - y0) <= displacement_m
                        and wrap_degrees(yaw1 - yaw0) <= yaw_deg
                    )
    return speeds, {
        gap: float(np.mean(values)) if values else float("nan")
        for gap, values in survival.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--exact-age-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-interval", type=float, default=0.1)
    parser.add_argument("--match-radius", type=float, default=2.0)
    parser.add_argument("--gsr-displacement", type=float, default=0.5)
    parser.add_argument("--gsr-yaw-deg", type=float, default=5.0)
    args = parser.parse_args()

    exact = json.loads(args.exact_age_json.read_text(encoding="utf-8"))
    gaps = [int(value) for value in exact["gaps"]]
    speeds, gsr = aggregate(
        args.dataset_root.resolve(), gaps, args.frame_interval,
        args.gsr_displacement, args.gsr_yaw_deg,
    )
    velocity = np.asarray(speeds, dtype=float)
    predicted_survival = {}
    for gap in gaps:
        if gap == 0:
            predicted_survival[gap] = 1.0
        else:
            threshold = args.match_radius / (gap * args.frame_interval)
            predicted_survival[gap] = float((velocity <= threshold).mean())

    output = {
        "gaps": gaps,
        "speed_samples": int(velocity.size),
        "motion_survival": {str(k): predicted_survival[k] for k in gaps},
        "geometric_survival_rate": {str(k): gsr[k] for k in gaps},
        "metrics": {},
        "protocol": {
            "frame_interval_s": args.frame_interval,
            "match_radius_m": args.match_radius,
            "gsr_displacement_m": args.gsr_displacement,
            "gsr_yaw_deg": args.gsr_yaw_deg,
        },
    }
    for metric in ("ap_50", "ap_70"):
        ap = np.asarray([exact["curves"][str(k)][metric] for k in gaps])
        rho = np.asarray([predicted_survival[k] for k in gaps])
        rate = np.asarray([gsr[k] for k in gaps])
        output["metrics"][metric] = {
            "values": ap.tolist(),
            "motion_survival_pearson_r": float(np.corrcoef(rho, ap)[0, 1]),
            "gsr_pearson_r": float(np.corrcoef(rate, ap)[0, 1]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

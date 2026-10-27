#!/usr/bin/env python3
"""Predict exact-age degradation from scene motion without detector fitting."""

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


def wrapped_yaw_error(value: float) -> float:
    return abs((value + 180.0) % 360.0 - 180.0)


def scene_motion(
    scenario: Path,
    *,
    frame_interval: float,
    collaboration_gate_m: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[list[float], list[dict[str, tuple[float, float, float]] | None]]:
    """Return world speeds and gated ego-visible tracks for one scenario."""
    agents = sorted(path for path in scenario.iterdir() if path.is_dir())
    if len(agents) < 2:
        return [], []
    ego_dir, collaborator_dir = agents[:2]
    frame_ids = sorted(
        (path.stem for path in ego_dir.glob("*.yaml") if path.stem.isdigit()),
        key=int,
    )
    speeds: list[float] = []
    frames: list[dict[str, tuple[float, float, float]] | None] = []
    previous: dict[str, tuple[float, float, float]] = {}
    for frame_id in frame_ids:
        ego = load_yaml(ego_dir / f"{frame_id}.yaml")
        collaborator = load_yaml(collaborator_dir / f"{frame_id}.yaml")
        ego_pose, collaborator_pose = ego.get("lidar_pose"), collaborator.get("lidar_pose")
        if not ego_pose or not collaborator_pose:
            previous = {}
            frames.append(None)
            continue
        distance = math.hypot(
            float(collaborator_pose[0]) - float(ego_pose[0]),
            float(collaborator_pose[1]) - float(ego_pose[1]),
        )
        if collaboration_gate_m > 0 and distance > collaboration_gate_m:
            previous = {}
            frames.append(None)
            continue

        ego_x, ego_y, ego_yaw = (
            float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[4])
        )
        theta = math.radians(-ego_yaw)
        cosine, sine = math.cos(theta), math.sin(theta)
        current: dict[str, tuple[float, float, float]] = {}
        for identity, vehicle in (ego.get("vehicles") or {}).items():
            if not isinstance(vehicle, dict):
                continue
            pose = object_pose(vehicle)
            if pose is None:
                continue
            world_x, world_y, yaw = pose
            delta_x, delta_y = world_x - ego_x, world_y - ego_y
            local_x = delta_x * cosine - delta_y * sine
            local_y = delta_x * sine + delta_y * cosine
            if not (x_range[0] <= local_x < x_range[1]
                    and y_range[0] <= local_y < y_range[1]):
                continue
            current[str(identity)] = (world_x, world_y, yaw)
        for identity in current.keys() & previous.keys():
            old_x, old_y, _ = previous[identity]
            new_x, new_y, _ = current[identity]
            speeds.append(
                math.hypot(new_x - old_x, new_y - old_y) / frame_interval
            )
        previous = current
        frames.append(current)
    return speeds, frames


def predicted_loss(speeds: list[float], gaps: list[int], radius: float,
                   frame_interval: float) -> dict[int, float]:
    velocity = np.asarray(speeds, dtype=float)
    output = {}
    for gap in gaps:
        if gap == 0:
            output[gap] = 0.0
        else:
            threshold = radius / (gap * frame_interval)
            output[gap] = float((velocity > threshold).mean())
    return output


def geometric_survival(
    scenes: list[list[dict[str, tuple[float, float, float]] | None]],
    gaps: list[int], displacement_m: float, yaw_deg: float,
) -> dict[int, float]:
    survival: dict[int, list[bool]] = {gap: [] for gap in gaps}
    for frames in scenes:
        for index, current in enumerate(frames):
            if current is None:
                continue
            for gap in gaps:
                if gap == 0:
                    survival[gap].extend(True for _ in current)
                    continue
                if index < gap or frames[index - gap] is None:
                    continue
                old = frames[index - gap]
                assert old is not None
                for identity in current.keys() & old.keys():
                    old_x, old_y, old_yaw = old[identity]
                    new_x, new_y, new_yaw = current[identity]
                    survival[gap].append(
                        math.hypot(new_x - old_x, new_y - old_y)
                        <= displacement_m
                        and wrapped_yaw_error(new_yaw - old_yaw) <= yaw_deg
                    )
    return {
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
    parser.add_argument("--collaboration-gate", type=float, default=51.0)
    parser.add_argument("--x-range", default="-70.4,70.4")
    parser.add_argument("--y-range", default="-40,40")
    parser.add_argument("--gsr-displacement", type=float, default=0.5)
    parser.add_argument("--gsr-yaw-deg", type=float, default=5.0)
    args = parser.parse_args()

    exact = json.loads(args.exact_age_json.read_text(encoding="utf-8"))
    gaps = [int(value) for value in exact["gaps"]]
    x_range = tuple(float(value) for value in args.x_range.split(","))
    y_range = tuple(float(value) for value in args.y_range.split(","))
    if len(x_range) != 2 or len(y_range) != 2:
        raise ValueError("--x-range and --y-range require min,max")

    scene_rows, tracks = [], []
    for scenario in sorted(
        path for path in args.dataset_root.resolve().iterdir() if path.is_dir()
    ):
        speeds, frames = scene_motion(
            scenario,
            frame_interval=args.frame_interval,
            collaboration_gate_m=args.collaboration_gate,
            x_range=x_range,
            y_range=y_range,
        )
        if not speeds:
            continue
        prediction = predicted_loss(
            speeds, gaps, args.match_radius, args.frame_interval
        )
        scene_rows.append({
            "scenario": scenario.name,
            "speed_samples": len(speeds),
            "predicted_loss": {str(k): prediction[k] for k in gaps},
        })
        tracks.append(frames)
    if not scene_rows:
        raise RuntimeError("no usable synchronized scenarios")
    prediction = np.asarray([
        np.mean([row["predicted_loss"][str(gap)] for row in scene_rows])
        for gap in gaps
    ])
    gsr = geometric_survival(
        tracks, gaps, args.gsr_displacement, args.gsr_yaw_deg
    )
    output = {
        "gaps": gaps,
        "predicted_loss": prediction.tolist(),
        "geometric_survival_rate": {str(k): gsr[k] for k in gaps},
        "metrics": {},
        "per_scenario_prediction": scene_rows,
        "protocol": {
            "frame_interval_s": args.frame_interval,
            "match_radius_m": args.match_radius,
            "collaboration_gate_m": args.collaboration_gate,
            "x_range_m": x_range,
            "y_range_m": y_range,
            "gsr_displacement_m": args.gsr_displacement,
            "gsr_yaw_deg": args.gsr_yaw_deg,
            "detector_fitting": "none"
        },
    }
    for metric in ("ap_50", "ap_70"):
        values = np.asarray([exact["curves"][str(gap)][metric] for gap in gaps])
        relative_loss = 1.0 - values / values[0]
        output["metrics"][metric] = {
            "values": values.tolist(),
            "relative_loss": relative_loss.tolist(),
            "pearson_r": float(np.corrcoef(prediction, relative_loss)[0, 1]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

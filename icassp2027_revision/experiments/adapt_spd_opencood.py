#!/usr/bin/env python3
"""Create a symlink-backed OpenCOOD view of the paired SPD sequence."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spd_to_opv2v_adapter import read_transform, yaw_from_R  # noqa: E402


VEHICLE_TYPES = {"Car", "Van", "Truck", "Bus"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pose_vehicle(root: Path, frame: str):
    nw = read_transform(root / "vehicle-side/calib/novatel_to_world" / f"{frame}.json")
    ln = read_transform(root / "vehicle-side/calib/lidar_to_novatel" / f"{frame}.json")
    if nw is None or ln is None:
        return None
    r_nw, t_nw = nw; r_ln, t_ln = ln
    xyz = r_nw @ t_ln + t_nw
    yaw = yaw_from_R(r_nw @ r_ln)
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), 0.0, float(yaw), 0.0], r_nw, t_nw, r_ln, t_ln


def pose_infra(root: Path, frame: str):
    vw = read_transform(root / "infrastructure-side/calib/virtuallidar_to_world" / f"{frame}.json")
    if vw is None:
        return None
    rotation, translation = vw
    return [float(translation[0]), float(translation[1]), float(translation[2]),
            0.0, float(yaw_from_R(rotation)), 0.0]


def world_objects(root: Path, vehicle_frame: str, vehicle_tf):
    pose, r_nw, t_nw, r_ln, t_ln = vehicle_tf
    labels = load_json(root / "cooperative/label" / f"{vehicle_frame}.json")
    objects = {}
    for obj in labels:
        if obj.get("type") not in VEHICLE_TYPES:
            continue
        loc = obj.get("3d_location") or {}
        dims = obj.get("3d_dimensions") or {}
        try:
            local = np.array([float(loc["x"]), float(loc["y"]), float(loc["z"])])
            length, width, height = float(dims["l"]), float(dims["w"]), float(dims["h"])
            rotation = float(obj.get("rotation", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        world = r_nw @ (r_ln @ local + t_ln) + t_nw
        object_id = str(obj.get("track_id") or obj.get("token") or len(objects))
        objects[object_id] = {
            "angle": [0.0, math.degrees(rotation) + pose[4], 0.0],
            "center": [0.0, 0.0, height / 2.0],
            "extent": [length / 2.0, width / 2.0, height / 2.0],
            "location": [float(world[0]), float(world[1]), float(world[2])],
            "speed": 0.0,
        }
    return objects


def link(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spd-root", required=True)
    parser.add_argument("--output", default=str(
        ROOT / "icassp2027_revision/dataset_views/spd_opencood"))
    args = parser.parse_args()
    root = Path(args.spd_root).resolve(); output = Path(args.output).resolve()
    records = load_json(root / "cooperative/data_info.json")
    scenario = output / "0058"
    written = 0
    for record in records:
        vf = str(record["vehicle_frame"]); inf = str(record["infrastructure_frame"])
        vehicle_pcd = root / "vehicle-side/velodyne" / f"{vf}.pcd"
        infra_pcd = root / "infrastructure-side/velodyne" / f"{inf}.pcd"
        coop_label = root / "cooperative/label" / f"{vf}.json"
        vehicle_tf = pose_vehicle(root, vf); infra_pose = pose_infra(root, inf)
        if not (vehicle_pcd.exists() and infra_pcd.exists() and coop_label.exists()
                and vehicle_tf is not None and infra_pose is not None):
            continue
        objects = world_objects(root, vf, vehicle_tf)
        for agent, pcd, pose in (
            ("1", vehicle_pcd, vehicle_tf[0]),
            ("-1", infra_pcd, infra_pose),
        ):
            agent_dir = scenario / agent; agent_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "ego_speed": 0.0,
                "lidar_pose": pose,
                "true_ego_pos": pose,
                "vehicles": objects,
            }
            (agent_dir / f"{vf}.yaml").write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            link(pcd, agent_dir / f"{vf}.pcd")
        written += 1
    manifest = {
        "sequence": "0058", "paired_frames": written,
        "object_types": sorted(VEHICLE_TYPES),
        "note": "PCDs are symlinks; cooperative boxes are transformed from vehicle lidar to world.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())

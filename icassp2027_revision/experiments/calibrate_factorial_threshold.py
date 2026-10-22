#!/usr/bin/env python3
"""Calibrate C=0 fixed thresholds for the A/B/C factorial experiment.

Features are extracted once per frame.  Each candidate threshold maintains an
independent cache trajectory, so calibration accounts for the feedback between
selection decisions and future receiver-cache residuals.  No detection
post-processing is run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import evaluate_learned_direct as ev


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(
        ev.REVISION / "experiments/point_pillar_internal.yaml"))
    parser.add_argument("--checkpoint", default=str(
        ev.REVISION / "experiments/point_pillar_run/best.pth"))
    parser.add_argument("--dataset-dir", default=str(
        ev.REVISION / "dataset_views/opv2v_internal/validate"))
    parser.add_argument("--combo", choices=("000", "100", "010", "110"),
                        required=True)
    parser.add_argument("--target", type=float, default=0.10)
    parser.add_argument("--candidates", default=(
        "0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25,0.30,0.40,0.60,1.0"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--com-range", type=float, default=70.0)
    args = parser.parse_args()

    a, b, c = (int(bit) for bit in args.combo)
    if c != 0:
        raise ValueError("threshold calibration is only defined for C=0")
    transport = bool(a)
    selection_mode = "direct" if b else "tx_delta"
    candidates = [float(item) for item in args.candidates.split(",")]

    hypes = ev.yaml_utils.load_yaml(args.config, None)
    ev.dataset_module.COM_RANGE = args.com_range
    hypes["validate_dir"] = str(Path(args.dataset_dir).resolve())
    dataset = ev.build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False,
        collate_fn=dataset.collate_batch_test,
    )
    model = ev.train_utils.create_model(hypes).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])

    cache_bank = {threshold: ev.fresh_caches(0.0) for threshold in candidates}
    transmitted = {threshold: 0 for threshold in candidates}
    available = 0
    previous_scenario = None
    frames = 0

    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.max_frames and index >= args.max_frames:
                break
            scenario, timestamp_key, pose = ev.index_context(dataset, index)
            if scenario != previous_scenario:
                cache_bank = {
                    threshold: ev.fresh_caches(0.0) for threshold in candidates
                }
                previous_scenario = scenario
            ego_pose = ev.pose_to_matrix(
                float(pose[0]), float(pose[1]), math.radians(float(pose[4])),
                device=torch.device("cuda"),
            )
            batch = ev.train_utils.to_device(batch, torch.device("cuda"))
            data = batch["ego"]
            processed = data["processed_lidar"]
            state = {
                "voxel_features": processed["voxel_features"],
                "voxel_coords": processed["voxel_coords"],
                "voxel_num_points": processed["voxel_num_points"],
                "record_len": data["record_len"],
            }
            state = model.pillar_vfe(state)
            state = model.scatter(state)
            x = state["spatial_features"]
            n_agents = min(2, int(data["record_len"][0]))
            if n_agents < 2:
                frames += 1
                continue
            for level, block in enumerate(model.backbone.blocks):
                x = block(x)
                if model.backbone.compress and level < len(model.backbone.compression_modules):
                    x = model.backbone.compression_modules[level](x)
                fresh = x[1]
                available += fresh.numel()
                for threshold in candidates:
                    out = cache_bank[threshold][level].step(
                        fresh,
                        ego_pose,
                        int(timestamp_key),
                        args.target,
                        transport=transport,
                        selection_mode=selection_mode,
                        selection_rule="threshold",
                        selection_threshold=threshold,
                    )
                    transmitted[threshold] += int(out.selected_regions.sum()) * (
                        cache_bank[threshold][level].region_h
                        * cache_bank[threshold][level].region_w
                        * fresh.shape[0]
                    )
            frames += 1

    rows = [
        {
            "threshold": threshold,
            "payload_fraction": transmitted[threshold] / available if available else 0.0,
        }
        for threshold in candidates
    ]
    best = min(rows, key=lambda row: abs(row["payload_fraction"] - args.target))
    result = {
        "combo": args.combo,
        "transport": transport,
        "selection_mode": selection_mode,
        "target_payload_fraction": args.target,
        "frames": frames,
        "candidates": rows,
        "selected": best,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "selection_protocol": "validation-only; cache trajectory per threshold",
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

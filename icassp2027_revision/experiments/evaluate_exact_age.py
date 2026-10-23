#!/usr/bin/env python3
"""Exact-age learned-feature degradation on one common evaluation frame set."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "icassp2027_revision"
OPENCOOD = ROOT / "baselines/code/OpenCOOD"
sys.path.insert(0, str(OPENCOOD))
sys.path.insert(0, str(REVISION / "cache"))
sys.path.insert(0, str(REVISION / "experiments"))

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
import opencood.data_utils.datasets as dataset_module  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.utils import eval_utils  # noqa: E402
from coordinate_cache import pose_to_matrix, warp_bev  # noqa: E402
from evaluate_learned_direct import index_context  # noqa: E402,F401


def extract_levels(model, data):
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
    levels = []
    for level, block in enumerate(model.backbone.blocks):
        x = block(x)
        if model.backbone.compress and level < len(model.backbone.compression_modules):
            x = model.backbone.compression_modules[level](x)
        levels.append(x[:min(2, int(data["record_len"][0]))])
    return levels


def detection_output(model, current_levels, collaborator_levels):
    ups = []
    for level, (current, collaborator) in enumerate(
            zip(current_levels, collaborator_levels)):
        pair = torch.stack((current[0], collaborator), dim=0)
        fused = model.backbone.fuse_modules[level](
            pair, torch.tensor([2], device=pair.device)
        )
        ups.append(model.backbone.deblocks[level](fused))
    feature = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
    if len(model.backbone.deblocks) > len(model.backbone.blocks):
        feature = model.backbone.deblocks[-1](feature)
    return {"psm": model.cls_head(feature), "rm": model.reg_head(feature)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(
        REVISION / "experiments/point_pillar_internal.yaml"))
    parser.add_argument("--checkpoint", default=str(
        REVISION / "experiments/point_pillar_run/best.pth"))
    parser.add_argument("--dataset-dir", default=str(
        REVISION / "dataset_views/opv2v_internal/test"))
    parser.add_argument("--dataset-label", default="opv2v_internal_test")
    parser.add_argument("--gaps", default="0,1,2,4,6,8")
    parser.add_argument("--output", default=str(
        REVISION / "experiments/learned_exact_age.json"))
    parser.add_argument("--com-range", type=float, default=70.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    args = parser.parse_args()

    gaps = sorted({int(x) for x in args.gaps.split(",")})
    kmax = max(gaps)
    hypes = yaml_utils.load_yaml(args.config, None)
    dataset_module.COM_RANGE = args.com_range
    hypes["validate_dir"] = str(Path(args.dataset_dir).resolve())
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False,
        collate_fn=dataset.collate_batch_test,
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model = train_utils.create_model(hypes).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])

    stats = {gap: {iou: {"tp": [], "fp": [], "gt": 0, "score": []}
                   for iou in (0.3, 0.5, 0.7)} for gap in gaps}
    histories = [deque(maxlen=kmax + 1) for _ in range(3)]
    pose_history = deque(maxlen=kmax + 1)
    previous_scenario = None
    n_common = 0
    random.seed(2027); np.random.seed(2027); torch.manual_seed(2027)

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.max_frames and index >= args.max_frames:
                break
            scenario, timestamp, pose = index_context(dataset, index)
            if scenario != previous_scenario:
                histories = [deque(maxlen=kmax + 1) for _ in range(3)]
                pose_history = deque(maxlen=kmax + 1)
                previous_scenario = scenario
            batch = train_utils.to_device(batch, device)
            levels = extract_levels(model, batch["ego"])
            if any(level.shape[0] < 2 for level in levels):
                histories = [deque(maxlen=kmax + 1) for _ in range(3)]
                pose_history = deque(maxlen=kmax + 1)
                continue
            current_pose = pose_to_matrix(
                float(pose[0]), float(pose[1]), math.radians(float(pose[4])),
                device=device,
            )
            for level, values in enumerate(levels):
                histories[level].append(values[1].detach())
            pose_history.append(current_pose.detach())
            if len(pose_history) < kmax + 1:
                continue
            n_common += 1
            for gap in gaps:
                source_pose = pose_history[-gap - 1]
                collaborators = [
                    warp_bev(
                        histories[level][-gap - 1], source_pose, current_pose,
                        (-140.8, 140.8), (-40.0, 40.0), mode="bilinear",
                    ) for level in range(3)
                ]
                output = detection_output(model, levels, collaborators)
                pred, score, gt = dataset.post_process(batch, {"ego": output})
                for iou in stats[gap]:
                    eval_utils.caluclate_tp_fp(
                        pred, score, gt, stats[gap], iou
                    )

    results = {
        str(gap): {
            f"ap_{int(iou * 100)}": eval_utils.calculate_ap(
                copy.deepcopy(stats[gap]), iou, True
            )[0] for iou in (0.3, 0.5, 0.7)
        } for gap in gaps
    }
    payload = {
        "dataset_label": args.dataset_label,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "gaps": gaps,
        "n_common_frames": n_common,
        "device": str(device),
        "curves": results,
        "protocol": "exact cache age k; all k evaluated on one common frame set",
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

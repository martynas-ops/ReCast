#!/usr/bin/env python3
"""Evaluate Direct and Periodic over multiple budgets in one feature pass."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.cm as mpl_cm
from torch.utils.data import DataLoader

# Matplotlib 3.10 removed ``cm.get_cmap`` while this OpenCOOD snapshot still
# imports it.  Keep the evaluator compatible without changing the baseline.
if not hasattr(mpl_cm, "get_cmap"):
    mpl_cm.get_cmap = lambda name: matplotlib.colormaps[name]
if importlib.util.find_spec("timm") is None:
    # PointPillar inference does not use timm; the legacy train utility imports
    # it unconditionally only for an optional scheduler.
    sys.modules["timm"] = types.ModuleType("timm")

ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "icassp2027_revision"
OPENCOOD = ROOT / "baselines/code/OpenCOOD"
sys.path.insert(0, str(OPENCOOD)); sys.path.insert(0, str(REVISION / "cache"))
sys.path.insert(0, str(REVISION / "experiments"))

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
import opencood.data_utils.datasets as dataset_module  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.utils import eval_utils  # noqa: E402
from coordinate_cache import CoordinateConsistentRegionCache, pose_to_matrix  # noqa: E402
from evaluate_exact_age import extract_levels, detection_output  # noqa: E402
from evaluate_learned_direct import index_context  # noqa: E402,F401


def caches(age_weight=0.0):
    return [
        CoordinateConsistentRegionCache((-140.8, 140.8), (-40, 40), (4, 4), age_weight),
        CoordinateConsistentRegionCache((-140.8, 140.8), (-40, 40), (2, 2), age_weight),
        CoordinateConsistentRegionCache((-140.8, 140.8), (-40, 40), (1, 1), age_weight),
    ]


def ego_output(model, levels):
    ups = [model.backbone.deblocks[i](level[0].unsqueeze(0))
           for i, level in enumerate(levels)]
    feature = torch.cat(ups, dim=1)
    return {"psm": model.cls_head(feature), "rm": model.reg_head(feature)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REVISION / "experiments/point_pillar_internal.yaml"))
    parser.add_argument("--checkpoint", default=str(REVISION / "experiments/point_pillar_run/best.pth"))
    parser.add_argument("--dataset-dir", default=str(REVISION / "dataset_views/opv2v_internal/test"))
    parser.add_argument("--dataset-label", default="opv2v_internal_test")
    parser.add_argument("--budgets", default="0.02,0.05,0.10,0.25,0.50")
    parser.add_argument("--methods", default="periodic,direct")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--com-range", type=float, default=70.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--score-mode", choices=("mean", "max", "saliency", "normalized"), default="mean")
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    parser.add_argument("--output", default=str(REVISION / "experiments/learned_budget_curve.json"))
    args = parser.parse_args()
    if not 0.0 <= args.blend_alpha <= 1.0:
        raise ValueError("--blend-alpha must be in [0,1]")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    budgets = [float(x) for x in args.budgets.split(",")]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    allowed = {"periodic", "direct", "direct_age", "direct_nowarp"}
    if set(methods) - allowed:
        raise ValueError(f"unknown methods: {sorted(set(methods)-allowed)}")

    hypes = yaml_utils.load_yaml(args.config, None)
    dataset_module.COM_RANGE = args.com_range
    hypes["validate_dir"] = str(Path(args.dataset_dir).resolve())
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False,
                        collate_fn=dataset.collate_batch_test)
    model = train_utils.create_model(hypes).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])

    keys = [(method, budget) for method in methods for budget in budgets]
    stats = {key: {iou: {"tp": [], "fp": [], "gt": 0, "score": []}
                   for iou in (0.3, 0.5, 0.7)} for key in keys}
    states = {key: caches(0.3 if key[0] == "direct_age" else 0.0) for key in keys}
    baseline_stats = {
        name: {iou: {"tp": [], "fp": [], "gt": 0, "score": []}
               for iou in (0.3, 0.5, 0.7)} for name in ("ego", "full")
    }
    tx = {key: 0 for key in keys}; available = {key: 0 for key in keys}
    previous_scenario = None; sequence_step = 0
    random.seed(2027); np.random.seed(2027); torch.manual_seed(2027)

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.max_frames and index >= args.max_frames:
                break
            scenario, timestamp, pose = index_context(dataset, index)
            if scenario != previous_scenario:
                states = {key: caches(0.3 if key[0] == "direct_age" else 0.0)
                          for key in keys}
                previous_scenario = scenario; sequence_step = 0
            batch = train_utils.to_device(batch, device)
            levels = extract_levels(model, batch["ego"])
            ego_prediction = (ego_output(model, levels)
                              if args.include_baselines or args.blend_alpha < 1.0 else None)
            current_pose = pose_to_matrix(
                float(pose[0]), float(pose[1]), math.radians(float(pose[4])),
                device=device,
            )
            if args.include_baselines:
                outputs = {"ego": ego_prediction}
                outputs["full"] = (
                    detection_output(model, levels, [level[1] for level in levels])
                    if all(level.shape[0] >= 2 for level in levels)
                    else outputs["ego"]
                )
                for name, output in outputs.items():
                    pred, score, gt = dataset.post_process(batch, {"ego": output})
                    for iou in baseline_stats[name]:
                        eval_utils.caluclate_tp_fp(
                            pred, score, gt, baseline_stats[name], iou
                        )
            for method, budget in keys:
                key = (method, budget)
                if any(level.shape[0] < 2 for level in levels):
                    output = ego_output(model, levels)
                else:
                    served = []
                    for level, values in enumerate(levels):
                        force = None
                        if method == "periodic":
                            force = sequence_step % max(1, round(1.0 / budget)) == 0
                        out = states[key][level].step(
                            values[1], current_pose, int(timestamp), budget,
                            force_refresh=force,
                            transport=(method != "direct_nowarp"),
                            score_mode=args.score_mode,
                        )
                        served.append(out.feature)
                        tx[key] += int(out.selected_regions.sum()) * (
                            states[key][level].region_h * states[key][level].region_w
                            * values.shape[1]
                        )
                        available[key] += values[1].numel()
                    output = detection_output(model, levels, served)
                if args.blend_alpha < 1.0 and ego_prediction is not None:
                    output = {
                        name: args.blend_alpha * output[name]
                              + (1.0 - args.blend_alpha) * ego_prediction[name]
                        for name in ("psm", "rm")
                    }
                pred, score, gt = dataset.post_process(batch, {"ego": output})
                for iou in stats[key]:
                    eval_utils.caluclate_tp_fp(pred, score, gt, stats[key], iou)
            sequence_step += 1

    curves = {}
    for method in methods:
        curves[method] = {}
        for budget in budgets:
            key = (method, budget)
            curves[method][str(budget)] = {
                **{f"ap_{int(iou*100)}": eval_utils.calculate_ap(
                    copy.deepcopy(stats[key]), iou, True)[0]
                   for iou in (0.3, 0.5, 0.7)},
                "feature_value_fraction_realized": (
                    tx[key] / available[key] if available[key] else 0.0
                ),
            }
    payload = {
        "dataset_label": args.dataset_label, "budgets": budgets,
        "frames": min(len(dataset), args.max_frames) if args.max_frames else len(dataset),
        "curves": curves,
        "communication_range_m": args.com_range,
        "device": str(device),
        "score_mode": args.score_mode,
        "blend_alpha": args.blend_alpha,
        "baselines": ({
            name: {f"ap_{int(iou*100)}": eval_utils.calculate_ap(
                copy.deepcopy(value), iou, True)[0]
                   for iou in (0.3, 0.5, 0.7)}
            for name, value in baseline_stats.items()
        } if args.include_baselines else {}),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

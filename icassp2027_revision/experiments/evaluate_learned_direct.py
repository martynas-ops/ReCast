#!/usr/bin/env python3
"""Evaluate Full, Ego-only, and Direct on held-out scenes with learned BEV features."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import matplotlib
import matplotlib.cm as mpl_cm
if not hasattr(mpl_cm, "get_cmap"):
    mpl_cm.get_cmap = matplotlib.colormaps.get_cmap
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
OPENCOOD = ROOT / "baselines/code/OpenCOOD"
REVISION = ROOT / "icassp2027_revision"
sys.path.insert(0, str(OPENCOOD))
sys.path.insert(0, str(REVISION / "cache"))

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
import opencood.data_utils.datasets as dataset_module  # noqa: E402
from opencood.data_utils.datasets import basedataset, build_dataset  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.utils import eval_utils, pcd_utils  # noqa: E402
from coordinate_cache import CoordinateConsistentRegionCache, pose_to_matrix  # noqa: E402


def compatible_pcd_to_np(path):
    """OpenCOOD PCD reader with a zero-intensity fallback for SPD files."""
    cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(cloud.points, dtype=np.float32)
    colors = np.asarray(cloud.colors, dtype=np.float32)
    intensity = colors[:, :1] if colors.shape[0] == xyz.shape[0] else np.zeros(
        (xyz.shape[0], 1), dtype=np.float32
    )
    return np.hstack((xyz, intensity)).astype(np.float32, copy=False)


pcd_utils.pcd_to_np = compatible_pcd_to_np

def compatible_read_pcd(path):
    return compatible_pcd_to_np(path), None

pcd_utils.read_pcd = compatible_read_pcd


def robust_retrieve_base_data(self, idx, cur_ego_pose_flag=True):
    scenario_index = next(i for i, end in enumerate(self.len_record) if idx < end)
    scenario = self.scenario_database[scenario_index]
    local_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
    timestamp = self.return_timestamp_key(scenario, local_index)
    active = OrderedDict((cid, cav) for cid, cav in scenario.items() if timestamp in cav)
    ego = self.calc_dist_to_ego(active, timestamp)
    data = OrderedDict()
    for cid, cav in active.items():
        data[cid] = OrderedDict()
        data[cid]["ego"] = cav["ego"]
        data[cid]["time_delay"] = 0
        data[cid]["params"] = self.reform_param(
            cav, ego, timestamp, timestamp, cur_ego_pose_flag
        )
        data[cid]["lidar_np"] = pcd_utils.pcd_to_np(cav[timestamp]["lidar"])
    return data


basedataset.BaseDataset.retrieve_base_data = robust_retrieve_base_data


def index_context(dataset, index: int) -> tuple[int, str, list[float]]:
    # OPV2V/V2XSet expose temporal scenario databases. DAIR-V2X uses a
    # flat synchronized pair list; the fixed pilot split is ordered by frame id.
    if not hasattr(dataset, "len_record"):
        base = dataset.retrieve_base_data(index)
        return 0, str(index), base[0]["params"]["lidar_pose"]
    scenario_index = next(i for i, end in enumerate(dataset.len_record) if index < end)
    scenario = dataset.scenario_database[scenario_index]
    local_index = index if scenario_index == 0 else index - dataset.len_record[scenario_index - 1]
    timestamp = dataset.return_timestamp_key(scenario, local_index)
    ego = next(cav for cav in scenario.values() if cav["ego"])
    params = yaml_utils.load_yaml(ego[timestamp]["yaml"])
    return scenario_index, timestamp, params["lidar_pose"]


def learned_forward(model, data, mode: str, caches, ego_pose, timestamp: int,
                    budget: float, sequence_step: int, threshold=None,
                    score_mode="mean"):
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
    ups = []
    transmitted_values = 0
    available_values = 0
    # Evaluate exactly one collaborator for all modes. This avoids giving Full
    # more agents than the cache policy and matches the paper's stated scope.
    n_agents = min(2, int(data["record_len"][0]))
    for level, block in enumerate(model.backbone.blocks):
        x = block(x)
        if model.backbone.compress and level < len(model.backbone.compression_modules):
            x = model.backbone.compression_modules[level](x)
        level_x = x[:n_agents]
        if n_agents > 1:
            available_values += level_x[1].numel()
        if mode == "ego" or n_agents == 1:
            fused = level_x[0].unsqueeze(0)
        elif mode == "full":
            transmitted_values += level_x[1].numel()
            fused = model.backbone.fuse_modules[level](
                level_x, torch.tensor([n_agents], device=level_x.device)
            )
        elif mode in ("direct", "direct_nowarp", "tx_delta", "random_b", "oldest_first") or mode.startswith("factor_"):
            if mode.startswith("factor_"):
                bits = mode.removeprefix("factor_")
                if len(bits) != 3 or any(bit not in "01" for bit in bits):
                    raise ValueError(f"invalid factorial mode: {mode}")
                transport = bits[0] == "1"
                selection_mode = "direct" if bits[1] == "1" else "tx_delta"
                selection_rule = "topb" if bits[2] == "1" else "threshold"
            else:
                transport = mode != "direct_nowarp"
                selection_mode = {
                    "tx_delta": "tx_delta",
                    "random_b": "random",
                    "oldest_first": "oldest",
                }.get(mode, "direct")
                selection_rule = "topb"
            cache_out = caches[level].step(
                level_x[1], ego_pose, timestamp, budget,
                transport=transport,
                selection_mode=selection_mode,
                selection_rule=selection_rule,
                selection_threshold=threshold,
                score_mode=score_mode,
            )
            served = cache_out.feature
            transmitted_values += int(cache_out.selected_regions.sum()) * (
                caches[level].region_h * caches[level].region_w * level_x.shape[1]
            )
            pair = torch.stack((level_x[0], served), dim=0)
            fused = model.backbone.fuse_modules[level](
                pair, torch.tensor([2], device=pair.device)
            )
        elif mode == "periodic":
            period = max(1, round(1.0 / budget))
            cache_out = caches[level].step(
                level_x[1], ego_pose, timestamp, budget,
                force_refresh=(sequence_step % period == 0),
            )
            served = cache_out.feature
            transmitted_values += int(cache_out.selected_regions.sum()) * (
                caches[level].region_h * caches[level].region_w * level_x.shape[1]
            )
            pair = torch.stack((level_x[0], served), dim=0)
            fused = model.backbone.fuse_modules[level](
                pair, torch.tensor([2], device=pair.device)
            )
        else:
            raise ValueError(mode)
        if model.backbone.deblocks:
            ups.append(model.backbone.deblocks[level](fused))
        else:
            ups.append(fused)
    feature = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
    if len(model.backbone.deblocks) > len(model.backbone.blocks):
        feature = model.backbone.deblocks[-1](feature)
    return ({"psm": model.cls_head(feature), "rm": model.reg_head(feature)},
            transmitted_values, available_values)


def fresh_caches(age_weight: float):
    # Physical region width is 3.2 m at every feature scale.
    return [
        CoordinateConsistentRegionCache((-140.8, 140.8), (-40, 40), (4, 4), age_weight),
        CoordinateConsistentRegionCache((-140.8, 140.8), (-40, 40), (2, 2), age_weight),
        CoordinateConsistentRegionCache((-140.8, 140.8), (-40, 40), (1, 1), age_weight),
    ]


def evaluate(mode, model, dataset, loader, budget, age_weight, max_frames=0,
             threshold=None, score_mode="mean"):
    stats = {iou: {"tp": [], "fp": [], "gt": 0, "score": []}
             for iou in (0.3, 0.5, 0.7)}
    caches = fresh_caches(age_weight)
    previous_scenario = None
    sequence_step = 0
    transmitted_values = 0
    available_values = 0
    per_scenario = {}
    for index, batch in enumerate(loader):
        if max_frames and index >= max_frames:
            break
        scenario, timestamp_key, pose = index_context(dataset, index)
        if scenario != previous_scenario:
            caches = fresh_caches(age_weight)
            previous_scenario = scenario
            sequence_step = 0
        ego_pose = pose_to_matrix(
            float(pose[0]), float(pose[1]), math.radians(float(pose[4])),
            device=torch.device("cuda"),
        )
        batch = train_utils.to_device(batch, torch.device("cuda"))
        output, tx_values, full_values = learned_forward(
            model, batch["ego"], mode, caches, ego_pose,
            int(timestamp_key), budget, sequence_step, threshold, score_mode,
        )
        transmitted_values += tx_values
        available_values += full_values
        pred, score, gt = dataset.post_process(batch, {"ego": output})
        scenario_stat = per_scenario.setdefault(
            str(scenario),
            {iou: {"tp": [], "fp": [], "gt": 0, "score": []}
             for iou in (0.3, 0.5, 0.7)},
        )
        for iou in stats:
            eval_utils.caluclate_tp_fp(pred, score, gt, stats, iou)
            eval_utils.caluclate_tp_fp(pred, score, gt, scenario_stat, iou)
        sequence_step += 1
    result = {f"ap_{int(iou * 100)}": eval_utils.calculate_ap(
        copy.deepcopy(stats), iou, True
    )[0] for iou in stats}
    result["frames"] = min(len(dataset), max_frames) if max_frames else len(dataset)
    result["budget_requested"] = (
        budget if mode in ("direct", "direct_nowarp", "periodic", "tx_delta", "random_b", "oldest_first") or mode.startswith("factor_")
        else (0 if mode == "ego" else 1)
    )
    result["feature_value_fraction_realized"] = (
        transmitted_values / available_values if available_values else 0.0
    )
    result["transmitted_feature_values"] = transmitted_values
    result["available_feature_values"] = available_values
    result["payload_bytes_fp16"] = 2 * transmitted_values
    result["full_reference_bytes_fp16"] = 2 * available_values
    result["per_scenario"] = {
        key: {f"ap_{int(iou * 100)}": eval_utils.calculate_ap(
            copy.deepcopy(value), iou, True
        )[0] for iou in (0.3, 0.5, 0.7)}
        for key, value in per_scenario.items()
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(
        REVISION / "experiments/point_pillar_internal.yaml"))
    parser.add_argument("--checkpoint", default=str(
        REVISION / "experiments/point_pillar_run/best.pth"))
    parser.add_argument("--output", default=str(
        REVISION / "experiments/learned_policy_results.json"))
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--age-weight", type=float, default=0.0)
    parser.add_argument("--score-mode", choices=("mean", "max", "saliency", "normalized"),
                        default="mean")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--dataset-dir", default=str(
        REVISION / "dataset_views/opv2v_internal/test"))
    parser.add_argument("--dataset-label", default="opv2v_internal_test")
    parser.add_argument(
        "--modes", default="ego,full,periodic,tx_delta,random_b,oldest_first,direct,direct_nowarp",
        help="comma-separated result labels to evaluate",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="absolute region-score threshold for factorial modes with C=0",
    )
    parser.add_argument("--com-range", type=float, default=70.0)
    parser.add_argument(
        "--max-cav", type=int, default=None,
        help="maximum agents including ego; use 2 for paired one-collaborator audits",
    )
    parser.add_argument(
        "--seed", type=int, default=2027,
        help="region-selection seed; point preprocessing remains fixed at 2027",
    )
    args = parser.parse_args()

    hypes = yaml_utils.load_yaml(args.config, None)
    if args.max_cav is not None:
        if args.max_cav < 1:
            raise ValueError("--max-cav must be positive")
        hypes["train_params"]["max_cav"] = args.max_cav
    dataset_module.COM_RANGE = args.com_range
    hypes["validate_dir"] = str(Path(args.dataset_dir).resolve())
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False,
        collate_fn=dataset.collate_batch_test,
    )
    model = train_utils.create_model(hypes).cuda().eval()
    # Checkpoint was saved with NumPy 2, whereas the inference container uses
    # NumPy 1.24. The private import path changed, but the array pickle layout
    # used by this checkpoint remains compatible.
    if not hasattr(np, "_core"):
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])

    results = {}
    all_runs = (
        ("ego", "ego", 0.0),
        ("full", "full", 0.0),
        ("periodic", "periodic", 0.0),
        ("factor_000", "factor_000", 0.0),
        ("factor_100", "factor_100", 0.0),
        ("factor_010", "factor_010", 0.0),
        ("factor_110", "factor_110", 0.0),
        ("factor_001", "factor_001", 0.0),
        ("factor_101", "factor_101", 0.0),
        ("factor_011", "factor_011", 0.0),
        ("factor_111", "factor_111", 0.0),
        ("tx_delta", "tx_delta", 0.0),
        ("random_b", "random_b", 0.0),
        ("oldest_first", "oldest_first", 0.0),
        ("direct", "direct", args.age_weight),
        ("direct_nowarp", "direct_nowarp", args.age_weight),
    )
    requested = {item.strip() for item in args.modes.split(",") if item.strip()}
    known = {label for label, _, _ in all_runs}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    if args.threshold is None and any(
        label.startswith("factor_") and label.endswith("0") for label in requested
    ):
        raise ValueError("--threshold is required for factorial modes with C=0")
    runs = tuple(run for run in all_runs if run[0] in requested)
    with torch.no_grad():
        for label, mode, age_weight in runs:
            # OpenCOOD shuffles points even in test mode. Reset the single-
            # process RNGs so every policy receives identical voxel inputs.
            random.seed(2027)
            np.random.seed(2027)
            torch.manual_seed(args.seed)
            results[label] = evaluate(
                mode, model, dataset, loader, args.budget, age_weight,
                args.max_frames, args.threshold, args.score_mode,
            )
            print(label, json.dumps(results[label]), flush=True)
    results["protocol"] = {
        "dataset_label": args.dataset_label,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "training_split": "8 OPV2V train / 2 OPV2V validation scenarios",
        "scope": ("scene-disjoint internal feasibility on OPV2V; cross-dataset "
                  "zero-shot when dataset_label is not opv2v_internal_test"),
        "collaborators": 1,
        "max_cav": hypes["train_params"]["max_cav"],
        "global_sort_detections": True,
        "feature_region_m": 3.2,
        "input_seed": 2027,
        "selection_seed": args.seed,
        "age_weight": args.age_weight,
        "score_mode": args.score_mode,
        "threshold": args.threshold,
    }
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

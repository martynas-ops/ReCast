#!/usr/bin/env python3
"""Reproducible single-GPU trainer for the ICASSP internal scene split."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml
import open3d as o3d
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
OPENCOOD = ROOT / "baselines" / "code" / "OpenCOOD"
sys.path.insert(0, str(OPENCOOD))

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
import opencood.data_utils.datasets as dataset_module  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.data_utils.datasets import basedataset  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.utils import pcd_utils  # noqa: E402


def compatible_pcd_to_np(path):
    cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(cloud.points, dtype=np.float32)
    colors = np.asarray(cloud.colors, dtype=np.float32)
    intensity = colors[:, :1] if colors.shape[0] == xyz.shape[0] else np.zeros(
        (xyz.shape[0], 1), dtype=np.float32
    )
    return np.hstack((xyz, intensity)).astype(np.float32, copy=False)


pcd_utils.pcd_to_np = compatible_pcd_to_np


def robust_retrieve_base_data(self, idx, cur_ego_pose_flag=True):
    """OpenCOOD loader variant that skips collaborators missing this frame.

    The public scenario dumps are not perfectly rectangular: an ego timestamp
    can be absent for one collaborator. Upstream OpenCOOD indexes every CAV
    unconditionally and raises KeyError. We retain the ego sample and include
    only collaborators that actually contain the selected timestamp. No frame
    or label is synthesized.
    """
    scenario_index = next(
        i for i, accumulated in enumerate(self.len_record) if idx < accumulated
    )
    scenario = self.scenario_database[scenario_index]
    timestamp_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
    timestamp_key = self.return_timestamp_key(scenario, timestamp_index)
    active = OrderedDict(
        (cav_id, cav) for cav_id, cav in scenario.items()
        if timestamp_key in cav
    )
    ego_entries = [cav for cav in active.values() if cav["ego"]]
    if not ego_entries:
        raise KeyError(f"ego frame {timestamp_key} is missing")
    ego_cav = self.calc_dist_to_ego(active, timestamp_key)

    data = OrderedDict()
    for cav_id, cav in active.items():
        delay = self.time_delay_calculation(cav["ego"])
        if timestamp_index - delay <= 0:
            delay = timestamp_index
        delayed_index = max(0, timestamp_index - delay)
        delayed_key = self.return_timestamp_key(scenario, delayed_index)
        if delayed_key not in cav:
            continue
        data[cav_id] = OrderedDict()
        data[cav_id]["ego"] = cav["ego"]
        data[cav_id]["time_delay"] = delay
        data[cav_id]["params"] = self.reform_param(
            cav, ego_cav, timestamp_key, delayed_key, cur_ego_pose_flag
        )
        data[cav_id]["lidar_np"] = pcd_utils.pcd_to_np(cav[delayed_key]["lidar"])
    return data


# Keep the compatibility policy local to this revision; upstream files remain
# byte-for-byte unchanged.
basedataset.BaseDataset.retrieve_base_data = robust_retrieve_base_data


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(
        ROOT / "icassp2027_revision/experiments/point_pillar_internal.yaml"))
    parser.add_argument("--output", default=str(
        ROOT / "icassp2027_revision/experiments/point_pillar_run"))
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--com-range", type=float, default=70.0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    hypes = yaml_utils.load_yaml(args.config, None)
    dataset_module.COM_RANGE = args.com_range
    # Preserve the authored configuration verbatim. OpenCOOD's parser injects
    # derived NumPy arrays that SafeDumper intentionally cannot serialize.
    (output / "config.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    metadata = {
        "seed": args.seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "split_warning": (
            "Official OPV2V test scenes are repartitioned by whole scenario into "
            "8 train / 2 validation / 3 held-out test scenes. This run is an "
            "internal feasibility study, not an official OPV2V benchmark result."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    train_set = build_dataset(hypes, visualize=False, train=True)
    val_set = build_dataset(hypes, visualize=False, train=False)
    batch_size = int(hypes["train_params"]["batch_size"])
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True,
        collate_fn=train_set.collate_batch_train,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=val_set.collate_batch_train,
    )

    device = torch.device("cuda")
    model = train_utils.create_model(hypes).to(device)
    if args.init_checkpoint and not args.resume:
        init_state = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(init_state.get("model", init_state), strict=True)
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)

    start_epoch = 0
    best_val = float("inf")
    stale_epochs = 0
    latest = output / "latest.pth"
    if args.resume and latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint["best_val"]
        stale_epochs = checkpoint.get("stale_epochs", 0)

    epochs = int(hypes["train_params"]["epoches"])
    metrics_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        skipped_train = 0
        torch.cuda.reset_peak_memory_stats()
        for batch in train_loader:
            batch = train_utils.to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=not args.no_amp):
                pred = model(batch["ego"])
                loss = criterion(pred, batch["ego"]["label_dict"])
            if not torch.isfinite(loss):
                skipped_train += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        skipped_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = train_utils.to_device(batch, device)
                with torch.amp.autocast("cuda", enabled=not args.no_amp):
                    pred = model(batch["ego"])
                    loss = criterion(pred, batch["ego"]["label_dict"])
                if torch.isfinite(loss):
                    val_losses.append(float(loss.detach().cpu()))
                else:
                    skipped_val += 1

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        improved = val_loss < best_val - 1e-4
        if improved:
            best_val = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val,
            "stale_epochs": stale_epochs,
            "config": hypes,
        }
        atomic_save(state, latest)
        if improved:
            atomic_save(state, output / "best.pth")
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
            "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
            "improved": improved,
            "skipped_nonfinite_train_batches": skipped_train,
            "skipped_nonfinite_val_batches": skipped_val,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if stale_epochs >= args.patience:
            print(f"early stop after {stale_epochs} stale validation epochs", flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

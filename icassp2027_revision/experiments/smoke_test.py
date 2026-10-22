#!/usr/bin/env python3
"""One-batch OpenCOOD acceptance test; it does not train on the test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
OPENCOOD = ROOT / "baselines" / "code" / "OpenCOOD"
sys.path.insert(0, str(OPENCOOD))

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.tools import train_utils  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "icassp2027_revision" / "experiments" /
                    "point_pillar_smoke.yaml"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "icassp2027_revision" / "experiments" /
                    "smoke_result.json"),
    )
    args = parser.parse_args()

    hypes = yaml_utils.load_yaml(args.config, None)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_train,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    device = torch.device("cuda")
    model = train_utils.create_model(hypes).to(device).train()
    criterion = train_utils.create_loss(hypes)
    batch = train_utils.to_device(batch, device)

    torch.cuda.reset_peak_memory_stats()
    output = model(batch["ego"])
    loss = criterion(output, batch["ego"]["label_dict"])
    loss.backward()
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    result = {
        "status": "ok",
        "dataset_size": len(dataset),
        "record_len": batch["ego"]["record_len"].detach().cpu().tolist(),
        "psm_shape": list(output["psm"].shape),
        "rm_shape": list(output["rm"].shape),
        "loss": float(loss.detach().cpu()),
        "peak_cuda_gib": peak_gib,
        "note": "Forward/backward only; no optimizer step and no test-set training.",
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Plot matched-payload AP and gain-per-payload from the audit summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path,
        default=ROOT / "experiments/results_matrix/same_pilot_multibudget_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    styles = {
        "Direct": ("#d62728", "o", 2.5),
        "How2comm": ("#1b9e77", "s", 1.35),
        "Where2comm-compatible": ("#4c78a8", "^", 1.35),
        "CoSDH": ("#9467bd", "D", 1.35),
    }
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7.3, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "legend.fontsize": 6.4, "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    for column, dataset in enumerate(("OPV2V", "V2XSet")):
        for name, method in summary[dataset].items():
            color, marker, width = styles[name]
            payload = [point["payload_percent"] for point in method["points"]]
            label = "ReCAST" if name == "Direct" else name
            axes[0, column].plot(
                payload, [100 * point["ap50"] for point in method["points"]],
                marker=marker, color=color, lw=width, ms=4, label=label,
            )
            axes[1, column].plot(
                payload, [point["gpp"] for point in method["points"]],
                marker=marker, color=color, lw=width, ms=4, label=label,
            )
        axes[0, column].set_title(f"{dataset}: AP@0.5 at matched payload")
        axes[1, column].set_title(f"{dataset}: gain per payload")
        axes[0, column].set_ylabel("AP@0.5 (%)")
        axes[1, column].set_ylabel("Gain per payload")
        for row in range(2):
            axes[row, column].set_xlabel(
                "Realized payload (% of method-specific Full)"
            )
            axes[row, column].grid(True, alpha=.22, lw=.5)
            axes[row, column].legend(frameon=False)
    figure.tight_layout(w_pad=1.0, h_pad=.8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "fig_same_pilot_multibudget.pdf",
                   bbox_inches="tight")
    figure.savefig(args.output_dir / "fig_same_pilot_multibudget.png",
                   dpi=260, bbox_inches="tight")
    print(args.output_dir / "fig_same_pilot_multibudget.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

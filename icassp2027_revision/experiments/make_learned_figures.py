#!/usr/bin/env python3
"""Generate the compact learned-model curve figure from auditable JSON."""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--budget-json", type=Path,
                    default=ROOT / "experiments/learned_budget_curve.json")
parser.add_argument("--age-json", type=Path,
                    default=ROOT / "experiments/learned_exact_age.json")
parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
args = parser.parse_args()
budget = json.loads(args.budget_json.read_text(encoding="utf-8"))
age = json.loads(args.age_json.read_text(encoding="utf-8"))

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.1))

ax = axes[0]
for method, label, color, marker in (
    ("direct", "Direct", "#1677b8", "o"),
    ("periodic", "Periodic", "#d95f02", "s"),
):
    rows = list(budget["curves"][method].values())
    x = [100 * row["feature_value_fraction_realized"] for row in rows]
    y = [100 * row["ap_70"] for row in rows]
    ax.plot(x, y, marker=marker, lw=1.6, ms=4, color=color, label=label)
ego = 100 * budget["baselines"]["ego"]["ap_70"]
full = 100 * budget["baselines"]["full"]["ap_70"]
ax.axhline(ego, color="#666666", ls=":", lw=1.1, label="Ego")
ax.axhline(full, color="#1b9e77", ls="--", lw=1.1, label="Full")
ax.set_xlabel("Transmitted feature values (%)")
ax.set_ylabel("AP@0.7 (%)")
ax.set_title("(a) Bandwidth--accuracy")
ax.grid(True, alpha=.22, lw=.5)
ax.legend(frameon=False, ncol=2, loc="lower right")

ax = axes[1]
gaps = age["gaps"]
for metric, label, color, marker in (
    ("ap_50", "AP@0.5", "#1677b8", "o"),
    ("ap_70", "AP@0.7", "#7b3294", "^"),
):
    values = [100 * age["curves"][str(k)][metric] for k in gaps]
    ax.plot(gaps, values, marker=marker, lw=1.6, ms=4, color=color, label=label)
ax.set_xlabel("Exact cache age $k$ (frames)")
ax.set_ylabel("Average precision (%)")
ax.set_title("(b) Learned-feature staleness")
ax.set_xticks(gaps)
ax.grid(True, alpha=.22, lw=.5)
ax.legend(frameon=False)

fig.tight_layout(h_pad=1.1)
out = args.output_dir
out.mkdir(exist_ok=True)
fig.savefig(out / "fig_learned_curves.pdf", bbox_inches="tight")
fig.savefig(out / "fig_learned_curves.png", dpi=240, bbox_inches="tight")
print(out / "fig_learned_curves.pdf")

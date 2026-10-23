#!/usr/bin/env python3
"""Generate the measured CPCoCo budget-adaptation row in Table 2."""
from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
matrix = json.loads((root / "experiments/results_matrix/"
                     "cumulative_communication_ap_three_dataset.json").read_text())
run_root = root / "baselines_recent/cpcoco/opencood/logs/opv2v_cpcoco"
datasets = (
    ("OPV2V", "opv2v", 71, "OPV2V (Xu et al., 2022; 71 frames)"),
    ("V2XSet", "v2xset", 48, "V2XSet (Xu et al., 2022; 48 frames)"),
    ("SPD", "spd", 50, "V2X-Seq-SPD (Yu et al., 2023; 50 frames)"),
)
budgets = (10, 20, 30, 40, 50)
methods = (
    ("Full Communication", "Full Communication (reference)"),
    ("Where2comm*", r"Where2comm$^{*}$ (Hu et al., 2022)"),
    ("How2comm", r"How2comm$^{\dagger}$ (Yang et al., 2023)"),
    ("CoSDH", "CoSDH (Xu et al., 2025)"),
    ("CPCoCo", r"CPCoCo$^{\ddagger}$ (Xu et al., 2026)"),
    ("Direct (Ours)", r"\textbf{ReCAST (Ours)}"),
)


def cell(mib, saving, ap):
    return f"{mib:.1f}/{saving:.1f}/{ap:.3f}"


audit = {}
for dataset, slug, frames, _ in datasets:
    full = matrix["metadata"]["full_reference_kib_per_frame"][dataset]
    audit[dataset] = []
    for budget in budgets:
        path = run_root / f"audit_table2_{slug}_budget{budget}.json"
        result = json.loads(path.read_text())
        if result["frames"] != frames:
            raise ValueError(f"{path}: incomplete frames")
        actual = 100 * result["payload_kib_per_frame"] / full
        if abs(actual - budget) > 5:
            raise ValueError(f"{path}: budget mismatch {actual:.2f}%")
        result["actual_percent_of_reference_full"] = actual
        result["reference_full_kib_per_frame"] = full
        result["source_file"] = str(path.relative_to(root))
        audit[dataset].append(result)

json_path = root / "experiments/results_matrix/cpcoco_table2_three_dataset.json"
json_path.write_text(json.dumps({
    "protocol": ("Official OPV2V CPCoCo checkpoint; no retraining; "
                 "inference-only energy-ranked BEV mask before matching; "
                 "OPV2V/V2XSet/SPD fixed pilot scenes, V2XSet/SPD zero-shot; "
                 "FP16 non-ego feature cells plus four-byte spatial IDs; "
                 "matcher/control overhead excluded; SPD matcher rejects "
                 "collaboration and outputs Ego-only."),
    "data": audit,
}, indent=2))

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Fixed-scene communication and detection at five nominal budget targets. Each entry is cumulative transmitted MiB / saving (\%) relative to the stated Full reference / AP@0.5.}",
    r"\label{tab:comm-ap-audit}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{2.2pt}",
    r"\renewcommand{\arraystretch}{1.03}",
    r"\begin{tabular}{lccccc}",
    r"\toprule",
]
for index, (dataset, _, _, title) in enumerate(datasets):
    if index:
        lines.append(r"\midrule")
    lines += [
        rf"\multicolumn{{6}}{{l}}{{\textbf{{{title}}}}} \\",
        r"Method & 10\% & 20\% & 30\% & 40\% & 50\% \\",
        r"\midrule",
    ]
    for key, label in methods:
        if key == "CPCoCo":
            values = [cell(row["total_payload_mib"],
                           100 - row["actual_percent_of_reference_full"],
                           row["ap50"]) for row in audit[dataset]]
        else:
            values = [
                cell(row["mib"], row["saving_percent"], row["ap50"])
                for row in matrix["table"][key][dataset]
            ]
        lines.append(label + " & " + " & ".join(values) + r" \\")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\vspace{2pt}",
    r"\begin{minipage}{.98\textwidth}\scriptsize",
    r"\textit{Notes.} Full references are 7,700 KiB/frame on OPV2V/V2XSet and 14,784 KiB/frame on SPD. $^{\ddagger}$CPCoCo is the released OPV2V detector with an inference-only energy-ranked spatial mask before its matcher, not an author-provided budget policy; V2XSet/SPD are zero-shot. Its estimated payload counts FP16 selected BEV values and spatial IDs, but excludes matcher/control overhead and network framing, so it is a lower bound. On SPD the CPCoCo matcher rejects the roadside collaborator in this pilot; its AP is effectively Ego-only despite attempted transmissions. Direct and Where2comm/CoSDH count transmitted features and IDs. $^{\dagger}$How2comm uses a rate-derived lower bound. Different backbones limit causal attribution. $^{*}$Where2comm is the CoSDH-compatible implementation.",
    r"\end{minipage}",
    r"\end{table*}",
    "",
]
tex_path = root / "tables/cumulative_comm_ap_cpcoco.tex"
tex_path.write_text("\n".join(lines))
print(json_path)
print(tex_path)

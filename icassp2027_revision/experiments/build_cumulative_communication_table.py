#!/usr/bin/env python3
"""Measured cumulative communication at nominal budgets for three pilots."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/payload_runs"
RESULT = ROOT / "experiments/results_matrix"
OUT = ROOT / "output/pdf"
SUMMARY = json.loads((RESULT / "same_pilot_multibudget_summary.json").read_text())
BUDGETS = (10, 20, 30, 40, 50)
FRAMES = {"OPV2V": 71, "V2XSet": 48, "SPD": 50}
DIRECT_FULL_KIB = 7700.0  # verified 3,942,400 feature values * 2 bytes
FULL_KIB = {"OPV2V": DIRECT_FULL_KIB, "V2XSet": DIRECT_FULL_KIB,
            "SPD": 14784.0}  # SPD Ours uses CoSDH backbone
METHODS = ("Ego-only", "Full Communication", "Where2comm*",
           "How2comm", "CoSDH", "InfoCom", "Direct (Ours)")


def read(path):
    return json.loads(Path(path).read_text())


def source_label(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def cell(dataset, kib_per_frame, payload_percent, source, ap50=None,
         qualifier="measured"):
    mib = FRAMES[dataset] * kib_per_frame / 1024.0
    return {"mib": mib, "saving_percent": 100 * (1 - kib_per_frame / FULL_KIB[dataset]),
            "kib_per_frame": kib_per_frame, "actual_payload_percent": payload_percent,
            "source": source_label(source), "ap50": ap50,
            "qualifier": qualifier}


def sparse_points(dataset, method):
    key = "cosdh" if method == "CoSDH" else "where2comm"
    path = RUN / f"{key}_{'opv2v' if dataset == 'OPV2V' else 'v2xset'}"
    if dataset == "OPV2V":
        pattern = "payload_stats_kratio_*_opv2v_cost_full*.json"
    elif dataset == "V2XSet":
        pattern = "payload_stats_kratio_*_v2xset_budget_full*.json"
    elif key == "cosdh":
        pattern = "payload_stats_kratio_*_spd_budget_full*.json"
    else:
        pattern = "payload_stats_kratio_*_spd50*.json"
    points = []
    for file in path.glob(pattern):
        datum = read(file)
        points.append((datum["payload_percent"], cell(
            dataset, datum["payload_kb_per_frame"], datum["payload_percent"],
            file, datum["ap50"])))
    return points


def how_points(dataset):
    points = []
    if dataset == "SPD":
        path = RUN / "how2comm_v2xset"
        files = list(path.glob("budget_stats_spd50_thre_*.json"))
        files += list(path.glob("budget_stats_spd_budget50_full_*.json"))
        data = [(read(file)["payload_percent"], read(file)["ap50"], file)
                for file in files]
    else:
        data = [(p["payload_percent"], p["ap50"],
                 RESULT / "same_pilot_multibudget_summary.json")
                for p in SUMMARY[dataset]["How2comm"]["points"]]
        path = RUN / f"how2comm_{dataset.lower()}"
        for file in path.glob(f"budget_stats_{dataset.lower()}_budget50_full*.json"):
            data.append((read(file)["payload_percent"], read(file)["ap50"],
                         file))
    for percent, ap50, file in data:
        if percent <= 0:
            continue
        points.append((percent, cell(dataset, 35200.0 * percent / 100.0,
                                     percent, file, ap50,
                                     "rate-derived; mask overhead excluded")))
    return points


def direct_points(dataset):
    if dataset == "SPD":
        path = RUN / "cosdh_v2xset"
        files = list(path.glob("payload_stats_direct_semantic_residual_b*_spd_test_budget_*.json"))
        files += list(path.glob("payload_stats_direct_semantic_residual_b0.0800_a0.150_m1_spd_test_semres_m1*.json"))
        points = []
        for file in files:
            datum = read(file)
            points.append((datum["payload_percent"], cell(
                dataset, datum["payload_kb_per_frame"],
                datum["payload_percent"], file, datum["ap50"])))
        return points
    path = RESULT / f"direct_{dataset.lower()}_same_pilot_budget_curve.json"
    data = read(path)
    points = []
    for budget in data["budgets"]:
        row = data["curves"]["direct"][str(budget)]
        feature_kib = DIRECT_FULL_KIB * row["feature_value_fraction_realized"]
        # Three scales have 2,200 regions each. The initial full frame is
        # contiguous; later selected regions use two uint16 coordinates.
        selected = 3 * round(budget * 2200)
        index_kib = (FRAMES[dataset] - 1) * selected * 4 / 1024 / FRAMES[dataset]
        kib = feature_kib + index_kib
        percent = 100 * kib / DIRECT_FULL_KIB
        points.append((percent, cell(dataset, kib, percent, path,
                                     row["ap_50"],
                                     "float16 features plus selected-region IDs")))
    return points


def choose(dataset, method, target):
    if method == "Ego-only":
        if dataset == "SPD":
            source = RUN / "cosdh_v2xset/payload_stats_thre_0.200000_spd_ego_for_ap_table_epoch22.json"
            ap50 = read(source)["ap50"]
        else:
            source = RESULT / "same_pilot_multibudget_summary.json"
            ap50 = SUMMARY[dataset]["Direct"]["ego_ap50"]
        return cell(dataset, 0.0, 0.0, source, ap50,
                    "zero-communication reference")
    if method == "Full Communication":
        if dataset == "SPD":
            source = RUN / "cosdh_v2xset/payload_stats_direct_semantic_residual_b1.0000_a0.150_m1_spd_full_for_ap_mean_epoch22.json"
            ap50 = read(source)["ap50"]
        else:
            source = RESULT / "same_pilot_multibudget_summary.json"
            ap50 = SUMMARY[dataset]["Direct"]["full_ap50"]
        return cell(dataset, FULL_KIB[dataset], 100.0, source, ap50,
                    "100-percent Full reference")
    if method == "InfoCom":
        return None
    points = (direct_points(dataset) if method == "Direct (Ours)" else
              how_points(dataset) if method == "How2comm" else
              sparse_points(dataset, method))
    if not points:
        return None
    actual, value = min(points, key=lambda item: abs(item[0] - target))
    return value if abs(actual - target) <= 5.0 else None


table = {method: {dataset: [choose(dataset, method, target)
                           for target in BUDGETS] for dataset in FRAMES}
         for method in METHODS}
info_native = {dataset: cell(dataset, 124.75,
                             100 * 124.75 / FULL_KIB[dataset],
                             RESULT / "communication_efficiency_two_dataset.json",
                             read(RESULT / "communication_efficiency_two_dataset.json")[dataset]["InfoCom"]["ap50"],
                             "fixed native point, outside 10-50% bins")
               for dataset in ("OPV2V", "V2XSet")}
payload = {"metadata": {"budgets_percent": BUDGETS, "frames": FRAMES,
                        "full_reference_kib_per_frame": FULL_KIB,
                        "info_native": info_native,
                        "how2comm_limitation": "rate-derived feature bytes; mask/control overhead excluded"},
           "table": table}
json_path = RESULT / "cumulative_communication_ap_three_dataset.json"
json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def tex_cell(value):
    if value is None:
        return "--"
    ap = "--" if value["ap50"] is None else f"{value['ap50']:.3f}"
    if value["mib"] == 0:
        return r"0/100\%/" + ap
    return f"{value['mib']:.1f}/{value['saving_percent']:.1f}" + r"\%/" + ap


head1 = (r"\textbf{Method} & \multicolumn{5}{c}{\textbf{OPV2V (71 frames)}}"
         r" & \multicolumn{5}{c}{\textbf{V2XSet (48 frames)}}"
         r" & \multicolumn{5}{c}{\textbf{SPD (50 frames)}} \\")
head2 = " & " + " & ".join(rf"{target}\%" for _dataset in FRAMES
                             for target in BUDGETS) + r" \\"
rows = [(r"\textbf{ReCAST}" if method == "Direct (Ours)" else
         method.replace("*", r"$^{*}$").replace("How2comm", r"How2comm$^{\dagger}$")) + " & " + " & ".join(
        tex_cell(value) for dataset in FRAMES for value in table[method][dataset]) + r" \\"
        for method in METHODS]
tex = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=17.0in,paperheight=3.5in,margin=0.36in]{geometry}
\usepackage{booktabs,array,graphicx,float,caption}
\pagestyle{empty}
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.24}
\begin{document}
\begin{table}[H]\centering
\caption{Cumulative communication and AP@0.5 at five nominal budgets. Each cell is transmitted MiB / percentage saved versus sending the Full intermediate feature every frame / AP@0.5.}
\resizebox{\textwidth}{!}{\begin{tabular}{l*{15}{c}}
\toprule
""" + head1 + "\n" + r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(l){12-16}" + "\n" + head2 + "\n" + r"\midrule" + "\n" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}}
\vspace{4pt}
\begin{minipage}{\textwidth}\footnotesize
\textit{Notes.} AP is dataset-level AP@0.5 from the same measured operating point as its communication cost; it is not an average of AP@0.5 and AP@0.7. Each nominal column uses the nearest measured point within 5 payload percentage points; exact payloads and sources are in the companion JSON. Full is 7,700 KiB/frame on OPV2V and V2XSet (verified Direct PointPillar multiscale tensors) and 14,784 KiB/frame on SPD (Direct using the CoSDH backbone). Direct and CoSDH/Where2comm include indexed feature bytes. $^\dagger$How2comm is a lower-bound byte estimate from its reported communication rate against a 35,200-KiB native Full reference; mask/control overhead is excluded. InfoCom has one fixed native point outside these bins: 8.6 MiB/AP 0.993 on OPV2V and 5.8 MiB/AP 0.988 on V2XSet; SPD payload is unverified. Negative savings mean more bytes than the Full reference. Ego-only and Full are reference rows. Absolute byte differences also reflect backbone representation size. $^*$Where2comm is the CoSDH-compatible implementation.
\end{minipage}
\end{table}
\end{document}
"""
tex_path = OUT / "Cumulative_Communication_AP_Three_Datasets.tex"
tex_path.write_text(tex, encoding="utf-8")
print(json_path)
print(tex_path)
for method in METHODS:
    print(method, [tex_cell(value) for dataset in FRAMES
                   for value in table[method][dataset]])

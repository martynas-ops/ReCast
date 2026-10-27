#!/usr/bin/env python3
"""Regenerate the controlled-policy and component-ablation LaTeX tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "recast"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.3f}".lstrip("0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "tables")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy = read(RESULTS / "opv2v_policy_results.json")
    tx = read(RESULTS / "controlled_policies/opv2v_tx_delta_10pct.json")["tx_delta"]
    oldest = read(
        RESULTS / "controlled_policies/opv2v_oldest_first_10pct.json"
    )["oldest_first"]
    random = read(
        RESULTS / "controlled_policies/opv2v_random_b_10seed_summary.json"
    )
    controlled = [
        ("Periodic", policy["periodic"]["feature_value_fraction_realized"],
         policy["periodic"]["ap_50"], policy["periodic"]["ap_70"]),
        ("Tx-Delta", tx["feature_value_fraction_realized"], tx["ap_50"], tx["ap_70"]),
        ("Random-B", random["payload"]["mean"], random["ap_50"]["mean"],
         random["ap_70"]["mean"]),
        ("Oldest-first", oldest["feature_value_fraction_realized"],
         oldest["ap_50"], oldest["ap_70"]),
        ("ReCAST", policy["direct"]["feature_value_fraction_realized"],
         policy["direct"]["ap_50"], policy["direct"]["ap_70"]),
    ]
    lines = [
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Policy & Payload [\%] & AP@0.5 & AP@0.7\\", r"\midrule",
    ]
    for name, payload, ap50, ap70 in controlled:
        if name == "Random-B":
            ap50_text = f"${fmt(ap50)}{{\\pm}}{fmt(random['ap_50']['stdev'])}$"
            ap70_text = f"${fmt(ap70)}{{\\pm}}{fmt(random['ap_70']['stdev'])}$"
        else:
            ap50_text, ap70_text = fmt(ap50), fmt(ap70)
        lines.append(
            f"{name} & {100 * payload:.2f} & {ap50_text} & {ap70_text}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (args.output_dir / "controlled_refresh.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    factorial = read(RESULTS / "factorial/opv2v_factorial_summary.json")["values"]
    lines = [
        r"\begin{tabular}{ccc rrr}", r"\toprule",
        r"Transport & Cache residual & Top-$B$ & Payload [\%] & AP@0.5 & AP@0.7\\",
        r"\midrule",
    ]
    for code in ("000", "100", "010", "001", "110", "101", "011", "111"):
        value = factorial[code]
        checks = [r"$\checkmark$" if bit == "1" else "" for bit in code]
        lines.append(
            " & ".join(checks + [
                f"{100 * value['payload']:.2f}", fmt(value["ap50"]),
                fmt(value["ap70"]),
            ]) + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (args.output_dir / "component_ablation.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

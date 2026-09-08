#!/usr/bin/env python3
"""
Generate the STAG-STI results/analysis document from all finished runs.

Produces:
  results/REPORT.md            -- headline metrics, literature comparison,
                                  ablation study, per-session tables
  results/combined_curves.png  -- per-session accuracy curves, all datasets

Renders whatever results currently exist under checkpoints/, so it can be run
repeatedly as runs complete (the training orchestrator calls it after each).

Published baselines: CUB-200 last-session accuracy / PD are taken from the
Review-1 deck (FACT CVPR'22, SAVC CVPR'23, CLOSER ECCV'24). CIFAR-100 and
miniImageNet published columns are left as TODO -- fill from the papers to
avoid citing approximate numbers.
"""
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CK = os.path.join(_REPO, "checkpoints")
_OUT = os.path.join(_REPO, "results")

DATASETS = [
    ("cifar100", "CIFAR-100"),
    ("miniimagenet", "miniImageNet"),
    ("cub200", "CUB-200"),
]
ABLATIONS = ["supcon", "topology", "agedecay"]

# From the Review-1 deck (CUB-200 last-session accuracy; PD where given).
PUBLISHED = {
    "cub200": {"FACT": (56.94, 18.96), "SAVC": (62.50, None), "CLOSER": (63.58, 15.82)},
    "cifar100": {},       # TODO: fill from papers
    "miniimagenet": {},   # TODO: fill from papers
}


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _acc(entry):
    return entry.get("accuracy", entry.get("acc"))


def _row(res):
    r = res["results"]
    a0, aT = _acc(r[0]) * 100, _acc(r[-1]) * 100
    last = r[-1]
    aB = last.get("acc_base")
    aN = last.get("acc_novel")
    hm = last.get("harmonic_mean")
    def pct(x):
        return f"{x*100:.2f}" if isinstance(x, (int, float)) and x == x else "-"
    return {
        "A0": a0, "AT": aT, "PD": res.get("PD", a0 - aT),
        "avg": res.get("avg_accuracy", float("nan")),
        "AB": pct(aB), "AN": pct(aN), "HM": pct(hm),
    }


def main():
    os.makedirs(_OUT, exist_ok=True)
    lines = []
    lines.append(f"# STAG-STI — Results & Analysis\n")
    lines.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M} • auto-updated as runs complete._\n")

    # ---- headline results ----
    lines.append("## In-domain results\n")
    lines.append("| Dataset | A₀ (base) | A_T (final) | PD ↓ | Avg | A_B | A_N | HM |")
    lines.append("|---|---|---|---|---|---|---|---|")
    curves = []
    for key, name in DATASETS:
        res = _load(os.path.join(_CK, key, "incremental_results.json"))
        if not res:
            lines.append(f"| {name} | _pending_ | | | | | | |")
            continue
        m = _row(res)
        lines.append(f"| {name} | {m['A0']:.2f} | {m['AT']:.2f} | {m['PD']:.2f} | "
                     f"{m['avg']:.2f} | {m['AB']} | {m['AN']} | {m['HM']} |")
        curves.append((name, [_acc(r) * 100 for r in res["results"]]))

    # ---- cross-domain ----
    cd = _load(os.path.join(_CK, "crossdomain_mini2cub", "crossdomain_results.json"))
    lines.append("\n## Cross-domain (miniImageNet → CUB)\n")
    if cd:
        m = _row(cd)
        lines.append(f"- Base (miniImageNet): **{m['A0']:.2f}%** → final (all seen): "
                     f"**{m['AT']:.2f}%**, PD **{m['PD']:.2f}**, "
                     f"A_B {m['AB']} / A_N {m['AN']} / HM {m['HM']}")
        curves.append(("cross-domain", [_acc(r) * 100 for r in cd["results"]]))
    else:
        lines.append("- _pending_")

    # ---- literature comparison ----
    lines.append("\n## Comparison vs. published methods (final-session accuracy %)\n")
    lines.append("| Method | CIFAR-100 | miniImageNet | CUB-200 |")
    lines.append("|---|---|---|---|")
    for method in ["FACT", "SAVC", "CLOSER"]:
        cells = []
        for key, _ in DATASETS:
            pub = PUBLISHED.get(key, {}).get(method)
            cells.append(f"{pub[0]:.2f}" if pub else "TODO")
        lines.append(f"| {method} | {cells[0]} | {cells[1]} | {cells[2]} |")
    ours = []
    for key, _ in DATASETS:
        res = _load(os.path.join(_CK, key, "incremental_results.json"))
        ours.append(f"**{_acc(res['results'][-1])*100:.2f}**" if res else "_pending_")
    lines.append(f"| **STAG-STI (ours)** | {ours[0]} | {ours[1]} | {ours[2]} |")
    lines.append("\n_CUB-200 baselines from the Review-1 deck; CIFAR-100 / miniImageNet "
                 "baselines are TODO — fill from the source papers._")

    # ---- ablation study (CIFAR) ----
    lines.append("\n## Ablation study (CIFAR-100)\n")
    lines.append("| Configuration | A₀ | A_T | PD ↓ | Avg |")
    lines.append("|---|---|---|---|---|")
    full = _load(os.path.join(_CK, "cifar100", "incremental_results.json"))
    if full:
        m = _row(full)
        lines.append(f"| Full model | {m['A0']:.2f} | {m['AT']:.2f} | {m['PD']:.2f} | {m['avg']:.2f} |")
    for ab in ABLATIONS:
        res = _load(os.path.join(_CK, "cifar100", f"ablate-{ab}", "incremental_results.json"))
        if res:
            m = _row(res)
            lines.append(f"| − {ab} | {m['A0']:.2f} | {m['AT']:.2f} | {m['PD']:.2f} | {m['avg']:.2f} |")
        else:
            lines.append(f"| − {ab} | _pending_ | | | |")

    # ---- combined curves figure ----
    if curves:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, ys in curves:
            ax.plot(range(len(ys)), ys, marker="o", label=name)
        ax.set_xlabel("session (0 = base)")
        ax.set_ylabel("cumulative test accuracy (%)")
        ax.set_title("STAG-STI — per-session accuracy across benchmarks")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig_path = os.path.join(_OUT, "combined_curves.png")
        fig.savefig(fig_path, dpi=130)
        plt.close(fig)
        lines.append(f"\n## Per-session curves\n\n![combined curves](combined_curves.png)")

    report = "\n".join(lines) + "\n"
    with open(os.path.join(_OUT, "REPORT.md"), "w") as f:
        f.write(report)
    print(f"[make_report] wrote {os.path.join(_OUT, 'REPORT.md')}")


if __name__ == "__main__":
    main()

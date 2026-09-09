#!/usr/bin/env python3
"""
Regenerate the auto-updating results block in README.md from whatever runs
currently exist under checkpoints/. Idempotent: rewrites only the content
between the AUTO:RESULTS markers, so it can run on every commit (via the
.githooks/pre-commit hook) and only produces a diff when results actually change.
"""
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CK = os.path.join(_REPO, "checkpoints")
_README = os.path.join(_REPO, "README.md")
START = "<!-- AUTO:RESULTS:START -->"
END = "<!-- AUTO:RESULTS:END -->"

DATASETS = [("cifar100", "CIFAR-100"), ("miniimagenet", "miniImageNet"), ("cub200", "CUB-200")]
ABLATIONS = [("supcon", "− contrastive"), ("topology", "− topology"), ("agedecay", "− age-decay")]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def acc(entry):
    return entry.get("accuracy", entry.get("acc"))


def row(res):
    if not res:
        return None
    r = res["results"]
    last = r[-1]
    def p(x):
        return f"{x*100:.2f}" if isinstance(x, (int, float)) and x == x else "–"
    return dict(A0=acc(r[0]) * 100, AT=acc(last) * 100, PD=res.get("PD", float("nan")),
                avg=res.get("avg_accuracy", float("nan")),
                AB=p(last.get("acc_base")), AN=p(last.get("acc_novel")), HM=p(last.get("harmonic_mean")))


def md_indomain():
    lines = ["| Dataset | A₀ base | A_T final | PD ↓ | Avg | A_B | A_N | HM |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]
    any_row = False
    for key, name in DATASETS:
        m = row(load(os.path.join(_CK, key, "incremental_results.json")))
        if not m:
            lines.append(f"| {name} | _pending_ | | | | | | |"); continue
        any_row = True
        lines.append(f"| {name} | {m['A0']:.2f} | {m['AT']:.2f} | {m['PD']:.2f} | {m['avg']:.2f} | {m['AB']} | {m['AN']} | {m['HM']} |")
    return "\n".join(lines), any_row


def md_transductive():
    lines = ["| Dataset | Final (ind.) | Final (transd.) | A_N (ind.) | A_N (transd.) | HM (ind.) | HM (transd.) |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for key, name in DATASETS:
        b = row(load(os.path.join(_CK, key, "incremental_results.json")))
        t = row(load(os.path.join(_CK, key, "transductive", "incremental_results.json")))
        if not (b and t):
            continue
        lines.append(f"| {name} | {b['AT']:.2f} | {t['AT']:.2f} | {b['AN']} | {t['AN']} | {b['HM']} | {t['HM']} |")
    return "\n".join(lines) if len(lines) > 2 else ""


def md_crossdomain():
    cd = row(load(os.path.join(_CK, "crossdomain_mini2cub", "crossdomain_results.json")))
    if not cd:
        return ""
    return (f"Base (miniImageNet) **{cd['A0']:.2f}%** → final (all seen) **{cd['AT']:.2f}%**, "
            f"PD **{cd['PD']:.2f}**, A_B {cd['AB']} / A_N {cd['AN']} / HM {cd['HM']}.")


def md_ablation():
    full = row(load(os.path.join(_CK, "cifar100", "incremental_results.json")))
    if not full:
        return ""
    lines = ["| Configuration | A₀ | A_T | A_B | A_N | HM |", "|---|--:|--:|--:|--:|--:|",
             f"| **Full model** | {full['A0']:.2f} | {full['AT']:.2f} | {full['AB']} | {full['AN']} | {full['HM']} |"]
    for key, label in ABLATIONS:
        m = row(load(os.path.join(_CK, "cifar100", f"ablate-{key}", "incremental_results.json")))
        if m:
            lines.append(f"| {label} | {m['A0']:.2f} | {m['AT']:.2f} | {m['AB']} | {m['AN']} | {m['HM']} |")
    return "\n".join(lines)


def md_closer():
    lines = ["| Dataset | A_T | A_B | A_N | HM | ΔA_N | ΔHM |", "|---|--:|--:|--:|--:|--:|--:|"]
    any_row = False
    for key, name in DATASETS:
        b = load(os.path.join(_CK, key, "incremental_results.json"))
        c = load(os.path.join(_CK, key, "closer", "incremental_results.json"))
        if not (b and c):
            continue
        any_row = True
        mc = row(c)
        dAN = (acc_last(c, "acc_novel") - acc_last(b, "acc_novel")) * 100
        dHM = (acc_last(c, "harmonic_mean") - acc_last(b, "harmonic_mean")) * 100
        lines.append(f"| {name} | {mc['AT']:.2f} | {mc['AB']} | {mc['AN']} | {mc['HM']} | {dAN:+.1f} | {dHM:+.1f} |")
    return "\n".join(lines) if any_row else ""


def acc_last(res, k):
    return res["results"][-1].get(k)


def build_block():
    parts = ["## Results", "", "_Auto-generated from `checkpoints/` by `scripts/gen_readme.py` "
             "(run on every commit). Single-run, first-pass numbers on official splits._", ""]
    ind, any_ind = md_indomain()
    parts += ["### In-domain (inductive)", "", ind, ""]
    td = md_transductive()
    if td:
        parts += ["### Transductive prototype rectification", "", td, ""]
    cd = md_crossdomain()
    if cd:
        parts += ["### Cross-domain (miniImageNet → CUB)", "", cd, ""]
    ab = md_ablation()
    if ab:
        parts += ["### Ablation study (CIFAR-100)", "", ab, ""]
    cl = md_closer()
    if cl:
        parts += ["### CLOSER-style base objective (Δ vs baseline)", "", cl, ""]
    parts += ["> Full narrative report with charts: **`results/report.html`**."]
    return "\n".join(parts)


def main():
    if not os.path.exists(_README):
        print("[gen_readme] README.md not found; skipping")
        return
    text = open(_README).read()
    if START not in text or END not in text:
        print("[gen_readme] AUTO:RESULTS markers not found; skipping")
        return
    pre = text.split(START)[0]
    post = text.split(END)[1]
    new = f"{pre}{START}\n{build_block()}\n{END}{post}"
    if new != text:
        open(_README, "w").write(new)
        print("[gen_readme] README results block updated")
    else:
        print("[gen_readme] README already current")


if __name__ == "__main__":
    main()

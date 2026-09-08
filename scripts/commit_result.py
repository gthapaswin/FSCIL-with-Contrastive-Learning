#!/usr/bin/env python3
"""
Auto-commit + push a finished run's results with a metric-tagged message.
Used by the training orchestrator so every result lands on GitHub per the
standing "commit after every result" rule, even unattended.

Usage:
    python scripts/commit_result.py --ckpt-dir checkpoints/cifar100 --label "CIFAR-100"
    python scripts/commit_result.py --ckpt-dir checkpoints/cifar100/ablate-supcon \
        --label "CIFAR-100 ablate-supcon"
    python scripts/commit_result.py --ckpt-dir checkpoints/crossdomain_mini2cub \
        --label "cross-domain miniImageNet->CUB" --results-file crossdomain_results.json
"""
import argparse
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    return subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True)


def _fmt(results):
    r = results["results"]
    a0, aT = r[0]["accuracy"] * 100, r[-1]["accuracy"] * 100
    pd = results.get("PD", a0 - aT)
    parts = [f"base {a0:.2f}%", f"final {aT:.2f}%", f"PD {pd:.2f}"]
    if "avg_accuracy" in results:
        parts.append(f"avg {results['avg_accuracy']:.2f}%")
    last = r[-1]
    if last.get("acc_base") == last.get("acc_base") and "acc_base" in last:  # not NaN
        parts.append(f"A_B {last['acc_base']*100:.2f}%")
        parts.append(f"A_N {last['acc_novel']*100:.2f}%")
        if last.get("harmonic_mean") == last.get("harmonic_mean"):
            parts.append(f"HM {last['harmonic_mean']*100:.2f}%")
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--results-file", default="incremental_results.json")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    ckpt_dir = os.path.join(_REPO, args.ckpt_dir) if not os.path.isabs(args.ckpt_dir) else args.ckpt_dir
    results_path = os.path.join(ckpt_dir, args.results_file)
    if not os.path.exists(results_path):
        print(f"[commit_result] no results at {results_path}; skipping")
        return 1
    with open(results_path) as f:
        results = json.load(f)
    summary = _fmt(results)

    _git("add", args.ckpt_dir)
    _git("add", "results")  # include the regenerated report in the same commit
    msg = (f"{args.label} result: {summary}\n\n"
           "Auto-committed by the training orchestrator.\n\n"
           "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01GDGJYxSeGz7wbgpdPrPcEm")
    cm = _git("commit", "-m", msg)
    if cm.returncode != 0 and "nothing to commit" in (cm.stdout + cm.stderr):
        print(f"[commit_result] nothing to commit for {args.label}")
        return 0
    print(f"[commit_result] committed: {args.label} result: {summary}")
    if not args.no_push:
        p = _git("push", "origin", "main")
        print("[commit_result] push:", "ok" if p.returncode == 0 else p.stderr.strip()[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())

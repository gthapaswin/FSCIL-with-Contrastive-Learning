#!/usr/bin/env bash
# Re-run EVAL ONLY (no retraining) across all runs after the base-prototype
# chunking fix, so every dataset uses the same base-build method. Commits each
# refreshed result + regenerates the report.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"

reeval () {  # <dataset> <ckpt_subdir> <label> [extra eval args...]
  local ds="$1" dir="$2" label="$3"; shift 3
  echo "[rerun] $(date) eval $label"
  $PY -m fscil.eval_incremental --dataset "$ds" "$@"
  $PY -m fscil.plotting --dataset "$ds" "$@" || true
  $PY scripts/make_report.py || true
  $PY scripts/commit_result.py --ckpt-dir "$dir" --label "$label" || true
}

reeval cifar100     checkpoints/cifar100     "CIFAR-100 (rescored)"
reeval miniimagenet checkpoints/miniimagenet "miniImageNet (rescored)"
reeval cub200       checkpoints/cub200       "CUB-200 (rescored)"
for ABL in supcon topology agedecay; do
  reeval cifar100 "checkpoints/cifar100/ablate-$ABL" "CIFAR-100 ablate-$ABL (rescored)" --ablate "$ABL"
done

echo "[rerun] $(date) cross-domain miniImageNet->CUB"
$PY -m fscil.eval_crossdomain
$PY scripts/make_report.py || true
$PY scripts/commit_result.py --ckpt-dir checkpoints/crossdomain_mini2cub \
    --label "cross-domain miniImageNet->CUB" --results-file crossdomain_results.json || true

echo "[rerun] $(date) ALL EVALS DONE. See results/REPORT.md"

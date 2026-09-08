#!/usr/bin/env bash
# Extend the CLOSER-style base objective (lambda_close=0.1) to miniImageNet and
# CUB, with inductive + transductive eval, committing each. Resumable: skips a
# dataset whose closer result already exists, and reuses a saved CLOSER backbone
# if Phase A finished but Phase B did not.
#
# Usage: scripts/run_closer.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"

for DS in miniimagenet cub200; do
  RESULT="checkpoints/$DS/closer/incremental_results.json"
  if [ -f "$RESULT" ]; then echo "[closer] $DS already done, skipping"; continue; fi
  echo "[closer] $(date) === $DS CLOSER (lambda_close=0.1) ==="
  BB="checkpoints/$DS/closer/backbone_base.pt"
  # salvage: if Phase A was interrupted before the final save but a best-val
  # backbone exists, promote it so we resume Phase B instead of retraining.
  if [ ! -f "$BB" ] && [ -f "checkpoints/$DS/closer/backbone_base_best.pt" ]; then
    echo "[closer] promoting backbone_base_best.pt -> backbone_base.pt for $DS"
    cp "checkpoints/$DS/closer/backbone_base_best.pt" "$BB"
  fi
  if [ -f "$BB" ]; then
    $PY -m fscil.train_session0 --dataset "$DS" --closer --skip_backbone_pretrain
  else
    $PY -m fscil.train_session0 --dataset "$DS" --closer
  fi
  $PY -m fscil.eval_incremental --dataset "$DS" --closer
  $PY -m fscil.eval_incremental --dataset "$DS" --closer --transductive
  $PY scripts/commit_result.py --ckpt-dir "checkpoints/$DS/closer" --label "$DS CLOSER"
done
echo "[closer] $(date) ALL CLOSER RUNS DONE"

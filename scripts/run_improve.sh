#!/usr/bin/env bash
# Improvement runs, resumable:
#  (1) CUB full-base + transductive eval-only measurement (quick).
#  (2) miniImageNet longer/better backbone retrain (the real ceiling-raiser):
#      100 Phase-A epochs, namespaced under checkpoints/miniimagenet/longbb/.
# Skips finished steps; salvages a best-val backbone if Phase A was interrupted.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"

# (1) CUB eval-only (uses existing best checkpoints)
CK=checkpoints/cub200
for FLAGS in "--full_base" "--full_base --transductive"; do
  $PY -m fscil.eval_incremental --dataset cub200 $FLAGS \
      --backbone_ckpt $CK/backbone_base_best.pt --stag_ckpt $CK/stag_sti_session0_best.pt || true
done

# (2) miniImageNet longer backbone
LB=checkpoints/miniimagenet/longbb
if [ ! -f "$LB/incremental_results.json" ]; then
  echo "[improve] $(date) === miniImageNet longer backbone (100 ep) ==="
  if [ ! -f "$LB/backbone_base.pt" ] && [ -f "$LB/backbone_base_best.pt" ]; then
    cp "$LB/backbone_base_best.pt" "$LB/backbone_base.pt"
  fi
  if [ -f "$LB/backbone_base.pt" ]; then
    $PY -m fscil.train_session0 --dataset miniimagenet --tag longbb --skip_backbone_pretrain
  else
    $PY -m fscil.train_session0 --dataset miniimagenet --tag longbb --backbone_epochs 100
  fi
  $PY -m fscil.eval_incremental --dataset miniimagenet --tag longbb
  $PY -m fscil.eval_incremental --dataset miniimagenet --tag longbb --full_base --transductive
  $PY scripts/commit_result.py --ckpt-dir "$LB" --label "miniImageNet longer-backbone" || true
fi
echo "[improve] $(date) DONE"

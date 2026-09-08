#!/usr/bin/env bash
# Second stage of the training queue: after CUB finishes, run miniImageNet
# (from-scratch backbone) then the cross-domain (miniImageNet->CUB) eval.
# Serializes the single mps GPU behind the CUB run started by train_queue.sh.
#
# Usage: scripts/train_queue2.sh [logfile]
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"
CUB_SENTINEL="checkpoints/cub200/incremental_results.json"

echo "[queue2] $(date) waiting for CUB to finish ($CUB_SENTINEL)..."
for _ in $(seq 1 600); do          # up to ~10h, checking each minute
  [ -f "$CUB_SENTINEL" ] && break
  sleep 60
done
if [ ! -f "$CUB_SENTINEL" ]; then
  echo "[queue2] CUB sentinel still missing -- aborting."
  exit 1
fi

echo "[queue2] $(date) starting miniImageNet (from-scratch backbone)"
$PY -m fscil.train_session0 --dataset miniimagenet
$PY -m fscil.eval_incremental --dataset miniimagenet

echo "[queue2] $(date) starting cross-domain eval (miniImageNet -> CUB)"
$PY -m fscil.eval_crossdomain

echo "[queue2] $(date) all done. Results under checkpoints/miniimagenet/ and checkpoints/crossdomain_mini2cub/."

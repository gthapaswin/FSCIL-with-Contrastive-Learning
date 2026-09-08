#!/usr/bin/env bash
# Serialize training on the single mps GPU: wait for the in-flight CIFAR run to
# finish (its incremental_results.json is written last), then run CUB-200.
# miniImageNet + cross-domain are queued separately once the mini data lands.
#
# Usage: scripts/train_queue.sh [logfile]
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"
LOG="${1:-/tmp/train_queue.log}"
CIFAR_SENTINEL="checkpoints/cifar100/incremental_results.json"

echo "[queue] $(date) waiting for CIFAR to finish ($CIFAR_SENTINEL)..."
for _ in $(seq 1 360); do          # up to ~6h, checking each minute
  [ -f "$CIFAR_SENTINEL" ] && break
  sleep 60
done
if [ ! -f "$CIFAR_SENTINEL" ]; then
  echo "[queue] CIFAR sentinel still missing after wait -- aborting so we don't"
  echo "        start CUB on a possibly-failed CIFAR run. Check cifar_baseline.log."
  exit 1
fi

echo "[queue] $(date) CIFAR done -> starting CUB-200 (pretrained backbone, fine-tune)"
$PY -m fscil.train_session0 --dataset cub200
$PY -m fscil.eval_incremental --dataset cub200
echo "[queue] $(date) CUB-200 done. Results under checkpoints/cub200/."

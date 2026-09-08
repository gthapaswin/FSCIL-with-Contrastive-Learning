#!/usr/bin/env bash
# Ablation study for STAG-STI (slide-17 protocol): train + eval the full model
# and each single-component ablation, on one dataset. The frozen backbone is
# shared across ablations (Phase A is identical), so we pretrain it once and
# reuse it with --skip_backbone_pretrain.
#
# Usage: scripts/run_ablations.sh <dataset>   e.g. scripts/run_ablations.sh cifar100
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:-cifar100}"
PY="${PYTHON:-.venv/bin/python}"

echo "=== [$DATASET] full model (also pretrains the shared backbone) ==="
$PY -m fscil.train_session0 --dataset "$DATASET"
$PY -m fscil.eval_incremental --dataset "$DATASET"

for ABL in supcon topology agedecay; do
  echo "=== [$DATASET] ablate: $ABL (reusing shared backbone) ==="
  $PY -m fscil.train_session0 --dataset "$DATASET" --ablate "$ABL" --skip_backbone_pretrain
  $PY -m fscil.eval_incremental --dataset "$DATASET" --ablate "$ABL"
done

echo "=== done. Results under checkpoints/$DATASET/ and checkpoints/$DATASET/ablate-*/ ==="

#!/usr/bin/env bash
# Master training orchestrator: runs the full remaining STAG-STI pipeline on the
# single mps GPU, committing + pushing each result (with the analysis report) as
# it lands. Picks up after the CUB run started by train_queue.sh.
#
# Sequence: [wait CUB] -> commit CUB -> miniImageNet -> cross-domain
#           -> CIFAR ablations (supcon, topology, agedecay) -> done
#
# Usage: scripts/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"

finish () {  # <ckpt_subdir> <label> <plotargs> [results_file]
  local dir="$1" label="$2" plotargs="$3" rf="${4:-incremental_results.json}"
  echo "[run_all] $(date) finalizing: $label"
  $PY -m fscil.plotting $plotargs || true
  $PY scripts/make_report.py || true
  $PY scripts/commit_result.py --ckpt-dir "$dir" --label "$label" --results-file "$rf" || true
}

wait_for () {  # <sentinel> <max_minutes>
  local sentinel="$1" max="${2:-600}"
  echo "[run_all] $(date) waiting for $sentinel ..."
  for _ in $(seq 1 "$max"); do [ -f "$sentinel" ] && return 0; sleep 60; done
  echo "[run_all] timed out waiting for $sentinel"; return 1
}

# 1) CUB (already training via train_queue.sh) -> commit when done
if wait_for "checkpoints/cub200/incremental_results.json" 600; then
  finish "checkpoints/cub200" "CUB-200" "--dataset cub200"
fi

# 2) miniImageNet (from-scratch backbone)
echo "[run_all] $(date) === miniImageNet ==="
$PY -m fscil.train_session0 --dataset miniimagenet
$PY -m fscil.eval_incremental --dataset miniimagenet
finish "checkpoints/miniimagenet" "miniImageNet" "--dataset miniimagenet"

# 3) cross-domain (needs the miniImageNet base model + CUB data)
echo "[run_all] $(date) === cross-domain miniImageNet->CUB ==="
$PY -m fscil.eval_crossdomain
finish "checkpoints/crossdomain_mini2cub" "cross-domain miniImageNet->CUB" "" "crossdomain_results.json"

# 4) CIFAR ablation study (reuse the shared frozen backbone)
for ABL in supcon topology agedecay; do
  echo "[run_all] $(date) === CIFAR ablation: $ABL ==="
  $PY -m fscil.train_session0 --dataset cifar100 --ablate "$ABL" --skip_backbone_pretrain
  $PY -m fscil.eval_incremental --dataset cifar100 --ablate "$ABL"
  finish "checkpoints/cifar100/ablate-$ABL" "CIFAR-100 ablate-$ABL" "--dataset cifar100 --ablate $ABL"
done

echo "[run_all] $(date) ALL DONE. See results/REPORT.md"

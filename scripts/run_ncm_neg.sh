#!/usr/bin/env bash
# Extend NCM's offset curve to negative offsets (b = -1, -0.5) on the seed-42
# held-out runs, so methods that favour novel classes (TEEN) can be compared
# at high base accuracy. Waits for scripts/run_teen.sh.
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"
while pgrep -f run_teen.sh > /dev/null; do sleep 60; done
for spec in "cifar100 reserve_cm100" "miniimagenet reserve_cm100" "cub200 reserve_cm30"; do
  set -- $spec; ds=$1; bb=$2; ho=checkpoints/$ds/${bb}_heldout
  for b in -1 -0.5; do
    [ -f "$ho/fullbase/fuse1/nb$b/incremental_results.json" ] ||
      "$PY" -m fscil.eval_incremental --dataset $ds --tag ${bb}_heldout --full_base \
        --backbone_ckpt checkpoints/$ds/$bb/backbone_base.pt --stag_ckpt $ho/stag_sti_session0_best.pt \
        --fuse 1 --novel_bias $b > /dev/null || echo "[ncm-neg] FAILED $ds $b"
  done
done
echo "[ncm-neg] $(date) ALL DONE"

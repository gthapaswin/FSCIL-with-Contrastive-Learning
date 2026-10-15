#!/usr/bin/env bash
# Calibration control applied to a published method: TEEN (Wang et al.,
# NeurIPS 2023) training-free prototype calibration on the novel NCM
# prototypes of the seed-42 recipe backbones, swept over its strength alpha
# and the novel offset b, so its (A_B, A_N) points can be laid on NCM's
# offset curve. Paper defaults: alpha 0.1 (CIFAR-100), 0.5 (mini, CUB), tau 16.
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"
for spec in "cifar100 reserve_cm100" "miniimagenet reserve_cm100" "cub200 reserve_cm30"; do
  set -- $spec; ds=$1; bb=$2; ho=checkpoints/$ds/${bb}_heldout
  for a in 0.1 0.5 0.9; do
    for b in -1 0 0.5 1; do
      out="$ho/fullbase/fuse1/teen$a"; [ "$b" = 0.5 ] || out="$out/nb$b"
      [ -f "$out/incremental_results.json" ] && continue
      nb=(); [ "$b" = 0.5 ] || nb=(--novel_bias "$b")
      echo "[teen] $(date) $ds alpha $a b $b"
      "$PY" -m fscil.eval_incremental --dataset $ds --tag ${bb}_heldout --full_base \
        --backbone_ckpt checkpoints/$ds/$bb/backbone_base.pt --stag_ckpt $ho/stag_sti_session0_best.pt \
        --fuse 1 --teen $a "${nb[@]}" > /dev/null || echo "[teen] FAILED $ds $a $b"
    done
  done
done
echo "[teen] $(date) ALL DONE"

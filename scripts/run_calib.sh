#!/usr/bin/env bash
# Calibrated NCM for every finished recipe run (all seeds, ResNet-20 too):
#   1. pick the novel-logit bias on the held-out validation  -> split_val.json
#   2. evaluate NCM (lam=1) with that bias                    -> fullbase/fuse1[/nb<b>]
# Steps already done are skipped, so rerun it whenever new seeds land.
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"
DEFAULT_BIAS=0.5
for spec in "cifar100 reserve_cm100" "miniimagenet reserve_cm100" "cub200 reserve_cm30" \
            "cifar100 reserve_cm100_resnet20 resnet20"; do
  set -- $spec; ds=$1; bb=$2; arch=$3
  for dir in checkpoints/$ds/${bb}_heldout checkpoints/$ds/${bb}_heldout/seed*; do
    [ -f "$dir/heldout_val.json" ] && [ -f "$dir/fullbase/fuse1/incremental_results.json" ] || continue
    seed=42; case "$dir" in */seed*) seed=${dir##*/seed} ;; esac
    a=(--dataset "$ds" --seed "$seed"); [ -n "$arch" ] && a+=(--backbone "$arch")
    [ -f "$dir/split_val.json" ] || "$PY" scripts/split_val.py "${a[@]}" --bb "$bb" --lams 1.0 > /dev/null || continue
    b=$("$PY" -c "import json; print('%g' % json.load(open('$dir/split_val.json'))['selected_ncm_bias'])")
    echo "[calib] $dir: selected bias $b"
    [ "$b" = "$DEFAULT_BIAS" ] && continue      # fuse1/ already has it
    bbdir=checkpoints/$ds/$bb; [ "$seed" != 42 ] && bbdir=$bbdir/seed$seed
    [ -f "$dir/fullbase/fuse1/nb$b/incremental_results.json" ] ||
      "$PY" -m fscil.eval_incremental "${a[@]}" --tag "${bb}_heldout" --backbone_ckpt "$bbdir/backbone_base.pt" \
        --stag_ckpt "$dir/stag_sti_session0_best.pt" --full_base --fuse 1 --novel_bias "$b" > /dev/null || continue
    "$PY" scripts/commit_result.py --ckpt-dir "$dir" --results-file "fullbase/fuse1/nb$b/incremental_results.json" \
        --label "$ds calibrated NCM, val-selected bias $b (${dir#checkpoints/$ds/})" || true
  done
done

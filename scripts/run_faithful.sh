#!/usr/bin/env bash
# Technical-Document ("faithful") STAG-STI on the seed-42 recipe backbones:
#   std     -- the document exactly: standard episodic Phase B, head alone
#   heldout -- same head, checkpoint selected on held-out base classes
#              (head trained alone: residual lambda 0)
# Each run gets the lam sweep and a novel-offset sweep of the head alone, so
# its (A_B, A_N) points can be laid on the NCM offset curve. Resumable;
# finished steps are skipped. Detached:
#   nohup caffeinate -i scripts/run_faithful.sh >> logs/faithful.log 2>&1 &
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"
BIASES="0 1 2 3 4 6"

run() {
  local ds=$1 bb=$2 kind=$3 tag=faithful_$3 dir="checkpoints/$1/faithful_$3"
  local extra=(); [ "$kind" = heldout ] && extra=(--heldout --residual_lambda 0)
  echo "[faithful] $(date) $ds $kind"
  [ -f "$dir/training_history.json" ] ||
    "$PY" -m fscil.train_session0 --dataset "$ds" --faithful --backbone_tag "$bb" --tag "$tag" "${extra[@]}" || return 1
  local ev=("$PY" -m fscil.eval_incremental --dataset "$ds" --faithful --tag "$tag" --full_base
            --backbone_ckpt "checkpoints/$ds/$bb/backbone_base.pt" --stag_ckpt "$dir/stag_sti_session0_best.pt")
  if [ ! -f "$dir/fullbase/fuse0/incremental_results.json" ]; then
    "${ev[@]}" --fuse 0 0.25 0.5 0.75 1 || return 1
    "$PY" scripts/commit_result.py --ckpt-dir "$dir" --results-file fullbase/fuse0/incremental_results.json \
        --label "$ds faithful tech-doc head ($kind), head alone" || true
  fi
  for b in $BIASES; do
    [ -f "$dir/fullbase/fuse0/nb$b/incremental_results.json" ] || "${ev[@]}" --fuse 0 --novel_bias "$b" || return 1
  done
  "$PY" scripts/commit_result.py --ckpt-dir "$dir" --results-file "fullbase/fuse0/nb3/incremental_results.json" \
      --label "$ds faithful tech-doc head ($kind), head alone, offset sweep $BIASES" || true
}

for kind in heldout std; do
  run cifar100 reserve_cm100 "$kind"
  run miniimagenet reserve_cm100 "$kind"
  run cub200 reserve_cm30 "$kind"
done
echo "[faithful] $(date) ALL DONE"

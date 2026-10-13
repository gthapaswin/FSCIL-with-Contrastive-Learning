#!/usr/bin/env bash
# Paper experiment queue, meant to run detached for ~1.5 days:
#   nohup caffeinate -i scripts/run_queue.sh >> logs/queue.log 2>&1 &
# Every step resumes or is skipped once done, so rerunning the whole queue
# after a crash or reboot continues where it stopped. A failed step is logged
# and the queue moves on. Order = importance for the paper.
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"
EPOCHS_cifar100=100; EPOCHS_miniimagenet=100; EPOCHS_cub200=30
ep() { eval echo "\$EPOCHS_$1"; }
step() { echo "[queue] $(date) START $*"; "$@" && echo "[queue] $(date) OK $*" || echo "[queue] $(date) FAILED $*"; }

# 0. let a recipe started outside the queue finish first (no double runs)
while pgrep -f "run_recipe.sh" > /dev/null; do sleep 60; done

# 1. finish CUB at seed 42
step scripts/run_recipe.sh cub200 30

# 2. fused eval of the STANDARD Phase B on the recipe backbones (ablation row:
#    held-out training off). mini already has it.
std_fuse() {
  local ds=$1 bb=reserve_cm$(ep "$1")
  [ -f "checkpoints/$ds/$bb/fullbase/fuse0.5/incremental_results.json" ] && return 0
  "$PY" -m fscil.eval_incremental --dataset "$ds" --tag "$bb" --full_base --fuse 0 0.25 0.5 0.75 1 &&
    "$PY" scripts/commit_result.py --ckpt-dir "checkpoints/$ds/$bb" \
        --results-file fullbase/fuse0.5/incremental_results.json --label "$ds ablation: standard Phase B, fused lam=0.5"
}
for ds in cifar100 cub200; do step std_fuse "$ds"; done

# 3. component ablations of the full recipe (held-out Phase B on the seed-42
#    recipe backbone, one STAG-STI component off)
ablate() {
  local ds=$1 ab=$2 bb=reserve_cm$(ep "$1")
  local ho=${bb}_heldout dir="checkpoints/$1/ablate-$2/${bb}_heldout"
  [ -f "$dir/fullbase/fuse0.5/incremental_results.json" ] && return 0
  [ -f "$dir/heldout_val.json" ] ||
    "$PY" -m fscil.train_session0 --dataset "$ds" --heldout --backbone_tag "$bb" --tag "$ho" --ablate "$ab" || return 1
  "$PY" -m fscil.eval_incremental --dataset "$ds" --tag "$ho" --ablate "$ab" \
      --backbone_ckpt "checkpoints/$ds/$bb/backbone_base.pt" --stag_ckpt "$dir/stag_sti_session0_best.pt" \
      --full_base --fuse 0.5 &&
    "$PY" scripts/commit_result.py --ckpt-dir "$dir" --results-file fullbase/fuse0.5/incremental_results.json \
        --label "$ds ablation: no $ab, fused lam=0.5"
}
for ds in cifar100 miniimagenet cub200; do
  for ab in supcon topology agedecay; do step ablate "$ds" "$ab"; done
done

# 4. seed 1 on every dataset
for ds in cifar100 miniimagenet cub200; do step scripts/run_recipe.sh "$ds" "$(ep "$ds")" --seed 1; done

# 5. CIFAR-100 with the standard ResNet-20 (equal-capacity comparison)
step scripts/run_recipe.sh cifar100 100 --backbone resnet20

# 6. seed 2 on every dataset
for ds in cifar100 miniimagenet cub200; do step scripts/run_recipe.sh "$ds" "$(ep "$ds")" --seed 2; done

echo "[queue] $(date) ALL DONE"

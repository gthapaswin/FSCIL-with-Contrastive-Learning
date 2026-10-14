#!/usr/bin/env bash
# Seed check of the one configuration that passed the calibration control:
# CUB-200 faithful held-out head fused at lam=0.75 (seed 42: +2.0 A_N above
# NCM's offset curve). For each extra seed, train the faithful head on that
# seed's recipe backbone, sweep lam at b=0.5, and evaluate NCM (lam=1) over the
# offsets that bracket the fused point, so the curve gap is paired per seed.
# Waits for scripts/run_faithful.sh to finish. Resumable. Detached:
#   nohup caffeinate -i scripts/run_faithful_seeds.sh >> logs/faithful_seeds.log 2>&1 &
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"
ds=cub200; bb=reserve_cm30; tag=faithful_heldout
while pgrep -f run_faithful.sh > /dev/null; do sleep 120; done
for seed in 1 2; do
  dir="checkpoints/$ds/$tag/seed$seed"; bbck="checkpoints/$ds/$bb/seed$seed/backbone_base.pt"
  echo "[faithful-seeds] $(date) seed $seed"
  [ -f "$bbck" ] || { echo "no backbone $bbck"; continue; }
  [ -f "$dir/training_history.json" ] ||
    "$PY" -m fscil.train_session0 --dataset $ds --faithful --backbone_tag $bb --tag $tag \
        --heldout --residual_lambda 0 --seed $seed || continue
  ev=("$PY" -m fscil.eval_incremental --dataset $ds --faithful --tag $tag --seed $seed --full_base
      --backbone_ckpt "$bbck" --stag_ckpt "$dir/stag_sti_session0_best.pt")
  [ -f "$dir/fullbase/fuse0.75/incremental_results.json" ] || "${ev[@]}" --fuse 0 0.25 0.5 0.75 1 || continue
  for b in 0 1 1.5 2; do
    [ -f "$dir/fullbase/fuse1/nb$b/incremental_results.json" ] || "${ev[@]}" --fuse 1 --novel_bias $b || continue 2
  done
  "$PY" scripts/commit_result.py --ckpt-dir "$dir" --results-file fullbase/fuse0.75/incremental_results.json \
      --label "$ds faithful tech-doc head (heldout, seed $seed), fused lam=0.75" || true
done
echo "[faithful-seeds] $(date) ALL DONE"

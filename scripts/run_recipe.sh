#!/usr/bin/env bash
# Full recipe for one dataset, safe to relaunch after the process is killed
# (every step resumes or is skipped once done):
#   1. reserve + CutMix backbone (Phase A) + standard Phase B and evals  [run_longbb.sh]
#   2. held-out Phase B with the residual on that frozen backbone
#   3. fused evaluation (full-base) of the held-out-selected checkpoint;
#      lam=0.5 (the training value) is the headline, the rest is the sweep
# usage: scripts/run_recipe.sh <dataset> [backbone_epochs] [--seed N]
set -o pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"
DS=$1; EP=${2:-100}; shift; [ $# -gt 0 ] && shift
SEED_ARGS=("$@")
BB=reserve_cm$EP; HO=${BB}_heldout
SUB=""
if [ "${SEED_ARGS[0]}" = "--seed" ] && [ "${SEED_ARGS[1]}" != "42" ]; then SUB="/seed${SEED_ARGS[1]}"; fi
BBDIR="checkpoints/$DS/$BB$SUB"; HODIR="checkpoints/$DS/$HO$SUB"

scripts/run_longbb.sh "$DS" "$BB" "$EP" --reserve --reserve_cutmix "${SEED_ARGS[@]}" || exit 1

[ -f "$HODIR/heldout_val.json" ] || \
  caffeinate -i "$PY" -m fscil.train_session0 --dataset "$DS" --heldout --backbone_tag "$BB" \
      --tag "$HO" "${SEED_ARGS[@]}" || exit 1

[ -f "$HODIR/fullbase/fuse0.5/incremental_results.json" ] || \
  "$PY" -m fscil.eval_incremental --dataset "$DS" --tag "$HO" "${SEED_ARGS[@]}" \
      --backbone_ckpt "$BBDIR/backbone_base.pt" --stag_ckpt "$HODIR/stag_sti_session0_best.pt" \
      --full_base --fuse 0 0.25 0.5 0.75 1 || exit 1

"$PY" scripts/commit_result.py --ckpt-dir "$HODIR" --results-file fullbase/fuse0.5/incremental_results.json \
    --label "$DS reserve+CutMix backbone + held-out Phase B, fused lam=0.5 ($HO${SUB:+, ${SUB#/}})" || true
echo "[recipe] $(date) $DS DONE"

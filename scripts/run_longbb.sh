#!/usr/bin/env bash
# Long-backbone run, safe to relaunch after the process is killed:
#   Phase A/B resume from their per-epoch state; finished steps are skipped.
# usage: scripts/run_longbb.sh <dataset> <tag> <backbone_epochs> [extra train args...]
#   e.g. scripts/run_longbb.sh miniimagenet bb100 100
#        scripts/run_longbb.sh miniimagenet bb100 100 --seed 1
set -o pipefail   # no -u: macOS bash 3.2 treats an empty "${arr[@]}" as unbound
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD"
DS=$1; TAG=$2; EP=$3; shift 3
EXTRA=("$@")

CK="checkpoints/$DS/$TAG"
SEED=""; BACKBONE=""
for ((i = 0; i < ${#EXTRA[@]}; i++)); do
  [ "${EXTRA[$i]}" = "--seed" ] && SEED="${EXTRA[$((i + 1))]}"
  [ "${EXTRA[$i]}" = "--backbone" ] && BACKBONE="${EXTRA[$((i + 1))]}"
done
EVAL_ARGS=(--dataset "$DS" --tag "$TAG")
[ -n "$BACKBONE" ] && EVAL_ARGS+=(--backbone "$BACKBONE")
if [ -n "$SEED" ] && [ "$SEED" != "42" ]; then
  CK="$CK/seed$SEED"; EVAL_ARGS+=(--seed "$SEED")
fi

echo "[longbb] $(date) $DS tag=$TAG epochs=$EP ${EXTRA[*]} -> $CK"
# patience 0: no early stop, so the cosine schedule always runs to the end
caffeinate -i "$PY" -m fscil.train_session0 --dataset "$DS" --tag "$TAG" \
    --backbone_epochs "$EP" --patience 0 "${EXTRA[@]}" || exit 1

[ -f "$CK/incremental_results.json" ] || \
  "$PY" -m fscil.eval_incremental "${EVAL_ARGS[@]}" || exit 1
[ -f "$CK/transductive/incremental_results.json" ] || \
  "$PY" -m fscil.eval_incremental "${EVAL_ARGS[@]}" --transductive || exit 1

"$PY" scripts/commit_result.py --ckpt-dir "$CK" --label "$DS ${EP}-epoch backbone ($TAG${SEED:+, seed $SEED})" || true
echo "[longbb] $(date) DONE"

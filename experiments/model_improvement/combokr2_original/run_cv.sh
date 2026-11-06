#!/usr/bin/env bash
# CV over CGKronRLS early-stopping iterations {3,5,10} on the author's CV folds (paper protocol),
# with a fixed random subsample of validation surfaces per fold (compute budget).
# Usage: bash run_cv.sh SCENARIO SEED "FOLDS" NSUB
set -uo pipefail
SCEN=$1; SEED=$2; FOLDS=$3; NSUB=${4:-1500}
HERE=$(cd "$(dirname "$0")" && pwd)
R=$HERE/../../../runs/model_improvement/combokr2_original_v1
for F in $FOLDS; do
  echo "[$(date +%T)] CV $SCEN rseed$SEED fold $F"
  bash "$HERE/drun.sh" --name ckr2_cv_${SCEN}_${SEED}_${F} python -I /mnt/scripts/run_combokr2.py --scenario "$SCEN" --seed "$SEED" \
    --cv-fold "$F" --n-iters 3,5,10 --workers 12 --chunk 25 --max-test "$NSUB" --subsample-seed 0 \
    --tag "cv/${SCEN}_r${SEED}_f${F}" > "$R/logs/cv_${SCEN}_r${SEED}_f${F}.log" 2>&1
  rc=$?; echo "[$(date +%T)] exit=$rc ; $(grep -E '^iters=' "$R/logs/cv_${SCEN}_r${SEED}_f${F}.log" | cut -c1-60 | tr '\n' ' ')"
done

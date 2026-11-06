#!/usr/bin/env bash
# One scenario/seed end to end: [CV over {3,5,10} on the author's CV folds] -> select -> final test run
# -> post-hoc evaluation on all ground-truth definitions.
# Usage: bash run_scenario.sh SCENARIO SEED "CV_FOLDS|none" [FIXED_ITERS]
set -uo pipefail
SCEN=$1; SEED=$2; FOLDS=$3; FIXED=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
R=$HERE/../../../runs/model_improvement/combokr2_original_v1
ts() { date +%T; }
if [[ "$FOLDS" != "none" ]]; then
  bash "$HERE/run_cv.sh" "$SCEN" "$SEED" "$FOLDS" 1500
  CVDIRS=""; for F in $FOLDS; do CVDIRS="$CVDIRS /out/cv/${SCEN}_r${SEED}_f${F}"; done
  echo "[$(ts)] posthoc CV"; bash "$HERE/drun.sh" python -I /mnt/scripts/posthoc_eval.py $CVDIRS > "$R/logs/posthoc_cv_${SCEN}_r${SEED}.log" 2>&1
  ITERS=$(bash "$HERE/drun.sh" python -I /mnt/scripts/select_iters.py "$SCEN" "$SEED" 2>>"$R/logs/select_${SCEN}_r${SEED}.log" | tail -1 | tr -d '\r')
else
  ITERS=$FIXED
fi
echo "[$(ts)] TEST $SCEN rseed$SEED n_iters=$ITERS"
bash "$HERE/drun.sh" --name ckr2_test_${SCEN}_${SEED} python -I /mnt/scripts/run_combokr2.py --scenario "$SCEN" --seed "$SEED" \
  --n-iters "$ITERS" --workers 12 --chunk 250 --tag "test/${SCEN}_r${SEED}" > "$R/logs/test_${SCEN}_r${SEED}.log" 2>&1
echo "[$(ts)] exit=$? ; $(grep -E '^iters=' "$R/logs/test_${SCEN}_r${SEED}.log" | cut -c1-80)"
bash "$HERE/drun.sh" python -I /mnt/scripts/posthoc_eval.py "/out/test/${SCEN}_r${SEED}" > "$R/logs/posthoc_test_${SCEN}_r${SEED}.log" 2>&1
echo "[$(ts)] posthoc done: $(tail -1 "$R/logs/posthoc_test_${SCEN}_r${SEED}.log" | cut -c1-300)"

#!/usr/bin/env bash
# Remaining pipeline after the new-cell rseed1 CV (sequential; one 12-CPU container at a time).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
R=$HERE/../../../runs/model_improvement/combokr2_original_v1
ts() { date +%T; }
log() { echo "[$(ts)] $*"; }

until [ "$(grep -c 'exit=' "$R/logs/cv_newcell_r1_driver.log")" -ge 4 ]; do sleep 10; done
log "CV newcell r1 finished"
if [ -n "${1:-}" ]; then python "$1" && log "run_combokr2.py patched (ablation options)"; fi  # optional; run_combokr2.py already contains the options

# --- new-cell rseed1: CV post-hoc + selection + test ---------------------------------------------
bash "$HERE/drun.sh" python -I /mnt/scripts/posthoc_eval.py /out/cv/newcell_r1_f0 /out/cv/newcell_r1_f1 \
  /out/cv/newcell_r1_f2 /out/cv/newcell_r1_f3 > "$R/logs/posthoc_cv_newcell_r1.log" 2>&1
NC=$(bash "$HERE/drun.sh" python -I /mnt/scripts/select_iters.py newcell 1 2>>"$R/logs/select_newcell_r1.log" | tail -1 | tr -d '\r')
log "selected n_iters newcell r1 = $NC"
bash "$HERE/run_scenario.sh" newcell 1 none "$NC"

# --- ablations on a fixed 5000-surface test subsample (new-cell rseed1) ---------------------------
for spec in "train1000:--max-train 1000" "train10000:--max-train 10000" "candidates:--preimage candidates" "full_gd:"; do
  name=${spec%%:*}; extra=${spec#*:}
  log "ablation $name"
  bash "$HERE/drun.sh" python -I /mnt/scripts/run_combokr2.py --scenario newcell --seed 1 --n-iters "$NC" \
    --workers 12 --chunk 100 --max-test 5000 --subsample-seed 0 $extra --tag "ablation/newcell_r1_$name" \
    > "$R/logs/ablation_newcell_r1_$name.log" 2>&1
  bash "$HERE/drun.sh" python -I /mnt/scripts/posthoc_eval.py "/out/ablation/newcell_r1_$name" \
    > "$R/logs/posthoc_ablation_newcell_r1_$name.log" 2>&1
  log "ablation $name done: $(grep -E '^iters=' "$R/logs/ablation_newcell_r1_$name.log" | cut -c1-60)"
done

# --- new-combo rseed1: CV + selection + test ------------------------------------------------------
bash "$HERE/run_scenario.sh" newcombo 1 "0 1 2 3"
log "selected newcombo: $(cat "$R/cv/newcombo_r1_selection.json" 2>/dev/null | grep selected_n_iters)"

# --- new-cell rseeds 2-5 with the rseed1-selected n_iters -----------------------------------------
for S in 2 3 4 5; do
  bash "$HERE/run_scenario.sh" newcell "$S" none "$NC"
done
log "ALL DONE"

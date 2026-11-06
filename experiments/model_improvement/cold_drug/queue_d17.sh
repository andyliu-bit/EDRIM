#!/usr/bin/env bash
# D17: ablation arm A (run_cold_v2.py --ablation B11) on the D11 new-drug split, seeds 41-43, one at a time. A job
# starts only when >= 8 GB commit headroom is free (query of dispatch_ablation.sh). A seed whose metrics.json exists is
# skipped; a leftover output dir without metrics.json is renamed *_failed_<epoch> (kept, with its console log) and run
# again. cold_test is never predicted. One queue at a time (lock dir). Afterwards run compare_d17.py by hand.
# Launch hidden: explorer.exe experiments\model_improvement\cold_drug\start_d17_hidden.vbs
cd C:/LHZ/research/bisurfgnn-model-experiments || exit 1
PY=.venv/Scripts/python.exe
D=runs/model_improvement/bisurfgnn_v2_dev/cold_drug
LOG=$D/queue_d17.log
LOCK=$D/.queue_d17.lock
MIN_FREE_COMMIT_GB=8
exec >> $LOG 2>&1
say() { echo "[$(date '+%m-%d %H:%M:%S')] d17: $*"; }
free_commit_gb() {
  powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)' 2>/dev/null | tr -d '\r'
}
mkdir $LOCK 2>/dev/null || { say "another queue_d17 holds $LOCK (remove it if no queue is running); exit"; exit 1; }
trap 'rmdir $LOCK' EXIT

say "queue start (pid $$)"
for s in 41 42 43; do
  out=$D/ablation_B11_seed_$s
  [ -f $out/metrics.json ] && { say "seed $s already done, skipped"; continue; }
  if [ -d $out ]; then
    t=$(date +%s); mv $out ${out}_failed_$t
    [ -f ${out}_console.log ] && mv ${out}_console.log ${out}_failed_${t}_console.log
    say "seed $s: leftover without metrics.json kept as ${out}_failed_$t"
  fi
  # an empty or failed memory query compares as an error, so the loop keeps waiting
  until [ "$(free_commit_gb)" -ge $MIN_FREE_COMMIT_GB ] 2>/dev/null; do sleep 60; done
  say "seed $s start (free commit $(free_commit_gb) GB)"
  $PY -I -B experiments/model_improvement/cold_drug/run_cold_v2.py --arm both --seed $s --ablation B11 > ${out}_console.log 2>&1
  rc=$?   # save the exit code before the $(date) in say resets $?
  say "seed $s exit=$rc metrics=$([ -f $out/metrics.json ] && echo yes || echo no)"
done
say "all jobs done; next: compare_d17.py (by hand)"

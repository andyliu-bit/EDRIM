#!/usr/bin/env bash
# D16 (card H): Ours (run_cold_v2.py --partition) on the 4 new eval-paper partitions x seeds 41-43, at most 2 at a
# time. A job starts only when >= 8 GB commit headroom is free (query of dispatch_ablation.sh). A job whose
# metrics.json exists is skipped; a leftover output dir without metrics.json is renamed *_failed_<epoch> (kept, with its
# console log) and the job is run again; a job that crashes is retried once. One queue at a time (lock dir).
# Scoring is NOT started here: score_d16.py is run by hand once every run has finished.
# Launch hidden: explorer.exe experiments\model_improvement\cold_drug\start_d16_hidden.vbs
cd C:/LHZ/research/bisurfgnn-model-experiments || exit 1
PY=.venv/Scripts/python.exe
D=runs/eval_paper/cold_drug_partitions
LOG=$D/queue_d16.log
LOCK=$D/.queue_d16.lock
MAX_PAR=2
MIN_FREE_COMMIT_GB=8
exec >> $LOG 2>&1
say() { echo "[$(date '+%m-%d %H:%M:%S')] d16: $*"; }
free_commit_gb() {
  powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)' 2>/dev/null | tr -d '\r'
}
# an empty or failed memory query compares as an error, so the loop keeps waiting
wait_memory() { until [ "$(free_commit_gb)" -ge $MIN_FREE_COMMIT_GB ] 2>/dev/null; do sleep 60; done; }
mkdir $LOCK 2>/dev/null || { say "another queue_d16 holds $LOCK (remove it if no queue is running); exit"; exit 1; }
trap 'rmdir $LOCK' EXIT

run_one() {   # partition seed
  local out=$D/p$1/ours_both_seed_$2 t
  if [ -d $out ] && [ ! -f $out/metrics.json ]; then
    t=$(date +%s); mv $out ${out}_failed_$t
    [ -f ${out}_console.log ] && mv ${out}_console.log ${out}_failed_${t}_console.log
    say "p$1 seed $2: leftover without metrics.json kept as ${out}_failed_$t"
  fi
  say "p$1 seed $2 start (free commit $(free_commit_gb) GB)"
  $PY -I -B experiments/model_improvement/cold_drug/run_cold_v2.py --arm both --seed $2 --partition $1 > ${out}_console.log 2>&1
}

job() {   # partition seed; save $? before any $(date) resets it
  local out=$D/p$1/ours_both_seed_$2 rc
  run_one $1 $2; rc=$?
  if [ ! -f $out/metrics.json ]; then
    say "p$1 seed $2 exit=$rc metrics=no, retry once"
    wait_memory; run_one $1 $2; rc=$?
  fi
  say "p$1 seed $2 exit=$rc metrics=$([ -f $out/metrics.json ] && echo yes || echo no)"
}

say "queue start (pid $$)"
for p in 20261018 20261019 20261020 20261021; do
  for s in 41 42 43; do
    [ -f $D/p$p/ours_both_seed_$s/metrics.json ] && { say "p$p seed $s already done, skipped"; continue; }
    until [ "$(jobs -rp | wc -l)" -lt $MAX_PAR ] && [ "$(free_commit_gb)" -ge $MIN_FREE_COMMIT_GB ] 2>/dev/null; do sleep 60; done
    job $p $s &
    sleep 300   # data loading takes minutes; let the new run's memory show before the next slot check
  done
done
wait
say "all jobs done; next: score_d16.py (by hand, once)"

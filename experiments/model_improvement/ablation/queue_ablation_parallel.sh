#!/usr/bin/env bash
# D12 ablation queue, one stream of up to two (user 10-16: ablation runs in parallel with card D, two streams after
# card D). Usage: bash queue_ablation_parallel.sh STREAM START
#   START=after_xgb    wait until no run_cold_xgb.py process is left (the CPU XGBoost chain of card D)
#   START=after_cardD  wait until the card-D GPU queue has finished (DeepSynBa metrics.json exists)
# Each stream takes the next unclaimed job in the original order (seed 41 B1..B9, then 42, then 43); a job is claimed
# by an atomic mkdir under ablation/.claims, so two streams never run the same job. Before every job the stream waits
# until at least MIN_FREE_COMMIT_GB of commit headroom is free (Windows commit limit = RAM + page file).
# A failed job keeps its claim and is not retried automatically; its console log shows the error.
STREAM=$1; START=$2
MIN_FREE_COMMIT_GB=12
cd C:/LHZ/research/bisurfgnn-model-experiments
A=runs/model_improvement/bisurfgnn_v2_dev/ablation
DSB_DONE=runs/model_improvement/bisurfgnn_v2_dev/cold_drug/deepsynba_seed_10/metrics.json
LOG=$A/queue_stream_${STREAM}.log
mkdir -p $A/.claims

free_commit_gb() {
  powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)' 2>/dev/null | tr -d '\r'
}
xgb_running() {
  powershell.exe -NoProfile -Command '@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "run_cold_[x]gb" }).Count' 2>/dev/null | tr -d '\r'
}
say() { echo "[$(date '+%m-%d %H:%M:%S')] stream $STREAM: $*" >> $LOG; }

say "waiting for start condition $START"
if [ "$START" = "after_xgb" ]; then
  quiet=0
  while [ $quiet -lt 2 ]; do                     # two consecutive checks, 60 s apart, with no XGBoost process
    if [ "$(xgb_running)" = "0" ]; then quiet=$((quiet + 1)); else quiet=0; fi
    sleep 60
  done
elif [ "$START" = "after_cardD" ]; then
  until [ -f $DSB_DONE ]; do sleep 60; done
fi
say "start condition met"

for s in 41 42 43; do for arm in B1 B2 B3 B4 B5 B6 B7 B8 B9; do
  job=${arm}_seed_${s}
  [ -f $A/$job/metrics.json ] && continue
  mkdir $A/.claims/$job 2>/dev/null || continue
  until [ "$(free_commit_gb)" -ge $MIN_FREE_COMMIT_GB ] 2>/dev/null; do say "$job waits for memory (free commit $(free_commit_gb) GB)"; sleep 120; done
  say "$job start (free commit $(free_commit_gb) GB)"
  .venv/Scripts/python.exe -I -B experiments/model_improvement/ablation/run_ablation.py --arm $arm --seed $s > $A/${job}_console.log 2>&1
  say "$job exit=$?"
done; done
say "no unclaimed jobs left"

#!/usr/bin/env bash
# Runs ablation jobs in the given priority order, at most MAX_PAR at a time. The count includes every running
# run_ablation.py job, also ones started by the older queue streams. Each job starts only when at least
# MIN_FREE_COMMIT_GB of commit headroom is free. Jobs that already have results or a claim (ablation/.claims) are
# skipped, so a restart is safe. 10-19: the user asked to get one result for every arm first and push repeat seeds
# later, so the arm without any result (B10) goes first.
# Usage: bash dispatch_ablation.sh JOB [JOB ...]      e.g. B10_seed_41 B10_seed_42 B6_seed_43
MAX_PAR=${MAX_PAR:-3}   # env override only for a one-off extra run (10-21: B5_seed_44 rerun as a 4th job)
MIN_FREE_COMMIT_GB=8   # 10-21: 12 -> 8, Docker VM (another project's postgres) holds ~5.7 GB; one run needs ~5-7 GB
cd C:/LHZ/research/bisurfgnn-model-experiments
A=runs/model_improvement/bisurfgnn_v2_dev/ablation
HELPER=experiments/model_improvement/ablation/ablation_procs.ps1
LOG=$A/queue_dispatch.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] dispatch: $*" >> $LOG; }
free_commit_gb() {
  powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)' 2>/dev/null | tr -d '\r'
}
running() { powershell.exe -NoProfile -File $HELPER -Mode list 2>/dev/null | tr -d '\r' | grep -c . ; }
mkdir -p $A/.claims
say "start: $*"
# A claim without results whose run is not alive comes from an interrupted run (reboot, closed window, crash): keep its
# leftovers under *_interrupted_<stamp> and release the claim so it is run again. Assumes only one dispatcher runs.
alive=$(powershell.exe -NoProfile -File $HELPER -Mode list 2>/dev/null | tr -d '\r')
stamp=$(date +%m%d_%H%M)
for job in "$@"; do
  if [ -d $A/.claims/$job ] && [ ! -f $A/$job/metrics.json ] && ! grep -qx "$job" <<< "$alive"; then
    [ -d $A/$job ] && mv $A/$job $A/${job}_interrupted_$stamp
    [ -f $A/${job}_console.log ] && mv $A/${job}_console.log $A/${job}_interrupted_${stamp}_console.log
    rmdir $A/.claims/$job
    say "$job: stale claim from an interrupted run, leftovers kept as ${job}_interrupted_$stamp, will run again"
  fi
done
for job in "$@"; do
  [ -f $A/$job/metrics.json ] && { say "$job already has results, skipped"; continue; }
  [ -d $A/.claims/$job ] && { say "$job already claimed, skipped"; continue; }
  until [ "$(running)" -lt $MAX_PAR ] && [ "$(free_commit_gb)" -ge $MIN_FREE_COMMIT_GB ] 2>/dev/null; do sleep 60; done
  mkdir $A/.claims/$job 2>/dev/null || { say "$job claimed by someone else, skipped"; continue; }
  arm=${job%_seed_*}; s=${job#*_seed_}
  say "$job start (free commit $(free_commit_gb) GB)"
  # Save the exit code first: the $(date) inside the log line would otherwise reset $? to 0 (10-21: a CUDA OOM of
  # B5_seed_44 was logged as exit=0). The metrics check makes a crash visible even if the exit code is lost.
  ( .venv/Scripts/python.exe -I -B experiments/model_improvement/ablation/run_ablation.py --arm $arm --seed $s > $A/${job}_console.log 2>&1
    rc=$?; [ -f $A/$job/metrics.json ] && m=yes || m=no
    say "$job exit=$rc metrics=$m" ) &
  sleep 90   # let the new run appear in the process list before the next slot check
done
wait
say "all jobs done"

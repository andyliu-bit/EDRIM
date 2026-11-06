#!/usr/bin/env bash
# Re-run ablation jobs whose first attempt was lost. 10-17: B6_seed_41 (its first start collided with a duplicate run
# launched by the leftover old queue_ablation.sh, which was then killed after ~4 minutes; see the progress notes).
# Usage: bash retry_jobs.sh B6_seed_41 [more jobs]
# Starts only when a GPU slot is free (either stream's log ends with "no unclaimed jobs left") and at least 12 GB of
# commit headroom is free, then claims each job through ablation/.claims like the streams do.
cd C:/LHZ/research/bisurfgnn-model-experiments
A=runs/model_improvement/bisurfgnn_v2_dev/ablation
LOG=$A/queue_retry.log
free_commit_gb() {
  powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)' 2>/dev/null | tr -d '\r'
}
say() { echo "[$(date '+%m-%d %H:%M:%S')] retry: $*" >> $LOG; }
slot_free() {
  tail -1 $A/queue_stream_1.log | grep -q "no unclaimed jobs left" || tail -1 $A/queue_stream_2.log | grep -q "no unclaimed jobs left"
}

say "waiting for a free stream slot for: $*"
until slot_free; do sleep 120; done
for job in "$@"; do
  arm=${job%_seed_*}; s=${job#*_seed_}
  [ -f $A/$job/metrics.json ] && { say "$job already has results"; continue; }
  mkdir $A/.claims/$job 2>/dev/null || { say "$job already claimed"; continue; }
  until [ "$(free_commit_gb)" -ge 12 ] 2>/dev/null; do say "$job waits for memory (free commit $(free_commit_gb) GB)"; sleep 120; done
  say "$job start (free commit $(free_commit_gb) GB)"
  .venv/Scripts/python.exe -I -B experiments/model_improvement/ablation/run_ablation.py --arm $arm --seed $s > $A/${job}_console.log 2>&1
  say "$job exit=$?"
done
say "done"

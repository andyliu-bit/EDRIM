#!/usr/bin/env bash
# Stops ablation runs that crashed but did not exit, so their stream can move on (10-18: B7_seed_42 hung for about six
# hours after a CUDA out-of-memory error). Every 5 minutes, for each running run_ablation.py job it checks the size of
# the job's console log (sizes, not timestamps: Windows updates the timestamp of a file that is still open lazily):
#   - the log contains a Traceback and has not grown for 2 checks (about 10 minutes), or
#   - the log has not grown for 9 checks (about 45 minutes; one epoch takes at most about 5 minutes),
# then the job's python processes are stopped and the action is logged. The stream records a non-zero exit and moves
# on; the stopped job has to be re-run with retry_jobs.sh. This script never starts or deletes anything.
cd C:/LHZ/research/bisurfgnn-model-experiments
A=runs/model_improvement/bisurfgnn_v2_dev/ablation
HELPER=experiments/model_improvement/ablation/ablation_procs.ps1
LOG=$A/queue_watchdog.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] watchdog: $*" >> $LOG; }
declare -A last_size same
say "started"
while true; do
  for job in $(powershell.exe -NoProfile -File $HELPER -Mode list 2>/dev/null | tr -d '\r'); do
    log=$A/${job}_console.log
    [ -f "$log" ] || continue
    size=$(stat -c %s "$log")
    if [ "${last_size[$job]}" = "$size" ]; then same[$job]=$(( ${same[$job]:-0} + 1 )); else same[$job]=0; last_size[$job]=$size; fi
    if grep -q "Traceback" "$log" && [ "${same[$job]}" -ge 2 ]; then
      say "$job: Traceback in its log and no new output for ${same[$job]} checks; stopping it"
      powershell.exe -NoProfile -File $HELPER -Mode kill -Job $job 2>&1 | tr -d '\r' >> $LOG
    elif [ "${same[$job]}" -ge 9 ]; then
      say "$job: no new output for ${same[$job]} checks; stopping it"
      powershell.exe -NoProfile -File $HELPER -Mode kill -Job $job 2>&1 | tr -d '\r' >> $LOG
    fi
  done
  sleep 300
done

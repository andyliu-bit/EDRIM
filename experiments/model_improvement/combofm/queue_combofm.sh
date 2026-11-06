#!/usr/bin/env bash
# D21 (docs/COMBOFM_CARD_20261027.md): comboFM on the NCI public split, validation only, two GPU runs at a time.
# Stage 1: reg in {100,1000,10000,100000} (written as %g, matching the run directories) at rank 50; stage 2: rank in {25,75,100} at the best reg; both seed 41,
# 100 epochs. Stage 3: the best (reg, rank) with seeds 41-45, 200 epochs. Then select_combofm.py writes the summary.
# A run whose metrics.json exists is skipped; run_combofm.py creates its output dir only at the end, so a crashed run
# leaves nothing and is retried once. A run starts only with >= 12 GiB free commit. One queue at a time (lock dir).
# Launch hidden: explorer.exe experiments\model_improvement\combofm\start_combofm_hidden.vbs
cd C:/LHZ/research/bisurfgnn-model-experiments || exit 1
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe -I -B"
X=experiments/model_improvement/combofm
D=runs/model_improvement/combofm/nci
LOCK=$D/.queue.lock
mkdir -p $D
exec >> $D/queue.log 2>&1
say() { echo "[$(date '+%m-%d %H:%M:%S')] d21: $*"; }
mkdir $LOCK 2>/dev/null || { say "another queue holds $LOCK (remove it if no queue is running); exit"; exit 1; }
trap 'rmdir $LOCK' EXIT

free_gib() { powershell.exe -NoProfile -Command "[int][math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" | tr -d ''; }
wait_mem() { local f; while f=$(free_gib); [ "${f:-0}" -lt 12 ]; do say "waiting: free commit ${f} GiB < 12"; sleep 300; done; }

run() {   # reg rank seed epochs; succeeds when metrics.json exists
  local m=$D/reg$1_rank$2/seed_$3_ep$4/metrics.json try rc
  [ -f $m ] && { say "reg $1 rank $2 seed $3 ep $4 already done, skipped"; return 0; }
  for try in 1 2; do
    wait_mem
    say "reg $1 rank $2 seed $3 ep $4 start (try $try)"
    $PY $X/run_combofm.py --reg $1 --rank $2 --seed $3 --epochs $4 >> $D/reg$1_rank$2_seed$3_ep$4_console.log 2>&1; rc=$?
    say "reg $1 rank $2 seed $3 ep $4 exit=$rc metrics=$([ -f $m ] && echo yes || echo no)"
    [ -f $m ] && return 0
  done
  return 1
}

say "queue start (pid $$)"
( run 100 50 41 100 ) & p1=$!; sleep 120; ( run 1000 50 41 100 ) & p2=$!
wait $p1 && wait $p2 || { say "stop: stage 1 (reg 100/1000) failed"; exit 1; }
( run 10000 50 41 100 ) & p1=$!; sleep 120; ( run 100000 50 41 100 ) & p2=$!
wait $p1 && wait $p2 || { say "stop: stage 1 (reg 10000/100000) failed"; exit 1; }
reg=$($PY $X/select_combofm.py --pick-reg) || { say "stop: --pick-reg failed"; exit 1; }
reg=${reg//$'\r'/}
say "best reg at rank 50: $reg"
( run $reg 25 41 100 ) & p1=$!; sleep 120; ( run $reg 75 41 100 ) & p2=$!
wait $p1 && wait $p2 || { say "stop: stage 2 (rank 25/75) failed"; exit 1; }
run $reg 100 41 100 || { say "stop: stage 2 (rank 100) failed"; exit 1; }
best=$($PY $X/select_combofm.py --pick-best) || { say "stop: --pick-best failed"; exit 1; }
read breg brank <<< "${best//$'\r'/}"
say "best setting: reg $breg rank $brank"
( run $breg $brank 41 200 ) & p1=$!; sleep 120; ( run $breg $brank 42 200 ) & p2=$!
wait $p1 && wait $p2 || { say "stop: final seeds 41/42 failed"; exit 1; }
( run $breg $brank 43 200 ) & p1=$!; sleep 120; ( run $breg $brank 44 200 ) & p2=$!
wait $p1 && wait $p2 || { say "stop: final seeds 43/44 failed"; exit 1; }
run $breg $brank 45 200 || { say "stop: final seed 45 failed"; exit 1; }
$PY $X/select_combofm.py && say "summary written: $D/summary.md" || say "summary failed"
say "queue done"

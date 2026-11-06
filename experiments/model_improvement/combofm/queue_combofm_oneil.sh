#!/usr/bin/env bash
# D21 (docs/COMBOFM_CARD_20261027.md): comboFM on O'Neil oneil-pairout-v1, one GPU run at a time (runs beside the
# NCI queue). Final 200-epoch runs write validation and test predictions (--final); the test is scored once later by
# experiments/oneil/eval/evaluate_combofm_test.py, not by this queue.
# Stage 1: reg in {100,1000,10000,100000} (written as %g, matching the run directories) at rank 50; stage 2: rank in {25,75,100} at the best reg; both seed 41,
# 100 epochs. Stage 3: the best (reg, rank) with seeds 41-45, 200 epochs, predictions written.
# A run whose metrics.json exists is skipped; run_combofm.py creates its output dir only at the end, so a crashed run
# leaves nothing and is retried once. A run starts only with >= 12 GiB free commit. One queue at a time (lock dir).
# Launch hidden: explorer.exe experiments\model_improvement\combofm\start_combofm_oneil_hidden.vbs
cd C:/LHZ/research/bisurfgnn-model-experiments || exit 1
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"  # no -I: the card E audit hook breaks imports under -I
X=experiments/model_improvement/combofm
D=runs/model_improvement/combofm/oneil
LOCK=$D/.queue.lock
mkdir -p $D
exec >> $D/queue.log 2>&1
say() { echo "[$(date '+%m-%d %H:%M:%S')] d21-oneil: $*"; }
mkdir $LOCK 2>/dev/null || { say "another queue holds $LOCK (remove it if no queue is running); exit"; exit 1; }
trap 'rmdir $LOCK' EXIT

free_gib() { powershell.exe -NoProfile -Command "[int][math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" | tr -cd '0-9'; }
wait_mem() { local f; while f=$(free_gib); [ "${f:-0}" -lt 12 ]; do say "waiting: free commit ${f} GiB < 12"; sleep 300; done; }

run() {   # reg rank seed epochs; succeeds when metrics.json exists
  local m=$D/reg$1_rank$2/seed_$3_ep$4/metrics.json try rc
  [ -f $m ] && { say "reg $1 rank $2 seed $3 ep $4 already done, skipped"; return 0; }
  for try in 1 2; do
    wait_mem
    say "reg $1 rank $2 seed $3 ep $4 start (try $try)"
    fin=; [ "$4" = 200 ] && fin=--final
    $PY $X/run_combofm_oneil.py --reg $1 --rank $2 --seed $3 --epochs $4 $fin >> $D/reg$1_rank$2_seed$3_ep$4_console.log 2>&1; rc=$?
    say "reg $1 rank $2 seed $3 ep $4 exit=$rc metrics=$([ -f $m ] && echo yes || echo no)"
    [ -f $m ] && return 0
  done
  return 1
}

say "queue start (pid $$)"
for reg in 100 1000 10000 100000; do run $reg 50 41 100 || { say "stop: reg $reg failed"; exit 1; }; done
reg=$($PY $X/select_combofm_oneil.py --pick-reg) || { say "stop: --pick-reg failed"; exit 1; }
reg=${reg//$'
'/}
say "best reg at rank 50: $reg"
for rank in 25 75 100; do run $reg $rank 41 100 || { say "stop: rank $rank failed"; exit 1; }; done
best=$($PY $X/select_combofm_oneil.py --pick-best) || { say "stop: --pick-best failed"; exit 1; }
read breg brank <<< "${best//$'
'/}"
say "best setting: reg $breg rank $brank"
for seed in 41 42 43 44 45; do run $breg $brank $seed 200 || { say "stop: final seed $seed failed"; exit 1; }; done
say "queue done (test not scored here)"

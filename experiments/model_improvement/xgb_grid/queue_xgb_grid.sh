#!/usr/bin/env bash
# D20 (docs/XGB_GRID_CARD_20261026.md): XGBoost + identity validation grid, one GPU run at a time. The 16 grid configs
# with seed 41, then the best one (summarize.py --pick: val surface-macro, then fewer trees, then config id) with seeds
# 42-45 (seed 41 reused), then summarize.py writes summary.json + summary.md. A run whose metrics.json exists is skipped;
# run_grid.py creates its output dir only at the end, so a crashed run leaves nothing and is retried once. Split 3 is
# never read. One queue at a time (lock dir).
# Launch hidden: explorer.exe experiments\model_improvement\xgb_grid\start_xgb_grid_hidden.vbs
cd C:/LHZ/research/bisurfgnn-model-experiments || exit 1
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe -I -B"
X=experiments/model_improvement/xgb_grid
D=runs/model_improvement/bisurfgnn_v2_dev/xgb_grid
LOCK=$D/.queue.lock
mkdir -p $D
exec >> $D/queue.log 2>&1
say() { echo "[$(date '+%m-%d %H:%M:%S')] d20: $*"; }
mkdir $LOCK 2>/dev/null || { say "another queue holds $LOCK (remove it if no queue is running); exit"; exit 1; }
trap 'rmdir $LOCK' EXIT

run() {   # config seed; succeeds when metrics.json exists
  local m=$D/$1/seed_$2/metrics.json rc try
  [ -f $m ] && { say "$1 seed $2 already done, skipped"; return 0; }
  for try in 1 2; do
    say "$1 seed $2 start (try $try)"
    $PY $X/run_grid.py --config $1 --seed $2 >> $D/$1_seed_$2_console.log 2>&1; rc=$?
    say "$1 seed $2 exit=$rc metrics=$([ -f $m ] && echo yes || echo no)"
    [ -f $m ] && return 0
  done
  return 1
}

say "queue start (pid $$)"
configs=$($PY $X/summarize.py --list) || { say "stop: --list failed"; exit 1; }
for c in ${configs//$'\r'/}; do
  run $c 41 || { say "stop: $c seed 41 failed twice"; exit 1; }
done
best=$($PY $X/summarize.py --pick) || { say "stop: --pick failed"; exit 1; }
best=${best//$'\r'/}
say "best config by val surface-macro (seed 41): $best"
for s in 42 43 44 45; do
  run $best $s || { say "stop: $best seed $s failed twice"; exit 1; }
done
$PY $X/summarize.py && say "summary written: $D/summary.md" || say "summary failed"
say "queue done"

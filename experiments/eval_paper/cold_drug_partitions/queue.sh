#!/usr/bin/env bash
# Eval-paper supplement G: 12 XGBoost runs (4 new partitions x seeds 41-43), sequential. XGBoost here is CPU hist with
# 6 threads (gate config); no GPU is used. A run whose metrics.json exists is skipped; then the final scoring runs.
# Disk: each cap-20000 model.json is ~2 GB and score.py does not need it, so it is gzip-compressed (kept) after the run.
# A directory left by a crashed run (no metrics.json) is renamed *_failed_<epoch> (kept) and the run is retried once.
cd C:/LHZ/research/bisurfgnn-model-experiments
PY=.venv/Scripts/python.exe
D=runs/eval_paper/cold_drug_partitions
run_one() {
  local p=$1 s=$2 out=$D/p$1/xgb_seed_$2
  [ -d $out ] && [ ! -f $out/metrics.json ] && mv $out ${out}_failed_$(date +%s)
  $PY -I -B experiments/eval_paper/cold_drug_partitions/run_xgb.py --partition $p --seed $s > $D/p$p/xgb_seed_${s}_console.log 2>&1
}
echo "[$(date +%T)] queue start"
for p in 20261018 20261019 20261020 20261021; do
  for s in 41 42 43; do
    out=$D/p$p/xgb_seed_$s
    [ -f $out/metrics.json ] && { echo "[$(date +%T)] p$p seed $s already done"; continue; }
    run_one $p $s || { echo "[$(date +%T)] p$p seed $s failed, retry once"; run_one $p $s; }
    rc=$?; echo "[$(date +%T)] p$p seed $s exit=$rc metrics=$([ -f $out/metrics.json ] && echo yes || echo no)"
    [ -f $out/model.json ] && gzip -1 $out/model.json && echo "[$(date +%T)] p$p seed $s model.json gzipped"
  done
done
$PY -I experiments/eval_paper/cold_drug_partitions/score.py > $D/score_console.log 2>&1
rc=$?; echo "[$(date +%T)] score exit=$rc"   # save $? before $(date) resets it

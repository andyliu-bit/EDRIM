#!/usr/bin/env bash
# Runs one O'Neil stream of jobs sequentially (card E v2, D13). Every entry script refuses to overwrite its own
# predictions, so re-running a stream after an interruption only redoes jobs without outputs (clean partial files first).
# Usage: bash experiments/oneil/queue_oneil.sh ours|baselines|ddprism|deepsynba_4x4|deepsynba_5x5|ours_e500_a|ours_e500_b
cd C:/LHZ/research/bisurfgnn-model-experiments
unset ONEIL_PRED_ROOT
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PY=.venv/Scripts/python.exe
STREAM=$1
LOG=runs/oneil_v1/logs/queue_$STREAM.log
mkdir -p runs/oneil_v1/logs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }
run() { say "start: $*"; "$@" >> runs/oneil_v1/logs/queue_${STREAM}_stdout.log 2>&1; say "exit=$? : $*"; }
case $1 in
  ours)      for s in 41 42 43 44 45; do run $PY experiments/oneil/ours/run_ours.py --seed $s; done ;;
  baselines) say "start: make_hsa"; $PY experiments/oneil/baselines/make_hsa.py > runs/oneil_v1/logs/hsa_deterministic.log 2>&1; say "exit=$? : make_hsa"
             for s in 41 42 43 44 45; do run $PY experiments/oneil/baselines/run_xgb_id.py --seed $s; done ;;
  ddprism)   for s in 41 42 43 44 45; do run $PY experiments/oneil/ddprism/run_ddprism.py --seed $s; done ;;
  ours_e500_a) for s in 41 43 45; do run $PY experiments/oneil/ours/run_ours.py --seed $s --patience 50; done ;;
  ours_e500_b) for s in 42 44; do run $PY experiments/oneil/ours/run_ours.py --seed $s --patience 50; done ;;
  # D14 cause checks (val only): H1 lookup anchors, H3a batch 16
  h1_xgb)  export ONEIL_DATA_ROOT=runs/oneil_v1/analysis/h1/data ONEIL_PRED_ROOT=runs/oneil_v1/analysis/h1/preds
           run $PY experiments/oneil/baselines/run_xgb_id.py --seed 41 ;;
  h1_ours) export ONEIL_DATA_ROOT=runs/oneil_v1/analysis/h1/data ONEIL_PRED_ROOT=runs/oneil_v1/analysis/h1/preds
           run $PY experiments/oneil/ours/run_ours.py --seed 41 --patience 50 ;;
  h3a)     export ONEIL_PRED_ROOT=runs/oneil_v1/analysis/h3a/preds
           run $PY experiments/oneil/ours/run_ours.py --seed 41 --patience 50 --batch-surfaces 16 ;;
  deepsynba_4x4|deepsynba_5x5) for s in 41 42 43 44 45; do
               run $PY experiments/oneil/deepsynba/run_deepsynba.py --method $1 --seed $s; done ;;
  *) echo "unknown stream $1"; exit 2 ;;
esac
say "stream $1 done"

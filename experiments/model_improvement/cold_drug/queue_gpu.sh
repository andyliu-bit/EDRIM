#!/usr/bin/env bash
# D11 GPU queue: waits for the D10 DD-PRiSM seeds, then v2 both 41-45, v2 random 41-43, DeepSynBa seed 10.
cd C:/LHZ/research/bisurfgnn-model-experiments
D=runs/model_improvement/bisurfgnn_v2_dev/cold_drug
PY=.venv/Scripts/python.exe
for i in $(seq 1 720); do [ -f runs/model_improvement/bisurfgnn_v2_dev/ddprism_measured_mono/seed_45/metrics.json ] && break; sleep 30; done
echo "[$(date +%T)] queue start"
for s in 41 42 43 44 45; do $PY -I -B experiments/model_improvement/cold_drug/run_cold_v2.py --arm both --seed $s > $D/v2_both_seed_${s}_console.log 2>&1; rc=$?; echo "[$(date +%T)] both $s exit=$rc"; done
for s in 41 42 43; do $PY -I -B experiments/model_improvement/cold_drug/run_cold_v2.py --arm random --seed $s > $D/v2_random_seed_${s}_console.log 2>&1; rc=$?; echo "[$(date +%T)] random $s exit=$rc"; done
C:/LHZ/research/bisurfgnn/.venvs/deepsynba/Scripts/python.exe -I experiments/model_improvement/cold_drug/run_cold_deepsynba.py > $D/deepsynba_console.log 2>&1; rc=$?; echo "[$(date +%T)] deepsynba exit=$rc"

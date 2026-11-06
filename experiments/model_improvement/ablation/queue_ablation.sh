#!/usr/bin/env bash
# D12 GPU queue: waits for the D11 DeepSynBa run (end of the card-D queue), then B1-B9 x seeds 41-43.
cd C:/LHZ/research/bisurfgnn-model-experiments
A=runs/model_improvement/bisurfgnn_v2_dev/ablation
mkdir -p $A
for i in $(seq 1 5760); do [ -f runs/model_improvement/bisurfgnn_v2_dev/cold_drug/deepsynba_seed_10/metrics.json ] && break; sleep 30; done
echo "[$(date +%T)] ablation queue start"
for s in 41 42 43; do for arm in B1 B2 B3 B4 B5 B6 B7 B8 B9; do
  .venv/Scripts/python.exe -I -B experiments/model_improvement/ablation/run_ablation.py --arm $arm --seed $s > $A/${arm}_seed_${s}_console.log 2>&1
  rc=$?; echo "[$(date +%T)] $arm $s exit=$rc"   # save $? before $(date) resets it
done; done

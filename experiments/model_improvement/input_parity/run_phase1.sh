#!/usr/bin/env bash
# BiSurfGNN v2 phase 1 (input parity), formal screening runs: 3 arms x seeds 41, 42 = 6 runs, validation only.
# Protocol: docs/BISURFGNN_V2_DEV_PROTOCOL.md section 3 (frozen). Implementation: docs/BISURFGNN_V2_PHASE1_IMPLEMENTATION.md.
#
# The six jobs run strictly one after another (one GPU job at a time), arm by arm, so that every finished pair of
# seeds already gives a complete screening decision for its arm. Each job writes a fresh directory
# runs/model_improvement/bisurfgnn_v2_dev/phase1/<arm>_seed_<seed>/ plus <arm>_seed_<seed>_console.log next to it;
# the preflight refuses to start if any of the twelve targets exists, and the runner itself refuses an existing
# directory. The chain stops at the first failing job.
#
# Usage (Git Bash, from anywhere):  bash experiments/model_improvement/input_parity/run_phase1.sh
# Afterwards:  .venv/Scripts/python.exe -I -B experiments/model_improvement/input_parity/summarize_phase1.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY=".venv/Scripts/python.exe"
RUNNER="experiments/model_improvement/input_parity/run_parity.py"
OUT="runs/model_improvement/bisurfgnn_v2_dev/phase1"
ARMS=(molformer genes976 both)
SEEDS=(41 42)

[ -x "$PY" ] || { echo "missing interpreter $PY" >&2; exit 1; }
[ -f "$RUNNER" ] || { echo "missing runner $RUNNER" >&2; exit 1; }
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for target in "$OUT/${arm}_seed_${seed}" "$OUT/${arm}_seed_${seed}_console.log"; do
      if [ -e "$target" ]; then
        echo "refusing to start: $target already exists" >&2
        exit 1
      fi
    done
  done
done
mkdir -p "$OUT"

echo "phase 1 launcher start $(date '+%Y-%m-%dT%H:%M:%S%z')"
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    name="${arm}_seed_${seed}"
    echo "=== $(date '+%Y-%m-%dT%H:%M:%S%z') START $name"
    "$PY" -I -B "$RUNNER" --mode train --arm "$arm" --seed "$seed" --output-dir "$OUT/$name" 2>&1 \
      | tee "$OUT/${name}_console.log"
    echo "=== $(date '+%Y-%m-%dT%H:%M:%S%z') DONE $name"
  done
done
echo "phase 1 launcher finished $(date '+%Y-%m-%dT%H:%M:%S%z'); next: $PY -I -B experiments/model_improvement/input_parity/summarize_phase1.py"

#!/usr/bin/env bash
# Run a command in the comboKR2 reproduction container with read-only inputs and no network.
# Usage: bash drun.sh [--name NAME] python -I /mnt/scripts/<script>.py [args]
set -euo pipefail
NAME_ARGS=()
if [[ "${1:-}" == "--name" ]]; then NAME_ARGS=(--name "$2"); shift 2; fi
export MSYS_NO_PATHCONV=1
EXP=C:/LHZ/research/bisurfgnn-model-experiments
exec docker run --rm "${NAME_ARGS[@]}" --network none --cpus 12 --memory 12g \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" -e OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-1}" -e MKL_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  -v "$EXP/data/raw/model_comparators/combokr2:/mnt/code:ro" \
  -v "C:/LHZ/research/bisurfgnn/data/raw/external_benchmarks/combokr2_zenodo_14509938:/mnt/zenodo:ro" \
  -v "C:/LHZ/research/bisurfgnn/data/raw/nci_almanac:/mnt/almanac:ro" \
  -v "C:/LHZ/research/bisurfgnn/data/raw/cellminer:/mnt/cellminer:ro" \
  -v "$EXP/experiments/model_improvement/combokr2_original:/mnt/scripts:ro" \
  -v "$EXP/runs/model_improvement/combokr2_original_v1:/out" \
  combokr2-original:v1 "$@"

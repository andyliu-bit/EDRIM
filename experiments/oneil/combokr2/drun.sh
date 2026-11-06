#!/usr/bin/env bash
# O'Neil comboKR 2.0 (D13 supplement): run a command in combokr2-original:v1, no network, same limits as NCI (drun.sh).
# Only the response-free / train-only inputs are mounted; the author Zenodo folder (oneil_fitted_Braids.npy) is not.
# COMBOKR_OUT overrides the writable /out (smoke tests use the scratchpad).
# Usage: bash drun.sh [--name NAME] python -I /mnt/scripts/<script>.py [args]
set -euo pipefail
NAME_ARGS=()
if [[ "${1:-}" == "--name" ]]; then NAME_ARGS=(--name "$2"); shift 2; fi
export MSYS_NO_PATHCONV=1
EXP=C:/LHZ/research/bisurfgnn-model-experiments
OUT="${COMBOKR_OUT:-$EXP/runs/oneil_v1/combokr2}"
exec docker run --rm "${NAME_ARGS[@]}" --network none --cpus 12 --memory 12g \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" -e OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-1}" -e MKL_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  -v "$EXP/data/raw/model_comparators/combokr2:/mnt/code:ro" \
  -v "$EXP/experiments/model_improvement/combokr2_original:/mnt/nci:ro" \
  -v "$EXP/experiments/oneil/combokr2:/mnt/scripts:ro" \
  -v "$EXP/runs/oneil_v1/combokr2/inputs:/mnt/inputs:ro" \
  -v "$EXP/runs/oneil_v1/drug_features:/mnt/drug:ro" \
  -v "$EXP/runs/oneil_v1/cell_features:/mnt/cell:ro" \
  -v "$OUT:/out" \
  combokr2-original:v1 "$@"

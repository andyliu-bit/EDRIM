# Reproducibility status

This document describes the initial source snapshot, not a new experiment or a re-evaluation of the manuscript test sets.

## Available now

- The original `bisurfgnn` Python source, including the final parity/additive model classes.
- Historical configuration and experiment scripts, copied without changing model or evaluation logic.
- Data-source manifests and cell-name mappings.
- A focused set of existing synthetic model and statistical tests.

## Historical experiment dependencies

The historical runners were written in a workspace containing sibling `bisurfgnn` and `bisurfgnn-model-experiments` directories. Examples of dependencies that are not provided by a fresh clone include:

- `experiments/model_improvement/public_split_gate/gate_common.py` resolves its data workspace as `ROOT.parent / "bisurfgnn"`.
- `experiments/oneil/common.py` uses an absolute development-workspace path.
- Input-parity and long-schedule runners refer to earlier protocol bindings, prepared tables and saved validation histories under `runs/` and `docs/`.
- Final-evaluation scripts require the saved models, prepared matrices and prior run records, and contain safeguards against repeated evaluation.
- Comparator scripts require additional packages and pinned upstream files beyond the core installation extras. Their historical requirements files are included where available.

The original workspace's private notes, protocol records and run outputs were not copied into this public snapshot. The historical runners are therefore reference implementations rather than ready-to-run commands on a clean clone. Do not bypass their checks to make an evaluation run.

## Before claiming complete paper reproduction

1. Provide portable paths and a documented data-preparation sequence for each reported dataset and split.
2. Supply legally distributable split identifiers and feature/preprocessing specifications, with original data and code sources pinned.
3. Make historical training prerequisites recoverable without requiring private development records; retain test-access safeguards.
4. Provide the final per-experiment configurations, seeds and additional dependency versions.
5. Publish permitted saved predictions, per-seed metrics and optional weights, with the applicable licences and an actual deposit identifier.
6. Verify the documented clean-clone workflow, then assign a version tag to the verified snapshot.

No test-set responses were opened and no paper training or test evaluation was launched while preparing this repository.

## Safe local verification

```bash
python -m pip install -e ".[data,dev]"
python -m pytest
```

These checks validate code behaviour on synthetic inputs; they do not reproduce the paper's benchmark metrics. Detailed preparation-check results are recorded in `PREPARATION_CHECKS.md`.

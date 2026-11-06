# EDRIM

Research code for **EDRIM: Monotherapy-Informed Graph Interaction Learning for Predicting Dose–Response Matrices of Untested Pairs of Known Drugs**.

EDRIM predicts combination responses from a measured-monotherapy highest single agent (HSA) reference plus a learned correction. Molecular-graph and pathway-graph encoders, symmetric cross-drug exchange, cell-context fusion and a dose-conditioned decoder provide the correction. The intended setting is untested combinations of previously profiled drugs with available monotherapy measurements. Predictions support computational prioritisation and require experimental validation.

## Repository status

This is an initial source-code snapshot prepared on 2026-11-06. It includes the model implementation and historical experiment scripts. It is **not yet a self-contained reproduction package for every manuscript result**: the historical runners require prepared data, external sources and run records that are not bundled here. No archival release or Zenodo DOI has been issued.

The internal Python package remains named `bisurfgnn` to preserve imports and historical experiment paths. In particular, the synthetic CLI example uses an earlier model; it is an installation check, not the final EDRIM training protocol.

## Installation and local checks

Python 3.11 or newer is required. From the repository root:

```bash
python -m pip install -e ".[data,dev]"
python -m pytest
```

The included tests exercise model behaviour and paired statistics using synthetic inputs. They do not evaluate the manuscript test sets. The source snapshot was checked using Python 3.12, PyTorch 2.12.0+cu130 and pytest 9.0.3; the version ranges in `pyproject.toml` are not a lockfile for the paper experiments.

## Code map

| Component | Location |
|---|---|
| EDRIM model with external drug/cell inputs | `src/bisurfgnn/models/compact_bisurf_parity.py` — `ParityAdditiveCompactBiSurfGNN` |
| Additive HSA-reference output | `src/bisurfgnn/models/compact_bisurf_additive.py` |
| Graph, exchange, gate and decoder components | `src/bisurfgnn/models/` |
| Input processing and data utilities | `src/bisurfgnn/data/` |
| Historical NCI-ALMANAC training | `experiments/model_improvement/input_parity/` and `long_schedule/` |
| Historical main evaluation | `experiments/model_improvement/final_eval/evaluate_v2.py` |
| Comparator, ablation and sensitivity scripts | `experiments/model_improvement/` |
| O'Neil experiment scripts | `experiments/oneil/` |
| Additional manuscript analyses | `experiments/eval_paper/` |
| Configurations, including earlier development settings | `configs/` |
| Data-source manifests and name mappings | `data/raw_manifests/` and `data/mappings/` |

Development variants remain in the snapshot for tracing script dependencies. Their presence does not imply that they are part of the final EDRIM model or that each configuration reproduces a reported result.

## Data, reproducibility and licensing

Raw responses, downloaded third-party code, derived feature files, predictions, checkpoints, run logs and manuscript sources are excluded. Obtain external resources from their original providers and observe their terms. Historical paths and the outstanding reproduction requirements are described in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

See [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md) for the upstream references recorded by the project. A repository-wide software licence has not yet been assigned; do not assume that third-party adaptations or data are covered by MIT. The planned licence, release tag and data deposit must be resolved before claiming an archival code-and-data release in the manuscript.

## Interpretation

The measured-monotherapy reference defines an information setting and verifiable model behaviour; it is not a guarantee of lower prediction error. Results for combinations of known drugs do not establish generalisation to unseen drugs, clinical benefit or a pharmacological mechanism. The manuscript also reports settings in which the same-information baseline performs better.

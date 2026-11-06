# Source-snapshot preparation checks

Date: 2026-11-06.

- Python 3.12.7, PyTorch 2.12.0+cu130, pytest 9.0.3.
- Existing synthetic tests: **41 passed** in 1.61 seconds. The tests cover the shared drug exchange, compact and additive models, reference baselines, synthetic model behaviour and paired statistics.
- All copied Python files parsed successfully without executing experiment runners.
- A pattern scan of the 360 source/document files then present found no matches for private keys, common GitHub/Google/LLM/AWS credential formats or quoted credential assignments. This is a bounded pattern check, not a guarantee against every possible secret format.
- No raw data, derived features, weights, predictions, virtual environments, manuscript sources or private run records were included.
- No paper training or test evaluation was executed. Original experiment and manuscript files were not modified.

Reproduce the synthetic checks from the repository root after installing the package and its data/dev extras:

```bash
python -m pytest
```

Preparation used a pre-existing local environment and `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. A fresh-environment installation and end-to-end benchmark reproduction have not been validated.

## Publication state

The local Git snapshot is prepared. Remote repository creation and upload have not been completed: the command-line GitHub login was invalid, and the browser-origin approval was declined. No release tag, Zenodo deposit, DOI or repository-wide licence is claimed.

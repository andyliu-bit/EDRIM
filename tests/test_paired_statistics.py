import numpy as np
import pandas as pd
import pytest

from bisurfgnn.paired_statistics import PairedStatisticsError, paired_block_bootstrap


def _ledger() -> pd.DataFrame:
    rows = []
    # Unequal clusters: three correlated surfaces in cluster A, one in B.
    for surface in range(4):
        for seed in (41, 42):
            for model in ("hsa", "model"):
                rows.append({
                    "fold": 0, "seed": seed, "surface_id": f"s{surface}",
                    "model": model, "group_id": f"g{surface}",
                    "cluster_id": "A" if surface < 3 else "B",
                    "assay_block_id": f"plate{surface}", "n_points": 9,
                    "rmse": 1.0 if model == "hsa" else (0.8 if surface < 3 else 1.2),
                })
    return pd.DataFrame(rows)


def test_block_bootstrap_preserves_macro_estimator_and_is_reproducible() -> None:
    ledger = _ledger()
    result = paired_block_bootstrap(ledger, iterations=1000)
    assert result == paired_block_bootstrap(ledger.sample(frac=1, random_state=2), iterations=1000)
    assert result["observed"]["model"] == pytest.approx(0.9)  # Not equal-cluster 1.0.
    assert result["folds"] == [{"fold": 0, "blocks": 2, "surfaces": 4}]
    assert result["comparisons"]["model"]["difference_ci95"] == pytest.approx([-0.2, 0.2])
    assert result["seed_count"] == 2


def test_duplicate_surfaces_inside_a_cluster_do_not_create_independent_blocks() -> None:
    ledger = _ledger()
    copies = []
    for i in range(20):
        copy = ledger.copy()
        copy.surface_id += f"_copy{i}"
        copies.append(copy)
    duplicated = pd.concat(copies, ignore_index=True)
    small = paired_block_bootstrap(ledger, iterations=200)
    large = paired_block_bootstrap(duplicated, iterations=200)
    assert large["folds"][0]["blocks"] == 2
    np.testing.assert_allclose(small["comparisons"]["model"]["difference_ci95"],
                               large["comparisons"]["model"]["difference_ci95"])


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "metadata", "nonfinite", "seeds", "crossfold"])
def test_unpaired_or_ambiguous_ledgers_fail_closed(corruption) -> None:
    frame = _ledger()
    if corruption == "missing":
        frame = frame.iloc[1:].copy()
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif corruption == "metadata":
        frame.loc[0, "cluster_id"] = "other"
    elif corruption == "nonfinite":
        frame.loc[0, "rmse"] = np.nan
    elif corruption == "seeds":
        frame = frame[~((frame.model == "model") & (frame.seed == 42))]
    else:
        second = frame.copy()
        second.fold = 1
        frame = pd.concat([frame, second])
    with pytest.raises(PairedStatisticsError):
        paired_block_bootstrap(frame, iterations=20)


def test_zero_anchor_has_no_relative_interval() -> None:
    frame = _ledger()
    frame.loc[frame.model == "hsa", "rmse"] = 0
    result = paired_block_bootstrap(frame, unit="group_id", iterations=20)
    assert result["comparisons"]["model"]["relative_improvement"] is None
    assert result["comparisons"]["model"]["relative_ci95"] is None

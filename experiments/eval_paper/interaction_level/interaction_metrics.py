"""Shared interaction-level metrics for the eval-paper reanalysis (frozen card:
bisurfgnn/docs/EVAL_INTERACTION_LEVEL_20261029.md). Used unchanged by settings W, C and D.

Per matrix: S = mean over its combination cells of (R - y), positive = synergy direction;
R^HSA = min(a, b), R^Bliss = a * b / c (c = 100 percent growth, c = 1 viability fraction).
A model score replaces y by yhat with the same measured-mono reference, so S_hat - S = mean cell error.

Self-test:  .venv/Scripts/python.exe -I experiments/eval_paper/interaction_level/interaction_metrics.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

BOOT_SEED, BOOT_N, TOP_FRACTION = 20261029, 2000, 0.05


# ------------------------------------------------------------------------------------------------- scores
def reference(a, b, kind: str, c: float) -> np.ndarray:
    """Point-level reference from the measured monos a, b of each combination cell."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if kind == "HSA":
        return np.minimum(a, b)
    if kind == "Bliss":
        return a * b / c
    raise ValueError(kind)


def matrix_mean(values, codes) -> np.ndarray:
    """Mean of point values per matrix; codes are integer matrix ids 0..M-1 (every id present)."""
    codes = np.asarray(codes)
    return np.bincount(codes, weights=np.asarray(values, float)) / np.bincount(codes)


def matrix_score(ref, response, codes) -> np.ndarray:
    """S = per-matrix mean of (R - response); pass y for the observed score, yhat for a model score."""
    return matrix_mean(np.asarray(ref, float) - np.asarray(response, float), codes)


# ------------------------------------------------------------------------------------------------ metrics
def top_k(n: int, frac: float = TOP_FRACTION) -> int:
    return int(np.floor(frac * n + 0.5))      # round half up, as in SYNERGY_CHECK


def pearson(p, o) -> float:
    return float(np.corrcoef(p, o)[0, 1])     # numpy only: scipy's BLAS pearsonr clashed with torch's OpenMP here


def spearman(p, o) -> float:
    return float(np.corrcoef(rankdata(p), rankdata(o))[0, 1])


def precision_top(p, o, frac: float = TOP_FRACTION) -> float:
    """Share of the k highest observed scores that are also among the k highest predicted (stable ties)."""
    k = top_k(len(o), frac)
    tp = np.argsort(-np.asarray(p), kind="stable")[:k]
    to = np.argsort(-np.asarray(o), kind="stable")[:k]
    return float(len(np.intersect1d(tp, to)) / k)


def rmse(p, o) -> float:
    e = np.asarray(p, float) - np.asarray(o, float)
    return float(np.sqrt(np.mean(e * e)))


def slope(p, o) -> float:
    """cov(S_hat, S) / var(S)."""
    return float(np.cov(p, o)[0, 1] / np.var(o, ddof=1))


def all_metrics(p, o, degenerate: bool = False) -> dict:
    out = dict(rmse=rmse(p, o))
    if degenerate:
        out.update(pearson=np.nan, spearman=np.nan, precision_top5=np.nan, slope=np.nan)
    else:
        out.update(pearson=pearson(p, o), spearman=spearman(p, o), precision_top5=precision_top(p, o), slope=slope(p, o))
    return out


# -------------------------------------------------------------------------------------- cluster bootstrap
def cluster_draws(cluster_ids, n_boot: int = BOOT_N, seed: int = BOOT_SEED):
    """Yield row-index arrays: each draw resamples the clusters with replacement and keeps whole clusters."""
    _, codes = np.unique(np.asarray(cluster_ids), return_inverse=True)
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    g_n = len(sizes)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        g = rng.integers(0, g_n, g_n)
        sz = sizes[g]
        offs = np.arange(sz.sum()) - np.repeat(np.cumsum(sz) - sz, sz)
        yield order[np.repeat(starts[g], sz) + offs]


def bootstrap(stat_fn, cluster_ids, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """stat_fn(idx) -> dict of floats; returns dict of (n_boot,) arrays (same draws for every statistic)."""
    rows = [stat_fn(idx) for idx in cluster_draws(cluster_ids, n_boot, seed)]
    return {k: np.array([r[k] for r in rows], float) for k in rows[0]}


def percentile_ci(draws, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(draws, float), [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def boot_p(diff_draws) -> float:
    """Two-sided bootstrap p for H0: difference = 0, with the +1 correction (never exactly 0)."""
    d = np.asarray(diff_draws, float)
    tail = min(np.sum(d <= 0), np.sum(d >= 0)) + 1
    return float(min(1.0, 2 * tail / (len(d) + 1)))


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1), returned in input order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


# ----------------------------------------------------------------------------------------------- self-test
def _self_test() -> None:
    from scipy import stats
    # scores: one 2x2 matrix in percent; monos a=(80, 40), b=(50, 20); y = all 10
    a = np.array([80, 80, 40, 40.]); b = np.array([50, 20, 50, 20.]); y = np.full(4, 10.); codes = np.zeros(4, int)
    assert np.allclose(reference(a, b, "HSA", 100), [50, 20, 40, 20])
    assert np.allclose(reference(a, b, "Bliss", 100), [40, 16, 20, 8])
    assert np.isclose(matrix_score(reference(a, b, "HSA", 100), y, codes)[0], (130 / 4) - 10)
    assert np.isclose(matrix_score(reference(a / 100, b / 100, "Bliss", 1), y / 100, codes)[0] * 100,
                      matrix_score(reference(a, b, "Bliss", 100), y, codes)[0])
    # shared reference => identical HSA and Bliss score errors = matrix-mean cell error
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(50), 9); a, b = rng.uniform(0, 100, (2, 450)); y, yh = rng.normal(40, 20, (2, 450))
    errs = [matrix_score(reference(a, b, k, 100), yh, codes) - matrix_score(reference(a, b, k, 100), y, codes)
            for k in ("HSA", "Bliss")]
    assert np.allclose(errs[0], errs[1]) and np.allclose(errs[0], matrix_mean(y - yh, codes))
    # metrics against scipy / hand values
    p, o = rng.normal(size=200), rng.normal(size=200)
    p[:20] = 0.0                                                   # ties
    assert np.isclose(pearson(p, o), stats.pearsonr(p, o)[0])
    assert np.isclose(spearman(p, o), stats.spearmanr(p, o)[0])
    assert np.isclose(slope(o * 2 + 1, o), 2.0) and np.isclose(rmse([1, 3], [0, 0]), np.sqrt(5))
    assert top_k(54615) == 2731 and top_k(20) == 1 and top_k(30) == 2  # 1.5 rounds up
    assert precision_top(np.arange(100.), np.arange(100.)) == 1.0
    assert precision_top(-np.arange(100.), np.arange(100.)) == 0.0
    oo = np.arange(40.); pp = oo.copy(); pp[[39, 0]] = pp[[0, 39]]  # k = 2: top {39, 38} vs predicted {0, 38}
    assert precision_top(pp, oo) == 0.5
    m = all_metrics(p, o, degenerate=True)
    assert np.isnan(m["pearson"]) and np.isclose(m["rmse"], rmse(p, o))
    # cluster bootstrap: whole clusters, reproducible, plain bootstrap when every cluster is one row
    cl = np.repeat(["x", "y", "z", "w"], [3, 1, 2, 4])
    for idx in cluster_draws(cl, 50, 1):
        for c in np.unique(cl[idx]):
            n_c = np.sum(cl[idx] == c)
            assert n_c % np.sum(cl == c) == 0                      # each selected cluster appears whole
        assert np.isin(idx, np.arange(len(cl))).all()
    d1 = [i.tolist() for i in cluster_draws(cl, 5, 7)]; d2 = [i.tolist() for i in cluster_draws(cl, 5, 7)]
    assert d1 == d2
    single = np.arange(30)
    rng2 = np.random.default_rng(3); ref_idx = [np.sort(rng2.integers(0, 30, 30)) for _ in range(4)]
    got = [np.sort(i) for i in cluster_draws(single, 4, 3)]
    assert all(np.array_equal(r, g) for r, g in zip(ref_idx, got))
    res = bootstrap(lambda idx: dict(n=len(idx), mean=float(np.mean(idx))), cl, 20, 2)
    assert res["n"].shape == (20,) and (res["n"] > 0).all()
    # p-values and Holm
    assert boot_p(np.full(1999, 1.0)) == 2 / 2000 and boot_p(np.r_[-np.ones(50), np.ones(50)]) == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.02, 0.5]), [0.04, 0.5]) and np.allclose(holm([0.6, 0.7]), [1.0, 1.0])
    lo, hi = percentile_ci(np.arange(1001.))
    assert np.isclose(lo, 25) and np.isclose(hi, 975)
    print("interaction_metrics self-test: all asserts passed")


if __name__ == "__main__":
    _self_test()

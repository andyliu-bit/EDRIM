"""BiSurfGNN v2 dev protocol section 5 item 2: zero-training synergy-score check, VALIDATION split (split 2) only.

Pre-declared in runs/model_improvement/bisurfgnn_v2_dev/synergy_check/PREDECLARATION.md (hash-checked below against
PREDECLARATION_FREEZE.json, which was written before any score was computed).

Per evaluable validation matrix (gate set: 54,615 matrices, MDA-MB-468 excluded), averaged over the 9 combination cells:
  observed score   S^X  = mean_ij [ R^X(a, b)_ij - y_ij ]
  derived score    S^X^ = mean_ij [ R^X(a, b)_ij - yhat_ij ]          (observed monotherapy a, b for every model)
  DeepSynBa own    S^X~ = mean_ij [ R^X(ahat, bhat)_ij - yhat_ij ]    (DeepSynBa's predicted edges; labelled as such)
  R^HSA = min(a_i, b_j);  R^Bliss = a_i * b_j / 100;  a_i = truth[m, i, 0], b_j = truth[m, 0, j] (the matrix's own edges).
Positive = combination below the reference = synergy direction.

Reads only: the S1 validation point table (manifest hash), the gate's saved validation predictions (BiSurfGNN additive
seeds 41-45, XGBoost residual seeds 41-45; alignment checked as in V2-gap D2) and the DeepSynBa v2 epoch-484 validation
inference file written by V2-gap D1. Nothing under s5/, no marker, no split-3 file, no DeepSynBa test predictions, no
released pickle. Nothing is trained.

  .venv/Scripts/python.exe -I experiments/model_improvement/synergy_check/synergy_check.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GATE = ROOT / "runs/model_improvement/public_split_gate_v1"
S1 = GATE / "s1"
DSB_NPZ = ROOT / "runs/model_improvement/v2_gap_diagnosis/deepsynba_val/full_epoch484_selected_val_predictions.npz"
CHECK = ROOT / "runs/model_improvement/bisurfgnn_v2_dev/synergy_check"
PRE, PRE_FREEZE = CHECK / "PREDECLARATION.md", CHECK / "PREDECLARATION_FREEZE.json"
OUT = CHECK / "results"
PROTOCOL = ROOT / "docs/BISURFGNN_V2_DEV_PROTOCOL.md"
PROTOCOL_SHA256 = "155ae4f03ceb8275f68d10f9830f5bcd193c41f70564cc138bc175840788a333"
VAL_OFFSET = 10_000_000
BOOT_SEED, BOOT_N, BOOT_CHUNK = 20261013, 2000, 50
K_FRACTIONS = (0.05, 0.01, 0.10)          # 0.05 primary; 0.01 / 0.10 secondary, point estimates only
PRIMARY_K = 0.05
EXCLUDED_DRUG = "pralatrexate"
SCORES = ("HSA", "Bliss")
BISURF = {
    "BS_s41": GATE / "s3_additive_r2/bisurf_additive_seed_41",
    "BS_s42": GATE / "s4_bisurf_additive_a/bisurf_additive_seed_42",
    "BS_s43": GATE / "s4_bisurf_additive_b/bisurf_additive_seed_43",
    "BS_s44": GATE / "s4_bisurf_additive_a/bisurf_additive_seed_44",
    "BS_s45": GATE / "s4_bisurf_additive_b/bisurf_additive_seed_45",
}
XGB = {f"XGB_s{s}": GATE / f"s4_xgb_r2/xgb_residual_d10_m1_seed_{s}" for s in (41, 42, 43, 44, 45)}
BS_SEEDS, XGB_SEEDS = list(BISURF), list(XGB)
# Series whose derived score equals the observed reference up to fallback anchors: correlation / top-k undefined.
DEGENERATE = {("LK_HSA", "HSA"), ("LK_Bliss", "Bliss")}
# Pre-declared paired comparisons A - B; "BS" / "XGB" = mean of the 5 per-seed metrics (single-model convention).
COMPARISONS = {
    "C1 BS - DSB(observed ref)": ("BS", "DSB"),
    "C2 BS_ens5 - DSB(observed ref)": ("BS_ens5", "DSB"),
    "C3 BS - DSB(own mono)": ("BS", "DSB_own"),
    "C4 BS_ens5 - DSB(own mono)": ("BS_ens5", "DSB_own"),
    "C5 BS - XGB": ("BS", "XGB"),
    "C6 XGB - DSB(observed ref)": ("XGB", "DSB"),
}
BOOT_METRICS = ("pearson", "spearman", "rmse", "precision_top5")
FORBIDDEN_PARTS = ("s5",)
FORBIDDEN_NAMES = ("S5_TEST_EVALUATED.marker", "test_predictions.npz", "final_predictions.npz", "test.parquet",
                   "points_test.parquet", "nci_synergy_dose_response.pickle")


# ---------------------------------------------------------------------------------------------------------------
# access guard and hashing
# ---------------------------------------------------------------------------------------------------------------
READ_LOG: list[str] = []


def allowed(path: Path) -> Path:
    path = Path(path).resolve()
    if any(p in FORBIDDEN_PARTS for p in path.parts) or path.name in FORBIDDEN_NAMES or "split3" in path.name.lower():
        raise PermissionError(f"Refusing to open a test-side file: {path}")
    READ_LOG.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with allowed(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# score formulas
# ---------------------------------------------------------------------------------------------------------------
def references(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    """a: (M, 3) drug-A edges at doses 1..3; b: (M, 3) drug-B edges. Returns (M, 3, 3) references."""
    return {"HSA": np.minimum(a[:, :, None], b[:, None, :]), "Bliss": a[:, :, None] * b[:, None, :] / 100.0}


def matrix_scores(reference: dict[str, np.ndarray], combination: np.ndarray) -> dict[str, np.ndarray]:
    """Mean over the 9 combination cells of (reference - combination response); positive = synergy direction."""
    return {x: (reference[x] - combination).mean(axis=(1, 2)) for x in SCORES}


def k_of(n) -> np.ndarray:
    return np.floor(PRIMARY_K * np.asarray(n, float) + 0.5).astype(np.int64)


def k_frac(n: int, frac: float) -> int:
    return int(np.floor(frac * n + 0.5))


# ---------------------------------------------------------------------------------------------------------------
# metrics, unweighted (point estimates) and weighted by bootstrap multiplicities (rows of W)
# ---------------------------------------------------------------------------------------------------------------
def topk_set(x: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-x, kind="stable")[:k]


def precision_at_k(pred: np.ndarray, obs: np.ndarray, k: int) -> float:
    return float(len(np.intersect1d(topk_set(pred, k), topk_set(obs, k))) / k)


def point_metrics(pred: np.ndarray, obs: np.ndarray, degenerate: bool) -> dict:
    e = pred - obs
    out = dict(rmse=float(np.sqrt(np.mean(e * e))), bias=float(np.mean(e)), sd_pred=float(np.std(pred)),
               sd_obs=float(np.std(obs)))
    if degenerate:
        out.update(pearson=None, spearman=None, slope_pred_on_obs=None,
                   **{f"precision_top{round(f * 100)}": None for f in K_FRACTIONS})
        return out
    # numpy-only on purpose: scipy's BLAS-backed pearsonr initialises a second OpenMP runtime next to torch's
    # (both are loaded in this environment) and aborts with OMP error #15. rankdata is pure numpy.
    out["pearson"] = float(np.corrcoef(pred, obs)[0, 1])
    out["spearman"] = float(np.corrcoef(stats.rankdata(pred), stats.rankdata(obs))[0, 1])
    out["slope_pred_on_obs"] = float(np.cov(pred, obs)[0, 1] / np.var(obs, ddof=1))
    for f in K_FRACTIONS:
        out[f"precision_top{round(f * 100)}"] = precision_at_k(pred, obs, k_frac(len(obs), f))
    return out


def w_pearson_fixed(x: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted Pearson of fixed vectors x, y (N,) for every weight row of W (B, N)."""
    x = x - x.mean(); y = y - y.mean()
    sw = W.sum(1)
    mx, my = W @ x / sw, W @ y / sw
    sxy = W @ (x * y) - sw * mx * my
    sxx = W @ (x * x) - sw * mx * mx
    syy = W @ (y * y) - sw * my * my
    return sxy / np.sqrt(sxx * syy)


def w_pearson_rows(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted Pearson where X, Y are (B, N) (e.g. bootstrap-specific ranks)."""
    sw = W.sum(1, keepdims=True)
    mx = (W * X).sum(1, keepdims=True) / sw
    my = (W * Y).sum(1, keepdims=True) / sw
    dx, dy = X - mx, Y - my
    return (W * dx * dy).sum(1) / np.sqrt((W * dx * dx).sum(1) * (W * dy * dy).sum(1))


class RankPlan:
    """Sort structure of a fixed vector, reused to get expanded-sample average ranks for any weight matrix."""

    def __init__(self, x: np.ndarray):
        self.order = np.argsort(x, kind="stable")
        xs = x[self.order]
        new = np.r_[True, xs[1:] != xs[:-1]]
        self.starts = np.flatnonzero(new)
        self.group = np.cumsum(new) - 1

    def midranks(self, W: np.ndarray) -> np.ndarray:
        """Average ranks (1-based) each item would get in the sample where item i appears W[b, i] times."""
        ws = W[:, self.order]
        g = np.add.reduceat(ws, self.starts, axis=1)
        mid = (np.cumsum(g, axis=1) - g) + (g + 1.0) / 2.0
        out = np.empty_like(ws)
        out[:, self.order] = mid[:, self.group]
        return out


class TopPlan:
    """Descending stable order of a fixed vector; copies of item i included in the top k of the expanded sample."""

    def __init__(self, x: np.ndarray):
        self.order = np.argsort(-x, kind="stable")

    def included(self, W: np.ndarray, k: np.ndarray) -> np.ndarray:
        ws = W[:, self.order]
        before = np.cumsum(ws, axis=1) - ws
        inc = np.clip(k[:, None] - before, 0.0, ws)
        out = np.empty_like(inc)
        out[:, self.order] = inc
        return out


def w_precision(inc_pred: np.ndarray, inc_obs: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.minimum(inc_pred, inc_obs).sum(1) / k


def bootstrap_weights(pair_codes: np.ndarray, n_pairs: int) -> np.ndarray:
    """(BOOT_N, n_pairs) multinomial pair counts, seed 20261013 (same mechanism as V2-gap D2)."""
    rng = np.random.default_rng(BOOT_SEED)
    return rng.multinomial(n_pairs, np.full(n_pairs, 1.0 / n_pairs), size=BOOT_N).astype(float)


def run_bootstrap(series: dict, observed: dict, pair_codes: np.ndarray, n_pairs: int) -> dict:
    """Per-draw metrics for every (series, score); series[name][score] = derived matrix scores (N,)."""
    pair_w = bootstrap_weights(pair_codes, n_pairs)
    obs_rank = {x: RankPlan(observed[x]) for x in SCORES}
    obs_top = {x: TopPlan(observed[x]) for x in SCORES}
    plans = {(s, x): (RankPlan(v[x]), TopPlan(v[x])) for s, v in series.items() for x in SCORES
             if (s, x) not in DEGENERATE}
    res = {(s, x, m): np.empty(BOOT_N) for s in series for x in SCORES for m in BOOT_METRICS}
    for lo in range(0, BOOT_N, BOOT_CHUNK):
        W = pair_w[lo:lo + BOOT_CHUNK][:, pair_codes]               # (B, N) matrix multiplicities
        k = k_of(W.sum(1))
        sl = slice(lo, lo + len(W))
        for x in SCORES:
            r_obs = obs_rank[x].midranks(W)
            inc_obs = obs_top[x].included(W, k)
            for s, v in series.items():
                e = v[x] - observed[x]
                res[(s, x, "rmse")][sl] = np.sqrt(W @ (e * e) / W.sum(1))
                if (s, x) in DEGENERATE:
                    for m in ("pearson", "spearman", "precision_top5"):
                        res[(s, x, m)][sl] = np.nan
                    continue
                rank_plan, top_plan = plans[(s, x)]
                res[(s, x, "pearson")][sl] = w_pearson_fixed(v[x], observed[x], W)
                res[(s, x, "spearman")][sl] = w_pearson_rows(rank_plan.midranks(W), r_obs, W)
                res[(s, x, "precision_top5")][sl] = w_precision(top_plan.included(W, k), inc_obs, k)
    return res


def aggregate(res: dict, name: str, x: str, m: str) -> np.ndarray:
    if name == "BS":
        return np.mean([res[(s, x, m)] for s in BS_SEEDS], axis=0)
    if name == "XGB":
        return np.mean([res[(s, x, m)] for s in XGB_SEEDS], axis=0)
    return res[(name, x, m)]


def point_value(points: dict, name: str, x: str, m: str) -> float:
    if name == "BS":
        return float(np.mean([points[(s, x)][m] for s in BS_SEEDS]))
    if name == "XGB":
        return float(np.mean([points[(s, x)][m] for s in XGB_SEEDS]))
    return points[(name, x)][m]


# ---------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------
def check_predeclaration() -> dict:
    freeze = json.loads(allowed(PRE_FREEZE).read_text(encoding="utf-8"))
    digest = sha256(PRE)
    if digest != freeze["sha256"]:
        raise RuntimeError("PREDECLARATION.md changed after its freeze record")
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("Frozen development protocol changed")
    return dict(predeclaration_sha256=digest, predeclaration_frozen_local_time=freeze["frozen_local_time"],
                protocol_sha256=PROTOCOL_SHA256)


def load_matrices():
    manifest = json.loads(allowed(S1 / "manifest.json").read_text(encoding="utf-8"))
    if sha256(S1 / "points_validation.parquet") != manifest["points_validation.parquet"]:
        raise RuntimeError("S1 validation table changed")
    v = pd.read_parquet(allowed(S1 / "points_validation.parquet"))
    n_m = int(v.surface_index.max()) + 1
    if not (v.surface_index.to_numpy() == np.repeat(np.arange(n_m), 9)).all():
        raise RuntimeError("Validation points are not 9 consecutive rows per matrix")
    m_idx, i_idx, j_idx = v.surface_index.to_numpy(), v.row_i.to_numpy() - 1, v.col_j.to_numpy() - 1
    cell_check = np.zeros((n_m, 3, 3), int)
    np.add.at(cell_check, (m_idx, i_idx, j_idx), 1)
    if not (cell_check == 1).all():
        raise RuntimeError("Each validation matrix must have each combination cell exactly once")

    def grid(values) -> np.ndarray:
        out = np.full((n_m, 3, 3), np.nan)
        out[m_idx, i_idx, j_idx] = np.asarray(values, float)
        return out

    y = grid(v.response_percent)
    lk_a = grid(v.mono_a_percent)[:, :, 0]                   # constant over j by construction (checked below)
    lk_b = grid(v.mono_b_percent)[:, 0, :]
    if np.nanmax(np.abs(grid(v.mono_a_percent) - lk_a[:, :, None])) > 0 or \
            np.nanmax(np.abs(grid(v.mono_b_percent) - lk_b[:, None, :])) > 0:
        raise RuntimeError("Lookup anchors are not constant along the other drug's dose")
    first = v.drop_duplicates("surface_index").reset_index(drop=True)
    meta = first[["matrix_key", "pair_id", "deepsynba_cell", "deepsynba_drug_a", "deepsynba_drug_b"]].copy()
    checks = {}
    preds = {}
    for tag, path in {**BISURF, **XGB}.items():
        frame = pd.read_csv(allowed(path / "validation_predictions.csv"))
        metrics = json.loads(allowed(path / "metrics.json").read_text(encoding="utf-8"))
        ok_order = bool((frame.source_row.to_numpy() == VAL_OFFSET + np.arange(len(v))).all())
        ok_surface = bool((frame.surface_id.to_numpy() == v.matrix_key.to_numpy()).all())
        target_diff = float(np.max(np.abs(frame.target.to_numpy(float) * 100.0 - v.response_percent.to_numpy(float))))
        p = frame.prediction.to_numpy(float) * 100.0
        r = float(np.sqrt(np.mean((p - v.response_percent.to_numpy(float)) ** 2)))
        logged = metrics["validation_pooled_rmse_percent"]
        if not (ok_order and ok_surface and target_diff < 1e-4 and abs(r - logged) < 1e-6):
            raise RuntimeError(f"{tag}: alignment failure {ok_order} {ok_surface} {target_diff} {r} {logged}")
        preds[tag] = grid(p)
        checks[tag] = dict(order_ok=ok_order, surface_ok=ok_surface, target_max_abs_diff=target_diff,
                           combination_rmse_recomputed=r, combination_rmse_logged=logged)
    z = np.load(allowed(DSB_NPZ), allow_pickle=False)
    keys = pd.Index(z["matrix_key"])
    if not keys.is_unique:
        raise RuntimeError("DeepSynBa validation keys are not unique")
    pos = keys.get_indexer(meta.matrix_key)
    if (pos < 0).any():
        raise RuntimeError(f"{int((pos < 0).sum())} gate validation matrices have no DeepSynBa prediction")
    truth = z["truth"].astype(float)[pos]
    dsb = z["predictions"].astype(float)[pos]
    truth_diff = float(np.max(np.abs(truth[:, 1:, 1:] - y)))
    if truth_diff > 1e-3:
        raise RuntimeError(f"DeepSynBa truth differs from the gate table on combination cells ({truth_diff})")
    obs_a, obs_b = truth[:, 1:, 0], truth[:, 0, 1:]
    lk_diff = np.abs(np.concatenate([(lk_a - obs_a).ravel(), (lk_b - obs_b).ravel()]))
    checks["DSB_e484"] = dict(file_sha256=sha256(DSB_NPZ), matrices_in_file=int(len(keys)), gate_matrices_matched=int(len(pos)),
                              truth_vs_gate_combination_max_abs_diff=truth_diff)
    checks["observed_edges"] = dict(
        source="truth[:, i, 0] and truth[:, 0, j] of each validation matrix (DeepSynBa validation cache)",
        edges=int(lk_diff.size), share_lookup_equals_observed_1e4=float(np.mean(lk_diff < 1e-4)),
        max_abs_lookup_minus_observed=float(lk_diff.max()),
        observed_control_min=float(truth[:, 0, 0].min()), observed_control_max=float(truth[:, 0, 0].max()))
    checks["DSB_own_edges"] = dict(
        edge_rmse_vs_observed=float(np.sqrt(np.mean(np.concatenate([(dsb[:, 1:, 0] - obs_a).ravel(),
                                                                    (dsb[:, 0, 1:] - obs_b).ravel()]) ** 2))),
        predicted_control_rmse_vs_100=float(np.sqrt(np.mean((dsb[:, 0, 0] - 100.0) ** 2))),
        predicted_control_mean=float(dsb[:, 0, 0].mean()))
    preds["DSB"] = dsb[:, 1:, 1:]
    preds["LK_HSA"] = np.minimum(lk_a[:, :, None], lk_b[:, None, :])
    preds["LK_Bliss"] = lk_a[:, :, None] * lk_b[:, None, :] / 100.0
    preds["BS_ens5"] = np.mean([preds[k] for k in BS_SEEDS], axis=0)
    preds["XGB_ens5"] = np.mean([preds[k] for k in XGB_SEEDS], axis=0)
    return meta, y, obs_a, obs_b, lk_a, lk_b, dsb, preds, checks


# ---------------------------------------------------------------------------------------------------------------
# analysis of one matrix subset
# ---------------------------------------------------------------------------------------------------------------
def analyse(mask: np.ndarray, label: str, meta, observed, derived, cell_rmse) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    obs = {x: observed[x][mask] for x in SCORES}
    series = {s: {x: v[x][mask] for x in SCORES} for s, v in derived.items()}
    points, rows = {}, []
    for s, v in series.items():
        for x in SCORES:
            pm = point_metrics(v[x], obs[x], (s, x) in DEGENERATE)
            points[(s, x)] = pm
            rows.append(dict(subset=label, series=s, score=x, matrices=int(mask.sum()),
                             cell_level_excess_rmse=cell_rmse[s][x](mask), **pm))
    point_table = pd.DataFrame(rows)
    pair_codes, pair_uniques = pd.factorize(meta.pair_id[mask])
    t0 = time.perf_counter()
    res = run_bootstrap(series, obs, pair_codes, len(pair_uniques))
    boot_seconds = time.perf_counter() - t0
    # cross-check the weighted implementation against the point estimates (unit weights)
    unit = np.ones((1, int(mask.sum())))
    k1 = k_of(unit.sum(1))
    for s in ("DSB", "BS_s41", "BS_ens5"):
        for x in SCORES:
            v = series[s][x]
            checks_ = dict(pearson=w_pearson_fixed(v, obs[x], unit)[0],
                           spearman=w_pearson_rows(RankPlan(v).midranks(unit), RankPlan(obs[x]).midranks(unit), unit)[0],
                           precision_top5=w_precision(TopPlan(v).included(unit, k1), TopPlan(obs[x]).included(unit, k1), k1)[0])
            for m, val in checks_.items():
                if abs(val - points[(s, x)][m]) > 1e-9:
                    raise RuntimeError(f"weighted metric {m} disagrees with the point estimate for {s}/{x}")
    comp_rows = []
    for cname, (a, b) in COMPARISONS.items():
        for x in SCORES:
            for m in BOOT_METRICS:
                da, db = aggregate(res, a, x, m), aggregate(res, b, x, m)
                diff = da - db
                pa, pb = point_value(points, a, x, m), point_value(points, b, x, m)
                lo, hi = np.percentile(diff, [2.5, 97.5])
                better = "higher" if m != "rmse" else "lower"
                comp_rows.append(dict(subset=label, comparison=cname, a=a, b=b, score=x, metric=m, a_value=pa, b_value=pb,
                                      difference=pa - pb, ci_low=float(lo), ci_high=float(hi), better_is=better,
                                      boot_sd=float(np.std(diff, ddof=1)), pairs=int(len(pair_uniques))))
    comp_table = pd.DataFrame(comp_rows)
    seed_rows = []
    for ref in ("DSB", "DSB_own"):
        for x in SCORES:
            for m in ("pearson", "spearman", "rmse", "precision_top5"):
                dv = points[(ref, x)][m]
                vals = [points[(s, x)][m] for s in BS_SEEDS]
                better = [(val < dv) if m == "rmse" else (val > dv) for val in vals]
                seed_rows.append(dict(subset=label, reference=ref, score=x, metric=m, reference_value=dv,
                                      **{s: val for s, val in zip(BS_SEEDS, vals)},
                                      bisurf_seeds_better=int(sum(better)), bisurf_seed_sd=float(np.std(vals, ddof=1))))
    info = dict(matrices=int(mask.sum()), pairs=int(len(pair_uniques)), bootstrap_seconds=round(boot_seconds, 1),
                k_primary=int(k_of(mask.sum())),
                observed_score_summary={x: dict(mean=float(obs[x].mean()), sd=float(obs[x].std()),
                                                q01=float(np.quantile(obs[x], 0.01)), q50=float(np.median(obs[x])),
                                                q95=float(np.quantile(obs[x], 0.95)), q99=float(np.quantile(obs[x], 0.99)),
                                                top5_threshold=float(np.sort(obs[x])[::-1][int(k_of(mask.sum())) - 1]))
                                        for x in SCORES},
                observed_hsa_bliss_score_corr=float(np.corrcoef(obs["HSA"], obs["Bliss"])[0, 1]))
    return point_table, comp_table, pd.DataFrame(seed_rows), info


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT, help="fresh output directory (default: synergy_check/results)")
    out = parser.parse_args().out
    t_start = time.perf_counter()
    pre = check_predeclaration()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)       # an empty directory left by a crashed attempt is reused
    meta, y, obs_a, obs_b, lk_a, lk_b, dsb_full, preds, checks = load_matrices()
    ref_obs = references(obs_a, obs_b)
    observed = matrix_scores(ref_obs, y)
    derived = {s: matrix_scores(ref_obs, p) for s, p in preds.items()}
    ref_own = references(dsb_full[:, 1:, 0], dsb_full[:, 0, 1:])
    derived["DSB_own"] = matrix_scores(ref_own, preds["DSB"])
    # invariance under the shared observed reference: identical HSA / Bliss score errors
    for s in preds:
        d = np.max(np.abs((derived[s]["HSA"] - observed["HSA"]) - (derived[s]["Bliss"] - observed["Bliss"])))
        if d > 1e-9:
            raise RuntimeError(f"{s}: HSA and Bliss score errors differ under the shared reference ({d})")
        e = (y - preds[s]).mean(axis=(1, 2))
        if np.max(np.abs((derived[s]["HSA"] - observed["HSA"]) - e)) > 1e-9:
            raise RuntimeError(f"{s}: score error is not the matrix-mean combination error")
    # anchor models' own matrix edges are the lookup anchors: how far that reference is from the observed one
    ref_lk = references(lk_a, lk_b)
    checks["anchor_models_lookup_vs_observed_reference"] = {
        x: dict(max_abs_score_diff=float(np.max(np.abs(matrix_scores(ref_lk, preds["BS_ens5"])[x] - derived["BS_ens5"][x]))),
                matrices_with_diff_above_1e4=int(np.sum(np.abs(matrix_scores(ref_lk, preds["BS_ens5"])[x]
                                                                - derived["BS_ens5"][x]) > 1e-4)))
        for x in SCORES}
    cell_rmse = {}
    for s, p in preds.items():
        cell_rmse[s] = {x: (lambda mask, p=p: float(np.sqrt(np.mean((p[mask] - y[mask]) ** 2)))) for x in SCORES}
    cell_rmse["DSB_own"] = {x: (lambda mask, x=x: float(np.sqrt(np.mean(
        (((ref_own[x] - preds["DSB"]) - (ref_obs[x] - y))[mask]) ** 2)))) for x in SCORES}
    has_drug = ((meta.deepsynba_drug_a == EXCLUDED_DRUG) | (meta.deepsynba_drug_b == EXCLUDED_DRUG)).to_numpy()
    subsets = {"all": np.ones(len(meta), bool), f"excluding_{EXCLUDED_DRUG}": ~has_drug}
    tables = {"points": [], "comparisons": [], "seeds": []}
    infos = {}
    for label, mask in subsets.items():
        p, c, s, info = analyse(mask, label, meta, observed, derived, cell_rmse)
        tables["points"].append(p); tables["comparisons"].append(c); tables["seeds"].append(s)
        infos[label] = info
        print(f"[{label}] {info['matrices']} matrices, {info['pairs']} pairs, bootstrap {info['bootstrap_seconds']} s", flush=True)
    points = pd.concat(tables["points"], ignore_index=True)
    comps = pd.concat(tables["comparisons"], ignore_index=True)
    seeds = pd.concat(tables["seeds"], ignore_index=True)
    points.to_csv(out / "point_estimates.csv", index=False)
    comps.to_csv(out / "paired_comparisons.csv", index=False)
    seeds.to_csv(out / "seed_level.csv", index=False)
    score_frame = meta.copy()
    score_frame["contains_" + EXCLUDED_DRUG] = has_drug
    for x in SCORES:
        score_frame[f"observed_{x}"] = observed[x]
        for s, v in derived.items():
            score_frame[f"{s}_{x}"] = v[x]
    score_frame.to_parquet(out / "matrix_scores.parquet", index=False)
    summary = dict(
        step="BiSurfGNN-v2-dev synergy-score check (protocol section 5 item 2)", split_read=2, test_rows_read=0,
        nature="validation-set development check, not a final result; split 2 also used for epoch selection",
        **pre, checks=checks, subsets=infos, pralatrexate_pairs=int(meta.pair_id[has_drug].nunique()),
        pralatrexate_matrices=int(has_drug.sum()), files_read=sorted(set(READ_LOG)),
        bootstrap=dict(seed=BOOT_SEED, draws=BOOT_N, unit="validation drug pair", k_primary_fraction=PRIMARY_K),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scipy=__import__("scipy").__version__),
        runtime_seconds=round(time.perf_counter() - t_start, 1), finished_local_time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=1, ensure_ascii=False, default=float) + "\n")
    show = points[points.series.isin(["DSB", "DSB_own", "BS_ens5", "XGB_ens5", "LK_HSA", "LK_Bliss"]) & (points.subset == "all")]
    print(show[["series", "score", "pearson", "spearman", "rmse", "precision_top5", "slope_pred_on_obs"]].to_string(index=False))
    print(comps[comps.subset == "all"][["comparison", "score", "metric", "a_value", "b_value", "difference",
                                         "ci_low", "ci_high"]].to_string(index=False))
    print(f"done in {summary['runtime_seconds']} s")


if __name__ == "__main__":
    main()

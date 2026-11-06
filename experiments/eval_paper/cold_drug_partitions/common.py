"""Eval-paper supplement G: the D11 new-drug split on 5 drug partitions.

Card and results: C:/LHZ/research/bisurfgnn/docs/EVAL_COLD_DRUG_PARTITIONS_20261018.md
Shared pieces: split rule (copied from experiments/model_improvement/cold_drug/cold_split.py; prepare.py checks that the
copy regenerates the saved D11 split exactly), a light S1 reader (split 1 + split 2 only), the D3 P4 probe, the
drug-pair bootstrap of score_cold.py and Holm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/model_improvement/v2_gap_diagnosis"))
import d3_probes as d3  # noqa: E402  (design / fit_probe, unchanged)

S1 = ROOT / "runs/model_improvement/public_split_gate_v1/s1"
D11_DIR = ROOT / "runs/model_improvement/bisurfgnn_v2_dev/cold_drug"
OUT = ROOT / "runs/eval_paper/cold_drug_partitions"
D11_SEED = 20261016
NEW_SEEDS = (20261018, 20261019, 20261020, 20261021)
PARTS = (D11_SEED,) + NEW_SEEDS
N_TEST, N_VAL = 15, 10
ROLES = ("train", "cold_val", "cold_test")
ROLE_OFFSET = {"train": 0, "validation": 10_000_000}   # gate_common.ROLE_OFFSET (source_combodrugseq)
XGB_SEEDS = (41, 42, 43)
PRALA = "S754230"
BLOCKS, LAM = ["position", "drug_dose", "drug_cell", "cell_position"], 100.0
DRAWS, BOOT_SEED = 2000, 20261009
COLS = ["matrix_key", "pair_id", "deepsynba_cell", "drug_a_id", "drug_b_id", "deepsynba_drug_a", "deepsynba_drug_b",
        "row_i", "col_j", "mono_a_percent", "mono_b_percent", "response_percent", "hsa_lookup_percent"]


def split_path(seed: int) -> Path:
    return D11_DIR / "split.json" if seed == D11_SEED else OUT / f"p{seed}/split.json"


def assign(frame: pd.DataFrame, test_drugs: set, val_drugs: set) -> np.ndarray:
    """cold_split.assign, unchanged."""
    a, b = frame.drug_a_id.to_numpy(), frame.drug_b_id.to_numpy()
    in_test = np.isin(a, list(test_drugs)) | np.isin(b, list(test_drugs))
    in_val = np.isin(a, list(val_drugs)) | np.isin(b, list(val_drugs))
    return np.where(in_test, "cold_test", np.where(in_val, "cold_val", "train"))


def read_points(columns=COLS) -> pd.DataFrame:
    """S1 points_train then points_validation (cold_split.load_frames order), with gate source ids. Split 3 never read."""
    parts = []
    for role in ("train", "validation"):
        f = pd.read_parquet(S1 / f"points_{role}.parquet", columns=columns)
        f["source_combodrugseq"] = ROLE_OFFSET[role] + np.arange(len(f), dtype=np.int64)
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def make_split(seed: int, both: pd.DataFrame) -> dict:
    """cold_split.main without the file write: the 101 split-1 drugs, permuted, first 15 test, next 10 validation."""
    tr = both[both.source_combodrugseq < ROLE_OFFSET["validation"]]
    drugs = np.array(sorted(set(tr.drug_a_id) | set(tr.drug_b_id)))
    perm = np.random.default_rng(seed).permutation(len(drugs))
    test, val = sorted(drugs[perm[:N_TEST]]), sorted(drugs[perm[N_TEST:N_TEST + N_VAL]])
    role = assign(both, set(test), set(val))
    first = both.drop_duplicates("matrix_key")
    first_role = assign(first, set(test), set(val))
    counts = {r: dict(points=int((role == r).sum()), matrices=int((first_role == r).sum()),
                      pairs=int(first.pair_id[first_role == r].nunique())) for r in ROLES}
    seen = set(both.drug_a_id[role == "train"]) | set(both.drug_b_id[role == "train"])
    if seen & (set(test) | set(val)):
        raise RuntimeError("Held-out drug in training")
    names = both.drop_duplicates("drug_a_id").set_index("drug_a_id").deepsynba_drug_a.to_dict()
    names.update(both.drop_duplicates("drug_b_id").set_index("drug_b_id").deepsynba_drug_b.to_dict())
    return dict(seed=seed, n_drugs=len(drugs), test_drugs=[str(d) for d in test], val_drugs=[str(d) for d in val],
                test_drug_names=[names[d] for d in test], val_drug_names=[names[d] for d in val], counts=counts,
                source="S1 points_train + points_validation; split 3 not read",
                registered="bisurfgnn/docs/EVAL_COLD_DRUG_PARTITIONS_20261018.md (rule of cold_split.py, D11)")


def frames(seed: int, both: pd.DataFrame) -> dict[str, pd.DataFrame]:
    split = json.loads(split_path(seed).read_text(encoding="utf-8"))
    role = assign(both, set(split["test_drugs"]), set(split["val_drugs"]))
    out = {r: both[role == r].reset_index(drop=True) for r in ROLES}
    held = set(split["test_drugs"]) | set(split["val_drugs"])
    assert not (set(out["train"].drug_a_id) | set(out["train"].drug_b_id)) & held, "held-out drug in training"
    return out


def lookup_hsa(frame: pd.DataFrame) -> np.ndarray:
    return np.minimum(frame.mono_a_percent, frame.mono_b_percent).to_numpy(float)


def fit_p4(train: pd.DataFrame, all_drugs: list) -> dict:
    """D3 P4 on y - lookup HSA, lambda 100. The design carries columns for every drug; held-out drugs have no training
    rows, so the ridge leaves their offsets at exactly 0 (checked)."""
    cells = sorted(set(train.deepsynba_cell))
    x = d3.design(train, cells, all_drugs, BLOCKS)
    coef, mean, iters = d3.fit_probe(x, train.response_percent.to_numpy(float) - lookup_hsa(train), LAM)
    used = np.zeros(x.shape[1], bool); used[x.indices] = True
    assert (coef[~used] == 0).all(), "unseen columns got a nonzero offset"
    seen = set(train.drug_a_id) | set(train.drug_b_id)
    new = [k for k, d in enumerate(all_drugs) if d not in seen]
    assert new and not used[[9 + 3 * k + i for k in new for i in range(3)]].any(), "held-out drug has training rows"
    return dict(coef=coef, mean=mean, iters=iters, cells=cells, drugs=all_drugs, coefficients=int(x.shape[1]),
                unseen_columns=int((~used).sum()))


def predict_p4(model: dict, frame: pd.DataFrame) -> np.ndarray:
    if not set(frame.deepsynba_cell) <= set(model["cells"]):
        raise RuntimeError("cell missing from training")
    return lookup_hsa(frame) + model["mean"] + d3.design(frame, model["cells"], model["drugs"], BLOCKS) @ model["coef"]


def rmse(e) -> float:
    return float(np.sqrt(np.mean(np.square(e))))


def bootstrap(frame: pd.DataFrame, preds: dict[str, dict[int, np.ndarray]]) -> dict:
    """score_cold.py scheme: one set of drug-pair draws (seed 20261009); multi-seed methods = mean of per-seed RMSE;
    the pralatrexate set masks the same draws. Returns point / per-seed RMSE and draws per method for both sets."""
    y = frame.response_percent.to_numpy(float)
    pairs, codes = np.unique(frame.pair_id.to_numpy(), return_inverse=True)
    n = np.bincount(codes).astype(float)
    sse = {(m, s): np.bincount(codes, weights=(p - y) ** 2) for m, d in preds.items() for s, p in d.items()}
    rng = np.random.default_rng(BOOT_SEED)
    W = np.stack([np.bincount(rng.integers(0, len(pairs), len(pairs)), minlength=len(pairs)) for _ in range(DRAWS)]).astype(float)
    prala = np.array([PRALA in p.split("|") for p in pairs])
    res = {}
    for label, keep in (("all", np.ones(len(pairs), bool)), ("excluding_pralatrexate", ~prala)):
        w = W * keep
        seeds = {m: {int(s): float(np.sqrt(sse[(m, s)][keep].sum() / n[keep].sum())) for s in d} for m, d in preds.items()}
        res[label] = dict(pairs=int(keep.sum()), points=int(n[keep].sum()), per_seed=seeds,
                          point={m: float(np.mean(list(v.values()))) for m, v in seeds.items()},
                          boot={m: np.mean([np.sqrt((w @ sse[(m, s)]) / (w @ n)) for s in d], axis=0) for m, d in preds.items()})
    return res


def compare(block: dict, a: str, b: str) -> dict:
    """Relative change a vs b (negative = a better), 95% percentile CI, two-sided bootstrap p (card formula)."""
    r = (block["boot"][a] - block["boot"][b]) / block["boot"][b]
    p = min(1.0, 2 * min(((r >= 0).sum() + 1) / (DRAWS + 1), ((r <= 0).sum() + 1) / (DRAWS + 1)))
    return dict(relative_change=(block["point"][a] - block["point"][b]) / block["point"][b],
                ci95=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))], p=float(p))


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    adj, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        adj[k] = run
    return adj


assert holm({"a": 0.125, "b": 0.5, "c": 0.25}) == {"a": 0.375, "c": 0.5, "b": 0.5}

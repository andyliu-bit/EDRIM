"""D21 check (2) of docs/COMBOFM_CARD_20261027.md: the PyTorch TFFM port on comboFM's own data and folds.

Mirrors comboFM__example.py at commit daa2e2b8: scenario new_drug_combinations, outer fold 1, fixed parameters
(order 5, rank 50, reg 1e4, Adam lr 1e-3, batch 1024, init_std 0.01, 50 epochs, seed 123), both drug orders,
auxiliary columns z-scored on the training rows. The paper reports Pearson 0.95 for this scenario (all outer folds,
tuned parameters, 200 epochs), so this is an approximate check; the gap is recorded, nothing is tuned to close it.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/combofm/repro_combofm_fold.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tffm_torch  # noqa: E402

ROOT = HERE.parents[2]
DATA = ROOT / "data/external/combofm_zenodo/extracted/comboFM_data/data"
FOLDS = ROOT / "data/external/combofm_repo/cross-validation_folds/new_drug_combinations"
OUT = ROOT / "runs/model_improvement/combofm/repro_fold1"
TENSOR = ("drug{a}_concentration__one-hot_encoding", "drug{b}_concentration__one-hot_encoding",
          "drug{a}__one-hot_encoding", "drug{b}__one-hot_encoding", "cell_lines__one-hot_encoding")
AUX = ("drug{a}_drug{b}_concentration__values", "drug{a}__estate_fingerprints",
       "drug{b}__estate_fingerprints", "cell_lines__gene_expression")


def features(a, b):
    load = lambda names: np.concatenate(
        [pd.read_csv(DATA / f"{n.format(a=a, b=b)}.csv").to_numpy(np.float32) for n in names], axis=1)
    return load(TENSOR), load(AUX)


def main():
    start = time.monotonic()
    t1, a1 = features(1, 2)
    t2, a2 = features(2, 1)
    i_aux = t1.shape[1]
    x = np.concatenate([np.concatenate([t1, a1], 1), np.concatenate([t2, a2], 1)], 0)
    y = pd.read_csv(DATA / "responses.csv").to_numpy(np.float32).ravel()
    y = np.concatenate([y, y])
    te = np.loadtxt(FOLDS / "test_idx_outer_fold-1.txt").astype(int)
    tr = np.loadtxt(FOLDS / "train_idx_outer_fold-1.txt").astype(int)
    assert not set(te) & set(tr)
    mu = x[tr, i_aux:].mean(0)
    sd = x[tr, i_aux:].std(0)  # StandardScaler: population SD; constant columns are left unscaled
    sd[sd == 0] = 1
    x[:, i_aux:] = (x[:, i_aux:] - mu) / sd
    dev = torch.device("cuda")
    xtr, ytr = torch.from_numpy(x[tr]).to(dev), torch.from_numpy(y[tr]).to(dev)
    xte = torch.from_numpy(x[te]).to(dev)
    print(f"features={x.shape[1]} (one-hot {i_aux}) train_rows={len(tr)} test_rows={len(te)}", flush=True)

    torch.manual_seed(123)
    model = tffm_torch.TFFM(x.shape[1], order=5, rank=50, init_std=0.01, seed=123).to(dev)
    history = []

    def on_epoch(epoch):
        if epoch % 10 == 0 or epoch == 1:
            pred = tffm_torch.predict(model, lambda i: ((xte[i],), None), len(te))
            rmse = float(np.sqrt(np.mean((y[te] - pred) ** 2)))
            history.append({"epoch": epoch, "test_rmse": rmse, "test_pearson": float(np.corrcoef(y[te], pred)[0, 1])})
            print(json.dumps(history[-1]), f"({time.monotonic() - start:.0f}s)", flush=True)

    tffm_torch.fit(model, lambda i: ((xtr[i],), ytr[i]), len(tr), n_epochs=50, batch_size=1024, lr=1e-3, reg=1e4,
                   seed=123, on_epoch=on_epoch)
    pred = tffm_torch.predict(model, lambda i: ((xte[i],), None), len(te))
    result = {"scenario": "new_drug_combinations", "outer_fold": 1, "params": "comboFM__example.py fixed",
              "test_rmse": float(np.sqrt(np.mean((y[te] - pred) ** 2))),
              "test_pearson": float(np.corrcoef(y[te], pred)[0, 1]),
              "test_spearman": float(spearmanr(y[te], pred)[0]),
              "paper_reference": "Pearson 0.95 (all outer folds, tuned, 200 epochs; Julkunen et al. 2020)",
              "history": history, "wall_seconds": time.monotonic() - start}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}))


if __name__ == "__main__":
    main()

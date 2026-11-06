"""D21 (docs/COMBOFM_CARD_20261027.md): comboFM on the NCI public split, validation only.

One run = one (reg, rank, seed) with order 5, Adam lr 1e-3, batch 1024, init_std 0.01 (comboFM's fixed values),
trained on split 1 and scored on split 2 every 10 epochs; the epoch with the lowest validation surface-macro RMSE
is kept. Split 3 is never read (gate.load_points refuses it).

Rows, as comboFM's data: every split-1 combination cell, plus the 6 monotherapy edge cells (one drug at its dose,
the other at 0) of every split-1 and split-2 matrix, valued from the split-1 lookup that Ours uses as its HSA
reference (mono_a_percent / mono_b_percent), so comboFM gets the same monotherapy information as Ours. Split-2
combination cells are never trained on. Both drug orders are included; validation predictions average the two
orders. Target: percent growth, as comboFM (metrics are reported on the ratio scale like the other methods).

Features, as comboFM__example.py: one-hot of drug-1 concentration, drug-2 concentration, drug 1, drug 2, cell line;
auxiliary: the two concentrations in M, EState fingerprint bits of each drug (RDKit, zero-variance bits dropped
over the drugs present, as comboFM's R preprocessing), and the 976-gene expression used by Ours (card L). Auxiliary
columns are z-scored with the mean and population SD over the training rows (StandardScaler).

  .venv/Scripts/python.exe -I -B experiments/model_improvement/combofm/run_combofm.py --reg 1e4 --rank 50 --seed 41 --epochs 100
  .venv/Scripts/python.exe -I -B experiments/model_improvement/combofm/run_combofm.py --reg 1e4 --rank 50 --seed 41 --epochs 2 --smoke --out-root <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/model_improvement/public_split_gate"))
sys.path.insert(0, str(ROOT / "experiments/model_improvement/input_parity"))
sys.path.insert(0, str(ROOT / "src"))
import tffm_torch  # noqa: E402
import gate_common as gate  # noqa: E402

OUT = ROOT / "runs/model_improvement/combofm/nci"
DRUGS_CSV = gate.DATA / "data/processed/nci_almanac_unclipped/drugs.csv"
EVAL_EVERY = 10


def dose_key(values):
    return np.char.mod("%.6g", np.asarray(values, float))


def estate_bits(drug_ids):
    from rdkit import Chem
    from rdkit.Chem.EState.Fingerprinter import FingerprintMol
    smiles = pd.read_csv(DRUGS_CSV, keep_default_na=False).set_index("drug_id").canonical_smiles
    bits = np.stack([(FingerprintMol(Chem.MolFromSmiles(smiles[d]))[0] > 0).astype(np.float32) for d in drug_ids])
    return bits[:, bits.var(0) > 0]


def gene_table(frames, cell_ids):
    import run_parity as rp
    from bisurfgnn.data.external_features import load_deepsynba_features
    drug_ids = sorted(set(pd.concat(frames.values()).drug_a_id) | set(pd.concat(frames.values()).drug_b_id))
    ext = load_deepsynba_features(drug_ids, cell_ids, frames, feature_dir=rp.guard(rp.FEATURE_DIR),
                                  mapping_dir=rp.guard(gate.S1))
    assert tuple(ext.cell_ids) == tuple(cell_ids)
    return ext.cell.numpy().astype(np.float32)


def build_rows(frames):
    """Long table of (drug1, drug2, dose1_M, dose2_M, cell, target_percent, kind) in one drug order."""
    tr, va = frames["train"], frames["validation"]
    combo = tr.assign(d1=tr.drug_a_id, d2=tr.drug_b_id, c1=tr.dose_a_M, c2=tr.dose_b_M, y=tr.response_percent)
    mono = []
    for f in (tr, va):
        a = f.drop_duplicates(["matrix_key", "row_i"])
        b = f.drop_duplicates(["matrix_key", "col_j"])
        mono.append(a.assign(d1=a.drug_a_id, d2=a.drug_b_id, c1=a.dose_a_M, c2=0.0, y=a.mono_a_percent))
        mono.append(b.assign(d1=b.drug_a_id, d2=b.drug_b_id, c1=0.0, c2=b.dose_b_M, y=b.mono_b_percent))
    cols = ["d1", "d2", "c1", "c2", "biological_cell_id", "cell_id", "y"]
    return pd.concat([combo[cols]] + [m[cols] for m in mono], ignore_index=True)


class Encoder:
    """Builds comboFM feature rows on the GPU from integer codes; auxiliary blocks are z-scored on training rows."""

    def __init__(self, train_rows, all_rows, drug_ids, cell_ids, estate, genes, device):
        self.device = device
        self.drug_index = {d: i for i, d in enumerate(drug_ids)}
        self.cell_index = {c: i for i, c in enumerate(cell_ids)}
        self.conc_keys = sorted(set(dose_key(all_rows.c1)) | set(dose_key(all_rows.c2)), key=float)
        self.conc_index = {k: i for i, k in enumerate(self.conc_keys)}
        nc, nd, nk = len(self.conc_keys), len(drug_ids), len(cell_ids)
        self.offsets = np.cumsum([0, nc, nc, nd, nd])  # conc1, conc2, drug1, drug2, cell
        self.n_onehot = 2 * nc + 2 * nd + nk
        self.offsets_t = torch.as_tensor(self.offsets, device=device)
        codes = self.codes(train_rows, swap=False)
        codes = np.concatenate([codes, self.codes(train_rows, swap=True)])
        # z-score statistics over training rows (both orders), as StandardScaler on the materialised matrix
        conc = codes[:, 5:7].astype(np.float64)
        ix = codes[:, :5].astype(np.int64)
        self.conc_mu, self.conc_sd = conc.mean(0), conc.std(0)

        def stats(table, index):  # mean / population SD over rows of table[index], from per-entity counts
            w = np.bincount(index, minlength=len(table)) / len(index)
            mu = w @ table
            sd = np.sqrt(w @ (table - mu) ** 2)
            return mu, np.where(sd > 0, sd, 1.0)
        (m1, s1), (m2, s2), (mg, sg) = stats(estate, ix[:, 2]), stats(estate, ix[:, 3]), stats(genes, ix[:, 4])
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        self.estate1, self.estate2 = t((estate - m1) / s1), t((estate - m2) / s2)
        self.genes = t((genes - mg) / sg)
        self.n_features = self.n_onehot + 2 + 2 * estate.shape[1] + genes.shape[1]

    def codes(self, rows, swap):
        a, b = ("d2", "d1") if swap else ("d1", "d2")
        ca, cb = ("c2", "c1") if swap else ("c1", "c2")
        return np.stack([
            np.array([self.conc_index[k] for k in dose_key(rows[ca])]),
            np.array([self.conc_index[k] for k in dose_key(rows[cb])]),
            rows[a].map(self.drug_index).to_numpy(), rows[b].map(self.drug_index).to_numpy(),
            rows.cell_id.map(self.cell_index).to_numpy(),
            rows[ca].to_numpy(float), rows[cb].to_numpy(float)], 1).astype(np.float64)

    def to_device(self, codes):
        return (torch.tensor(codes[:, :5].astype(np.int64), device=self.device),
                torch.tensor((codes[:, 5:7] - self.conc_mu) / self.conc_sd, dtype=torch.float32, device=self.device))

    def batch(self, idx_codes, conc):
        """(one-hot column indices, dense auxiliary block) for TFFM.forward_codes."""
        dense = torch.cat([conc, self.estate1[idx_codes[:, 2]], self.estate2[idx_codes[:, 3]],
                           self.genes[idx_codes[:, 4]]], 1)
        return idx_codes + self.offsets_t, dense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", type=float, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", type=Path, default=OUT)
    a = ap.parse_args()
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B`")
    if a.smoke and a.out_root.resolve() == OUT.resolve():
        raise SystemExit("--smoke needs its own --out-root")
    sys.stdout.reconfigure(encoding="utf-8")
    out = a.out_root / f"reg{a.reg:g}_rank{a.rank}" / f"seed_{a.seed}_ep{a.epochs}"
    if out.exists():
        raise SystemExit(f"Already run: {out}")
    start = time.monotonic()
    frames = {r: gate.load_points(r) for r in ("train", "validation")}
    if a.smoke:
        keep = lambda f, n: f[f.matrix_key.isin(f.matrix_key.drop_duplicates().sample(n, random_state=0))]
        frames = {"train": keep(frames["train"], 3000), "validation": keep(frames["validation"], 500)}
    va = frames["validation"]
    hsa = np.minimum(va.mono_response_a, va.mono_response_b).to_numpy(float)
    hsa_pooled = 100 * np.sqrt(np.mean((hsa - va.response.to_numpy(float)) ** 2))
    if not a.smoke:
        assert abs(hsa_pooled - 11.839) < 5e-4, hsa_pooled  # same validation target and HSA as Table 3

    drug_ids = sorted(set(pd.concat(frames.values()).drug_a_id) | set(pd.concat(frames.values()).drug_b_id))
    cell_ids = sorted(set(pd.concat(frames.values()).cell_id))
    rows = build_rows(frames)
    val_rows = va.assign(d1=va.drug_a_id, d2=va.drug_b_id, c1=va.dose_a_M, c2=va.dose_b_M, y=va.response_percent)
    dev = torch.device("cuda")
    enc = Encoder(rows, pd.concat([rows, val_rows]), drug_ids, cell_ids, estate_bits(drug_ids),
                  gene_table(frames, cell_ids), dev)
    tr_idx, tr_conc = enc.to_device(np.concatenate([enc.codes(rows, False), enc.codes(rows, True)]))
    tr_y = torch.tensor(np.concatenate([rows.y, rows.y]), dtype=torch.float32, device=dev)
    va_codes = [enc.to_device(enc.codes(val_rows, s)) for s in (False, True)]
    print(f"START D21 comboFM reg={a.reg:g} rank={a.rank} seed={a.seed} epochs={a.epochs} features={enc.n_features} "
          f"train_rows={len(tr_y)} validation_cells={len(va)} hsa_val_pooled={hsa_pooled:.4f}", flush=True)

    model = tffm_torch.TFFM(enc.n_features, order=5, rank=a.rank, init_std=0.01, seed=a.seed).to(dev)
    best = dict(surface_macro=np.inf)
    history = []

    def validate():
        preds = [tffm_torch.predict(model, lambda i, c=c: (enc.batch(c[0][i], c[1][i]), None), len(va),
                                    forward=model.forward_codes) for c in va_codes]
        pred = (preds[0] + preds[1]) / 200.0  # mean of both drug orders, percent -> ratio
        m = gate.pilot.metric(va.response.to_numpy(float), pred, va.surface_replicate_id.to_numpy())
        return pred, m["surface_macro_rmse"], 100 * m["point_rmse"]

    def on_epoch(epoch):
        if epoch % EVAL_EVERY and epoch != a.epochs:
            return
        pred, sm, pooled = validate()
        history.append(dict(epoch=epoch, validation_surface_macro_rmse=sm, validation_pooled_rmse_percent=pooled))
        if sm < best["surface_macro"]:
            best.update(surface_macro=sm, pooled=pooled, epoch=epoch, pred=pred)
        print(f"EPOCH {epoch} val_surface_macro={sm:.6f} val_pooled={pooled:.4f} best_epoch={best['epoch']} "
              f"({time.monotonic() - start:.0f}s)", flush=True)

    tffm_torch.fit(model, lambda i: (enc.batch(tr_idx[i], tr_conc[i]), tr_y[i]), len(tr_y), n_epochs=a.epochs,
                   batch_size=1024, lr=1e-3, reg=a.reg, seed=a.seed, on_epoch=on_epoch, forward=model.forward_codes)
    result = dict(task="D21", card="docs/COMBOFM_CARD_20261027.md", reg=a.reg, rank=a.rank, order=5, seed=a.seed,
                  max_epochs=a.epochs, eval_every=EVAL_EVERY, smoke=a.smoke, split_3_read=False,
                  best_epoch=best["epoch"], validation_surface_macro_rmse=best["surface_macro"],
                  validation_pooled_rmse_percent=best["pooled"], hsa_validation_pooled_rmse_percent=hsa_pooled,
                  n_features=enc.n_features, train_rows=int(len(tr_y)), history=history,
                  total_wall_seconds=time.monotonic() - start)
    out.mkdir(parents=True)  # created only now, so a crashed run leaves nothing behind
    np.savez_compressed(out / "validation_predictions.npz", prediction=best["pred"],
                        matrix_key=va.matrix_key.to_numpy(), row_i=va.row_i.to_numpy(), col_j=va.col_j.to_numpy())
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"SELECTED reg={a.reg:g} rank={a.rank} seed={a.seed} best_epoch={best['epoch']} "
          f"val_surface_macro={best['surface_macro']:.6f} val_pooled={best['pooled']:.4f} "
          f"({result['total_wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

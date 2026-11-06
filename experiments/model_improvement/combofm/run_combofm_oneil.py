"""D21 (docs/COMBOFM_CARD_20261027.md): comboFM on O'Neil `oneil-pairout-v1` (card E v2 data, D13 guard).

Same model, rows, features and selection as run_combofm.py (NCI), with O'Neil inputs: 4x4 combination grids,
monotherapy from the batch-matched values Ours uses (mono_a_percent / mono_b_percent, inputs on every role), the
968-gene table, and the O'Neil SMILES. Monotherapy edge rows come from the train, validation and test matrices (all
inputs; comboFM assumes known monotherapy); no validation or test combination cell is trained on. Test responses are
never read here: common.load_points("test") has no response column and the card E guard is installed first.

Tuning runs (--epochs 100) write metrics only. Final runs (--epochs 200 --final) also write validation and test
predictions at the best validation epoch through common.write_predictions("combofm", seed, ...); the test is scored
once by experiments/oneil/eval/evaluate_combofm_test.py.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/model_improvement/combofm/run_combofm_oneil.py --reg 1e4 --rank 50 --seed 41 --epochs 100
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/oneil"))
import common  # noqa: E402

common.install_guard()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(HERE))
import tffm_torch  # noqa: E402
from run_combofm import EVAL_EVERY, Encoder  # noqa: E402

OUT = ROOT / "runs/model_improvement/combofm/oneil"


def estate_bits(drug_ids):
    from rdkit import Chem
    from rdkit.Chem.EState.Fingerprinter import FingerprintMol
    smiles = pd.read_csv(common.DRUG / "drugs.csv", keep_default_na=False).set_index("drug_id").canonical_smiles
    bits = np.stack([(FingerprintMol(Chem.MolFromSmiles(smiles[d]))[0] > 0).astype(np.float32) for d in drug_ids])
    return bits[:, bits.var(0) > 0]


def build_rows(train, mono_frames):
    combo = train.assign(d1=train.drug_a_id, d2=train.drug_b_id, c1=train.dose_a_M, c2=train.dose_b_M,
                         y=train.response_percent)
    parts = [combo]
    for f in mono_frames:
        a = f.drop_duplicates(["matrix_key", "row_i"])
        b = f.drop_duplicates(["matrix_key", "col_j"])
        parts.append(a.assign(d1=a.drug_a_id, d2=a.drug_b_id, c1=a.dose_a_M, c2=0.0, y=a.mono_a_percent))
        parts.append(b.assign(d1=b.drug_a_id, d2=b.drug_b_id, c1=0.0, c2=b.dose_b_M, y=b.mono_b_percent))
    cols = ["d1", "d2", "c1", "c2", "cell_id", "y"]
    return pd.concat([p[cols] for p in parts], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", type=float, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="write validation and test predictions (200-epoch runs)")
    ap.add_argument("--out-root", type=Path, default=OUT)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = a.out_root / f"reg{a.reg:g}_rank{a.rank}" / f"seed_{a.seed}_ep{a.epochs}"
    if out.exists():
        raise SystemExit(f"Already run: {out}")
    start = time.monotonic()
    tr, va, te = (common.load_points(r) for r in ("train", "validation", "test"))
    allf = pd.concat([tr, va, te])
    drug_ids = sorted(set(allf.drug_a_id) | set(allf.drug_b_id))
    cell_ids = sorted(set(allf.cell_id))
    cell_name = allf.drop_duplicates("cell_id").set_index("cell_id").deepsynba_cell
    genes = pd.read_csv(common.GENE_CSV).set_index("genes")[[cell_name[c] for c in cell_ids]].T.to_numpy(np.float32)
    rows = build_rows(tr, (tr, va, te))
    as_rows = lambda f: f.assign(d1=f.drug_a_id, d2=f.drug_b_id, c1=f.dose_a_M, c2=f.dose_b_M)
    val_rows, test_rows = as_rows(va), as_rows(te)
    dev = torch.device("cuda")
    enc = Encoder(rows, pd.concat([rows, val_rows, test_rows]), drug_ids, cell_ids, estate_bits(drug_ids), genes, dev)
    tr_idx, tr_conc = enc.to_device(np.concatenate([enc.codes(rows, False), enc.codes(rows, True)]))
    tr_y = torch.tensor(np.concatenate([rows.y, rows.y]), dtype=torch.float32, device=dev)
    codes = {r: [enc.to_device(enc.codes(f, s)) for s in (False, True)] for r, f in (("validation", val_rows),
                                                                                       ("test", test_rows))}
    hsa = np.minimum(va.mono_response_a, va.mono_response_b).to_numpy(float)
    print(f"START D21 comboFM O'Neil reg={a.reg:g} rank={a.rank} seed={a.seed} epochs={a.epochs} "
          f"features={enc.n_features} train_rows={len(tr_y)} validation_cells={len(va)} "
          f"hsa_val_pooled={common.pooled_rmse_percent(va.response, hsa):.4f}", flush=True)

    model = tffm_torch.TFFM(enc.n_features, order=5, rank=a.rank, init_std=0.01, seed=a.seed).to(dev)
    best, history = dict(surface_macro=np.inf), []

    def predict(role):
        preds = [tffm_torch.predict(model, lambda i, c=c: (enc.batch(c[0][i], c[1][i]), None), len(c[1]),
                                    forward=model.forward_codes) for c in codes[role]]
        return (preds[0] + preds[1]) / 200.0  # mean of both drug orders, percent -> ratio

    def on_epoch(epoch):
        if epoch % EVAL_EVERY and epoch != a.epochs:
            return
        pred = predict("validation")
        sm, pooled = common.surface_macro_percent(va, pred), common.pooled_rmse_percent(va.response, pred)
        history.append(dict(epoch=epoch, validation_surface_macro_percent=sm, validation_pooled_rmse_percent=pooled))
        if sm < best["surface_macro"]:
            best.update(surface_macro=sm, pooled=pooled, epoch=epoch, val=pred, test=predict("test") if a.final else None)
        print(f"EPOCH {epoch} val_surface_macro={sm:.4f} val_pooled={pooled:.4f} best_epoch={best['epoch']} "
              f"({time.monotonic() - start:.0f}s)", flush=True)

    tffm_torch.fit(model, lambda i: (enc.batch(tr_idx[i], tr_conc[i]), tr_y[i]), len(tr_y), n_epochs=a.epochs,
                   batch_size=1024, lr=1e-3, reg=a.reg, seed=a.seed, on_epoch=on_epoch, forward=model.forward_codes)
    meta = dict(task="D21", card="docs/COMBOFM_CARD_20261027.md", dataset="oneil-pairout-v1", reg=a.reg, rank=a.rank,
                order=5, seed=a.seed, max_epochs=a.epochs, eval_every=EVAL_EVERY, best_epoch=best["epoch"],
                validation_surface_macro_percent=best["surface_macro"], validation_pooled_rmse_percent=best["pooled"],
                n_features=enc.n_features, train_rows=int(len(tr_y)), test_response_read=False, history=history,
                total_wall_seconds=time.monotonic() - start)
    if a.final:
        for role, frame, pred in (("validation", va, best["val"]), ("test", te, best["test"])):
            common.write_predictions("combofm", a.seed, role, frame[common.KEY].assign(prediction=pred),
                                     meta={k: v for k, v in meta.items() if k != "history"})
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"SELECTED reg={a.reg:g} rank={a.rank} seed={a.seed} best_epoch={best['epoch']} "
          f"val_surface_macro={best['surface_macro']:.4f} val_pooled={best['pooled']:.4f} "
          f"({meta['total_wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

"""D11 (card D): DeepSynBa v2 (author code + experimenter grid-point config, exactly as the accepted repro v2) on the
new-drug split, one seed (10, as in the repro). Only the data roles change: the repro's train/val cache (split 1 + 2,
no test rows) is re-assigned with cold_split.py's drug lists (DeepSynBa names mapped to canonical IDs through the S1
drug mapping); train = rows without a held-out drug, checkpoint selection on cold_val (authors' rule, min val loss
over 500 epochs), then the selected checkpoint predicts the cold_test rows (16 entries each). No metric here.

Run with the DeepSynBa environment (author code is read only, on sys.path through repro_common):
  C:/LHZ/research/bisurfgnn/.venvs/deepsynba/Scripts/python.exe -I experiments/model_improvement/cold_drug/run_cold_deepsynba.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2]
REPRO = Path("C:/LHZ/research/bisurfgnn/experiments/audit_revision/deepsynba_repro_v2")
sys.path.insert(0, str(REPRO))
import repro_common as rc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SPLIT = EXP / "runs/model_improvement/bisurfgnn_v2_dev/cold_drug/split.json"
S1 = EXP / "runs/model_improvement/public_split_gate_v1/s1"


def roles(d) -> np.ndarray:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    mapping = pd.read_csv(S1 / "drug_mapping_used.csv")
    name2id = {rc.normalize_drug_name(str(n)).strip(): c for n, c in zip(mapping.deepsynba_drug, mapping.canonical_drug_id)}
    ids1 = np.array([name2id.get(rc.normalize_drug_name(str(x)).strip(), "?") for x in d["drug1"]])
    ids2 = np.array([name2id.get(rc.normalize_drug_name(str(x)).strip(), "?") for x in d["drug2"]])
    test, val = list(split["test_drugs"]), list(split["val_drugs"])
    in_test = np.isin(ids1, test) | np.isin(ids2, test)
    in_val = np.isin(ids1, val) | np.isin(ids2, val)
    return np.where(in_test, "cold_test", np.where(in_val, "cold_val", "train")), int(((ids1 == "?") | (ids2 == "?")).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 2000/500/500 rows")
    a = ap.parse_args()
    out = SPLIT.parent / ("smoke_deepsynba_seed_10" if a.smoke else "deepsynba_seed_10")
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader
    from trainer import TrainingModule  # author code, read only

    cfg = rc.build_config(out)
    if (cfg["model_params"]["gene_channel"], cfg["model_params"]["emb_size"]) != (2048, 256):
        raise SystemExit("unexpected config")
    if a.smoke:
        cfg["trainer_params"]["max_epochs"] = 2
    t0 = time.time()
    L.seed_everything(rc.SEED)
    d = rc.load_cache()
    role, unmapped = roles(d)
    idx = {r: np.where(role == r)[0] for r in ("train", "cold_val", "cold_test")}
    if a.smoke:
        idx = {r: v[:n] for (r, v), n in zip(idx.items(), (2000, 500, 500))}
    cell2gex, emb = rc.load_features()
    ds = {r: rc.SubsetCacheDataset(d, v, cell2gex, emb) for r, v in idx.items()}
    bs, nw = cfg["trainer_params"]["batch_size"], cfg["trainer_args"]["num_workers"]
    train_loader = DataLoader(ds["train"], batch_size=bs, num_workers=nw, shuffle=True, pin_memory=True, drop_last=True,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(ds["cold_val"], batch_size=bs, num_workers=nw, shuffle=False, pin_memory=True,
                            persistent_workers=nw > 0)
    module = TrainingModule(config=cfg)
    ckpt = ModelCheckpoint(dirpath=out / "checkpoints", monitor="val/dose_response_loss", mode="min", filename="{epoch:03d}",
                           save_top_k=1, save_last=False)
    trainer = L.Trainer(devices=[0], accelerator="gpu", strategy="auto", precision=cfg["trainer_params"]["precision"],
                        gradient_clip_val=cfg["trainer_params"]["grad_norm_clip"], max_epochs=cfg["trainer_params"]["max_epochs"],
                        logger=CSVLogger(out, name="logs"), callbacks=[ckpt],
                        log_every_n_steps=cfg["trainer_params"]["logging_interval"], enable_progress_bar=False)
    print(f"START cold DeepSynBa train {len(ds['train'])} cold_val {len(ds['cold_val'])} cold_test {len(ds['cold_test'])} "
          f"unmapped_rows {unmapped}", flush=True)
    trainer.fit(module, train_loader, val_loader)
    best = TrainingModule.load_from_checkpoint(str(ckpt.best_model_path), strict=True, map_location="cuda").eval().to("cuda")
    loader = DataLoader(ds["cold_test"], batch_size=bs, num_workers=0, shuffle=False)
    preds, keys = [], []
    with torch.no_grad():
        for batch in loader:
            o = best.model(*(batch[k].to("cuda") for k in ("drug1_smile", "drug2_smile", "cell_gex", "drug1_dose", "drug2_dose")))
            preds.append(o["dose_response"].cpu().numpy().reshape(-1, 4, 4))
            keys.extend("_".join(k) for k in zip(batch["drug1"], batch["drug2"], batch["cell"]))
    np.savez_compressed(out / "cold_test_predictions.npz", predictions=np.concatenate(preds), keys=np.array(keys))
    (out / "metrics.json").write_text(json.dumps(
        {"seed": rc.SEED, "best_checkpoint": str(ckpt.best_model_path), "best_cold_val_loss": float(ckpt.best_model_score),
         "epochs": trainer.current_epoch, "rows": {r: int(len(v)) for r, v in idx.items()}, "unmapped_rows": unmapped,
         "wall_s": round(time.time() - t0, 1), "cold_test_scored_here": False}, indent=1), encoding="utf-8")
    print("DONE cold DeepSynBa", flush=True)


if __name__ == "__main__":
    main()

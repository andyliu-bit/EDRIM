"""NCI60-Pretrain-v1 pretraining of ``bisurfgnn.models.sparse_mono.MonoPriorNet`` on NCI-60 single-agent data.

Same module, same per-step objective as SparseMono ``NetPrior.fit`` (MSE on (drug, cell, dose) monotherapy,
AdamW, gradient clip 1.0, batch of points sampled uniformly with replacement); the only engineering change is
that each step encodes the unique molecules of the batch (not every molecule of a ~50k library) plus every
cell graph, exactly as NetPrior encodes all drugs and cells once per step. Compound-level 5% hold-out for early
stopping; the best hold-out checkpoint is kept.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .nci60 import GraphBank, sha256_file


def holdout_mask(nsc: np.ndarray, salt: str, modulus: int) -> np.ndarray:
    """Response-free compound split: sha256(salt|nsc) mod ``modulus`` == 0 -> hold-out."""
    return np.array([int(hashlib.sha256(f"{salt}|{int(n)}".encode()).hexdigest(), 16) % modulus == 0 for n in nsc])


def graph_hash(item) -> str:
    """Same digest as independent_graph_benchmark._graph_hash."""
    d = hashlib.sha256()
    for t in (item.node_features, item.adjacency):
        v = t.detach().cpu().contiguous()
        d.update(str(v.dtype).encode("ascii")); d.update(json.dumps(list(v.shape), separators=(",", ":")).encode("ascii"))
        d.update(v.numpy().tobytes())
    return d.hexdigest()


class PointTensors:
    def __init__(self, frame: pd.DataFrame, gindex: dict[int, int], cindex: dict[str, int], device):
        self.g = torch.as_tensor(frame.nsc.map(gindex).to_numpy(np.int64), device=device)
        cat = frame.cell.astype("category")
        lut = np.array([cindex[str(x)] for x in cat.cat.categories], dtype=np.int64)
        codes = cat.cat.codes.to_numpy()
        if (codes < 0).any():
            raise ValueError("Point without a cell")
        self.c = torch.as_tensor(lut[codes], device=device)
        self.z = torch.as_tensor(frame.dose_u.to_numpy(np.float32), device=device)
        self.y = torch.as_tensor(frame.y.to_numpy(np.float32), device=device)

    def __len__(self):
        return len(self.y)


@torch.no_grad()
def evaluate(model, bank: GraphBank, cell_in, pts: PointTensors, chunk_drugs: int = 1024, chunk_points: int = 1 << 16):
    model.eval()
    uniq, inv = torch.unique(pts.g, return_inverse=True)
    enc = torch.cat([model.drug_encoder(bank.batch(uniq[i:i + chunk_drugs])).pooled for i in range(0, len(uniq), chunk_drugs)])
    cells = model.cell_encoder(cell_in).pooled
    se = torch.zeros((), device=pts.y.device, dtype=torch.float64)
    for i in range(0, len(pts), chunk_points):
        p = model.predict(enc, cells, inv[i:i + chunk_points], pts.c[i:i + chunk_points], pts.z[i:i + chunk_points])
        se += ((p - pts.y[i:i + chunk_points]).double() ** 2).sum()
    return float(torch.sqrt(se / len(pts)))


def pretrain(model, bank: GraphBank, cell_in, train: PointTensors, hold: PointTensors, hp: dict, sched: dict,
             seed: int, out: Path, log=print) -> dict:
    """hp: SparseMono net_prior entry (lr, weight_decay, batch); sched: eval_every, patience_evals, max_steps,
    time_limit_minutes, min_delta."""
    dev = train.y.device
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    gen = torch.Generator(device=dev).manual_seed(seed)
    best = dict(rmse=evaluate(model, bank, cell_in, hold), step=0, state=copy.deepcopy(model.state_dict()))
    curve = [dict(step=0, train_mse_window=float("nan"), holdout_rmse=best["rmse"], elapsed_s=0.0, max_atoms_window=0)]
    log(f"  step 0 holdout_rmse {best['rmse']:.5f}")
    t0 = time.monotonic(); acc = torch.zeros((), device=dev, dtype=torch.float64); n_acc = 0; bad = 0; max_atoms = 0
    stop = "max_steps"
    for step in range(1, sched["max_steps"] + 1):
        model.train()
        i = torch.randint(0, len(train), (hp["batch"],), device=dev, generator=gen)
        uniq, inv = torch.unique(train.g[i], return_inverse=True)
        gb = bank.batch(uniq)
        max_atoms = max(max_atoms, gb.node_mask.shape[1])
        drugs = model.drug_encoder(gb).pooled
        cells = model.cell_encoder(cell_in).pooled
        loss = torch.mean((model.predict(drugs, cells, inv, train.c[i], train.z[i]) - train.y[i]) ** 2)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        acc += loss.detach().double(); n_acc += 1
        if step % sched["eval_every"] == 0:
            tr = float(acc / n_acc)
            if not np.isfinite(tr):
                raise RuntimeError("Non-finite pretraining loss")
            r = evaluate(model, bank, cell_in, hold); el = time.monotonic() - t0
            curve.append(dict(step=step, train_mse_window=tr, holdout_rmse=r, elapsed_s=el, max_atoms_window=max_atoms))
            pd.DataFrame(curve).to_csv(out / "curve.csv", index=False)
            improved = r < best["rmse"] - sched["min_delta"]
            if improved:
                best.update(rmse=r, step=step, state=copy.deepcopy(model.state_dict())); bad = 0
            else:
                bad += 1
            log(f"  step {step} train_mse {tr:.5f} holdout_rmse {r:.5f} best {best['rmse']:.5f}@{best['step']} {el / 60:.1f} min")
            acc.zero_(); n_acc = 0; max_atoms = 0
            if bad >= sched["patience_evals"]:
                stop = "early_stopping"; break
            if el > 60 * sched["time_limit_minutes"]:
                stop = "time_limit"; break
    model.load_state_dict(best["state"])
    return dict(best_step=best["step"], best_holdout_rmse=best["rmse"], stop_reason=stop, last_step=step,
                train_seconds=time.monotonic() - t0, curve=curve)


__all__ = ["holdout_mask", "graph_hash", "PointTensors", "evaluate", "pretrain"]

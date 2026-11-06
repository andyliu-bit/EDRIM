"""CDCR-Rank (adapted) on the public NCI new-combination split: split 1 trains, split 2 selects, no split-3 file is read.

Launch card docs/CDCRRANK_ADAPTATION_CARD_20261103.md. PyTorch re-implementation of CDCR-Rank (Moragheb et al.,
Bioinformatics Advances 6(1) vbag051, 2026). Layer sizes from the public code (ParvinRazzaghi/CDCR-Rank, commit 9240ec8,
lines 257-293 and 376-461); lists and the uRank loss from the paper (Eq. 5), whose code version has no gradient. The
model never sees the monotherapy of a test matrix (L1-implicit): monotherapy cells are training targets only. Data
handling, SMILES coding, dose scaling and metrics reuse the DeepDRP adapter.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/cdcrrank_adapted/cdcrrank.py --seed 41 --lam 0.5
  .venv/Scripts/python.exe -I -B experiments/model_improvement/cdcrrank_adapted/cdcrrank.py --export-check out.npz
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "deepdrp_adapted"))
import deepdrp as dd  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

RUNS = dd.ROOT / "runs/model_improvement/bisurfgnn_v2_dev/cdcrrank_adapted"
N_GENES, BATCH, LR, WEIGHT_DECAY = 1000, 16, 1e-4, 1e-5      # paper: 1,000 genes, AdamW 1e-4 / 1e-5; 16 matrices
LR_PATIENCE, LR_FACTOR, LR_MIN = 3, 0.5, 1e-6                # public code: ReduceLROnPlateau
DEVIATIONS = {
    "C1": "PyTorch re-implementation: the public script fills the cell-line features with np.random.randn, holds out "
          "random (drug, drug, cell line) triplets, uses only the first 3 rows of each combination, and its uRank "
          "multiplies elementwise, so T / exp(O) = M: the term does not depend on the predictions, and the gradient "
          "float32 returns for it is rounding noise scaled by gains of up to 2^108 (keras_check.py)",
    "C2": "lists as in the paper (all dose tuples of a pair and cell line): the 15 non-control cells of a training "
          "matrix, 9 combination + 6 monotherapy cells whose targets are the split-1 lookup values (as for DeepDRP)",
    "C3": "uRank as the paper's Eq. 5 (matrix product): sum_i g_i ln(1 + sum_{s_j < s_i} exp(O_j - O_i)) / (n - 1), "
          "log-sum-exp; gain g = 2^s' - 1 with s' = response min-max scaled to [0, 1] over the split-1 list cells "
          "(neither paper nor code scales s; 2^s of percent growth overflows above 128 and is negative below 0)",
    "C4": "hybrid weight lambda chosen on split 2 from {0.5 (code), 0.1, 0.02} (paper: not stated); outputs and MSE "
          "in percent, as in the code",
    "C5": "1,000 highest-variance genes of CellMiner 2.15 RNA-seq composite over the 59 split-1 cells, z-scored, into "
          "the code's MLP 128/256/128 (paper: 1,000 most variable genes reduced to 256-d)",
    "C6": "SMILES and doses as DeepDRP D4/D5: dative arrows removed, truncation at 100; log10 molar scaled to [0, 1] "
          "on split-1 doses, dose code zero for an absent drug (code: raw concentrations)",
    "C7": "AdamW lr 1e-4, weight decay 1e-5 (paper); 16 matrices = 240 cells per batch (paper: 256); early stopping "
          "on split-2 combination MSE, max 100 epochs (code: 20-30 epochs, paper: 10,000); one random drug order per "
          "matrix and epoch, prediction = mean of both orders. Reported runs keep the lr constant, as in the paper, "
          "with patience 10 (--lr-patience 0 --patience 10, user decision 11-03); under the code's schedule, the "
          "defaults (lr halved after 3 epochs without improvement, minimum 1e-6, patience 5), seeds 41-45 ended "
          "between 11.8% and 18.2% depending on when validation noise first triggered a cut",
    "C8": "padding token 0 has a trained embedding: the code's mask_zero=True does not reach the convolutions "
          "(keras_check.py)",
}


def cell_features(cells: list[str], n: int = N_GENES) -> tuple[torch.Tensor, list[str]]:
    """C5 (DeepDRP D3 with n genes): the n highest-variance genes over the split-1 cells, z-scored per gene."""
    e = pd.read_parquet(dd.EXPRESSION)
    x = e[cells].to_numpy(np.float64).T
    top = np.argsort(-x.var(0), kind="stable")[:n]
    z = x[:, top]
    z = (z - z.mean(0)) / z.std(0)
    genes = [f"{g}|{i}" for g, i in zip(e["Gene name"].iloc[top], e["Entrez gene id"].iloc[top])]
    return torch.tensor(z, dtype=torch.float32), genes


# ---------------------------------------------------------------------------------------------------------------
# model and loss
# ---------------------------------------------------------------------------------------------------------------
class CDCRRank(nn.Module):
    """Public code lines 257-293 and 376-461: SMILES embedding and 1D-CNN shared by both drugs (128-d each), 64-d
    sinusoidal dose codes, cell MLP 128/256/128, head 512-256-128-64-1 with dropout 0.3 / 0.2. Keras initialisers."""

    def __init__(self, n_genes: int = N_GENES):
        super().__init__()
        self.embed = nn.Embedding(len(dd.SMILES_DICT) + 1, 128)
        self.conv = nn.ModuleList([nn.Conv1d(128, 32, 4), nn.Conv1d(32, 64, 4), nn.Conv1d(64, 128, 4)])
        self.cell = nn.ModuleList([nn.Linear(n_genes, 128), nn.Linear(128, 256), nn.Linear(256, 128)])
        self.head = nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 1)])
        i = torch.arange(64, dtype=torch.float64)
        self.register_buffer("factor", (10.0 ** (-4.0 * i / 64)).float())      # 10^(-4i/64), Eq. 1
        self.register_buffer("odd", i % 2 == 1)                                # sin on even, cos on odd dimensions
        with torch.no_grad():
            self.embed.weight.uniform_(-0.05, 0.05)
            for m in self.conv:
                dd.glorot_(m.weight, m.in_channels * 4, m.out_channels * 4)
                m.bias.zero_()
            for m in [*self.cell, *self.head]:
                dd.glorot_(m.weight, m.in_features, m.out_features)
                m.bias.zero_()

    def tables(self, tokens, cellx):
        x = self.embed(tokens).transpose(1, 2)
        for k, m in enumerate(self.conv):
            x = F.relu(m(F.pad(x, (1, 2))))              # Keras padding='same', kernel 4: 1 left, 2 right
            if k < 2:
                x = F.max_pool1d(x, 2)
        h = cellx
        for m in self.cell:
            h = F.relu(m(h))
        return x.amax(2), h

    def dose(self, u, present):
        a = u.unsqueeze(-1) * self.factor
        return torch.where(self.odd, torch.cos(a), torch.sin(a)) * present.unsqueeze(-1).to(a.dtype)

    def forward(self, drug_tab, cell_tab, d1, d2, c, u1, u2, p1, p2):
        x = torch.cat([drug_tab[d1], drug_tab[d2], self.dose(u1, p1), self.dose(u2, p2), cell_tab[c]], -1)
        for k, m in enumerate(self.head[:-1]):
            x = F.relu(m(x))
            if k < 2:
                x = F.dropout(x, (0.3, 0.2)[k], self.training)
        return self.head[-1](x).squeeze(-1)


def urank(o: torch.Tensor, s: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """C3 for lists in rows [B, n]: mean over lists of sum_i gain_i ln(1 + sum_{j: s_j < s_i} exp(o_j - o_i)) / (n-1)."""
    lower = s.unsqueeze(1) < s.unsqueeze(2)                                  # [B, i, j]: s_j < s_i
    d = (o.unsqueeze(1) - o.unsqueeze(2)).masked_fill(~lower, float("-inf"))  # o_j - o_i
    lse = torch.logsumexp(torch.cat([torch.zeros_like(o).unsqueeze(-1), d], -1), -1)   # ln(1 + sum exp), no -inf row
    return (gain * lse).sum(1).mean() / (o.shape[1] - 1)


def urank_selfcheck() -> float:
    """Max |vectorised - loop| over random lists with ties, float64 (card: < 1e-6)."""
    g = torch.Generator().manual_seed(0)
    o = 30.0 * torch.randn(3, 15, generator=g, dtype=torch.float64)
    s = 100.0 * torch.rand(3, 15, generator=g, dtype=torch.float64)
    s[0, :5] = s[0, 5]
    gain = 2.0 ** (s / 100.0) - 1.0
    loop = sum(float(gain[b, i]) * math.log1p(sum(math.exp(float(o[b, j] - o[b, i])) for j in range(15)
                                                  if s[b, j] < s[b, i]))
               for b in range(3) for i in range(15)) / (3 * 14)
    err = abs(float(urank(o, s, gain)) - loop)
    assert err < 1e-6, err
    return err


def predict(model, tokens, cellx, M, nodes=None, chunk: int = 8192) -> torch.Tensor:
    """[n, len(nodes)] predictions in percent, mean of both drug orders; nodes default to the combination cells."""
    model.eval()
    k = M.combo if nodes is None else nodes
    out = torch.empty(M.n, len(k), device=M.y.device)
    with torch.no_grad():
        tables = model.tables(tokens, cellx)
        for lo in range(0, M.n, chunk):
            m = torch.arange(lo, min(lo + chunk, M.n), device=k.device).unsqueeze(1).expand(-1, len(k))
            kk = k.expand_as(m)
            out[lo:lo + len(m)] = sum(model(*tables, *dd.node_inputs(M, m, kk // M.G, kk % M.G,
                                                                     torch.full_like(m, s, dtype=torch.bool)))
                                      for s in (False, True)) / 2
    return out


# ---------------------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------------------
def make_step(model, opt, tokens, cellx, M, lam, s_all, gain_all, nodes):
    """One optimisation step on matrices m with drug order swapped where sw; returns [loss, mse, urank]."""
    G = M.G

    def step(m, sw):
        mm = m.unsqueeze(1).expand(-1, len(nodes))
        kk = nodes.expand_as(mm)
        o = model(*model.tables(tokens, cellx), *dd.node_inputs(M, mm, kk // G, kk % G, sw.unsqueeze(1).expand_as(mm)))
        mse, rank = F.mse_loss(o, s_all[m]), urank(o, s_all[m], gain_all[m])
        loss = lam * mse + (1.0 - lam) * rank
        opt.zero_grad(set_to_none=False)                        # static gradient buffers for the CUDA graph
        loss.backward()
        opt.step()
        return torch.stack([loss.detach(), mse.detach(), rank.detach()])
    return step


def capture(model, opt, step, dev):
    """CUDA graph of one full-batch step: same computation, launched as one graph (the eager step is launch-bound,
    about 20 ms). The warm-up steps the capture needs are undone in place: parameters back to their initial values,
    gradients and optimizer state back to zero, i.e. a fresh AdamW."""
    init = [p.detach().clone() for p in model.parameters()]
    sm, ssw = torch.arange(BATCH, device=dev), torch.zeros(BATCH, dtype=torch.bool, device=dev)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step(sm, ssw)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(sm, ssw)
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), init):
            p.copy_(p0)
            p.grad.zero_()
        for state in opt.state.values():
            for v in state.values():
                v.zero_()
    return graph, sm, ssw, out


def train(model, tokens, cellx, Mtr, Mva, lam, seed, max_epochs, patience, log, lr0=LR, lr_patience=LR_PATIENCE,
          passes=1):
    """lr_patience 0 keeps the learning rate constant. An epoch is `passes` shuffled passes over the training matrices
    (O'Neil: as many steps as one NCI epoch); validation, early stopping and lr halving count these epochs."""
    dev, G = tokens.device, Mtr.G
    nodes = torch.arange(1, G * G, device=dev)                  # C2: the non-control cells of a matrix form a list
    s_all = 100.0 * Mtr.y[:, nodes]
    lo, hi = float(s_all.min()), float(s_all.max())
    gain_all = 2.0 ** ((s_all - lo) / (hi - lo)) - 1.0          # C3
    yv = 100.0 * Mva.y[:, Mva.combo]
    lr_t = torch.tensor(lr0, device=dev)                         # a tensor, so the captured step sees lr changes
    opt = torch.optim.AdamW(model.parameters(), lr=lr_t, weight_decay=WEIGHT_DECAY, capturable=True)
    step = make_step(model, opt, tokens, cellx, Mtr, lam, s_all, gain_all, nodes)
    model.train()
    graph, sm, ssw, out = capture(model, opt, step, dev)
    full = Mtr.n // BATCH * BATCH
    hist, best, best_state, best_epoch, bad, lr_wait, stop = [], float("inf"), None, -1, 0, 0, "max_epochs"
    for epoch in range(max_epochs):
        t0 = time.perf_counter()
        model.train()
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)
        acc = torch.zeros(3, device=dev)
        for _ in range(passes):
            order = torch.randperm(Mtr.n, generator=gen).to(dev)
            swap = (torch.rand(Mtr.n, generator=gen) < 0.5).to(dev)
            for b in range(0, full, BATCH):
                sm.copy_(order[b:b + BATCH])
                ssw.copy_(swap[sm])
                graph.replay()
                acc += out
            if full < Mtr.n:                                     # last, smaller batch: the same step, eager
                m = order[full:]
                acc += step(m, swap[m])
        tr = (acc / (passes * math.ceil(Mtr.n / BATCH))).tolist()
        if not all(math.isfinite(v) for v in tr):
            raise RuntimeError("Non-finite training loss")
        val_mse = float(((predict(model, tokens, cellx, Mva) - yv) ** 2).mean())
        lr = float(lr_t)
        hist.append(dict(epoch=epoch, train_loss=tr[0], train_mse=tr[1], train_urank=tr[2], val_mse=val_mse,
                         val_rmse_percent=math.sqrt(val_mse), lr=lr, seconds=round(time.perf_counter() - t0, 1)))
        log(f"epoch {epoch} loss {tr[0]:.4f} mse {tr[1]:.3f} urank {tr[2]:.4f} val_rmse% {math.sqrt(val_mse):.3f} "
            f"lr {lr:.2e} {hist[-1]['seconds']}s")
        if val_mse < best:
            best, best_state, best_epoch, bad, lr_wait = val_mse, copy.deepcopy(model.state_dict()), epoch, 0, 0
            continue
        bad, lr_wait = bad + 1, lr_wait + 1
        if bad >= patience:
            stop = "early_stopping"
            break
        if lr_patience and lr_wait >= lr_patience:              # Keras ReduceLROnPlateau: wait resets after a cut
            lr_t.fill_(max(lr * LR_FACTOR, LR_MIN))
            lr_wait = 0
    model.load_state_dict(best_state)
    return dict(history=hist, best_epoch=best_epoch, best_val_mse=best, stopped_by=stop,
                gain_scale_percent=[lo, hi], list_cells=int(len(nodes)), train_matrices=int(Mtr.n), passes_per_epoch=passes,
                steps_per_epoch=passes * math.ceil(Mtr.n / BATCH))


# ---------------------------------------------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------------------------------------------
def setup(device, smoke: bool = False) -> dict:
    s = dd.setup(device, smoke)
    cellx, s["genes"] = cell_features(s["cells"])
    s["cellx"] = cellx.to(device)
    return s


def load_model(path: Path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model = CDCRRank(len(state["genes"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), state


def run(seed: int, lam: float, out_dir: Path, max_epochs: int, patience: int, smoke: bool, lr: float = LR,
        lr_patience: int = LR_PATIENCE) -> dict:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    out_dir.mkdir(parents=True, exist_ok=False)
    log_file = (out_dir / "log.txt").open("w", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    t_start = time.perf_counter()
    selfcheck = urank_selfcheck()
    dd.seed_everything(seed)
    s = setup(device, smoke)
    Mtr, Mva, tokens, cellx = s["Mtr"], s["Mva"], s["tokens"], s["cellx"]
    log(f"train matrices {Mtr.n}, validation matrices {Mva.n}, drugs {len(s['drugs'])}, cells {len(s['cells'])}, "
        f"lambda {lam}, urank self-check {selfcheck:.1e}")
    model = CDCRRank().to(device)
    h = train(model, tokens, cellx, Mtr, Mva, lam, seed, max_epochs, patience, log, lr, lr_patience)
    pv = predict(model, tokens, cellx, Mva)
    torch.save({"model": model.state_dict(), "drugs": s["drugs"], "cells": s["cells"], "genes": s["genes"],
                "dose_range": s["lohi"], "control_percent": s["control"], "lam": lam},
               out_dir / "model_selected.pt")
    m2, _ = load_model(out_dir / "model_selected.pt", device)
    if float((predict(m2, tokens, cellx, Mva) - pv).abs().max()) > 1e-4:
        raise RuntimeError("Reloaded model does not reproduce the validation predictions")
    pred, val = Mva.frame_order(pv), s["val"]
    np.savez_compressed(out_dir / "validation_predictions.npz", prediction=pred, matrix_key=val.matrix_key.to_numpy(str),
                        row_i=val.row_i.to_numpy(), col_j=val.col_j.to_numpy())
    result = dict(
        method="CDCR-Rank (adapted)", seed=seed, lam=lam, lr=lr, lr_patience=lr_patience, patience=patience,
        max_epochs=max_epochs, weight_decay=WEIGHT_DECAY, batch_matrices=BATCH,
        smoke=smoke, split_read=[1, 2], test_rows_read=0, validation=dd.metrics(pred, val),
        training={k: v for k, v in h.items() if k != "history"}, urank_selfcheck_max_abs=selfcheck,
        parameters=sum(p.numel() for p in model.parameters()), genes=s["genes"], dose_range_log10=s["lohi"],
        control_percent=s["control"], deviations=DEVIATIONS,
        source="ParvinRazzaghi/CDCR-Rank @ 9240ec8 (drug_dose_urank.py)",
        wall_seconds=round(time.perf_counter() - t_start, 1),
        gpu=dict(name=torch.cuda.get_device_name(device), peak_allocated_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20)),
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__))
    (out_dir / "history.json").write_text(json.dumps(h["history"], indent=1), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    log(f"validation combination RMSE% {result['validation']['combination_rmse_percent']:.4f} (lookup HSA "
        f"{result['validation']['lookup_hsa_rmse_percent']:.4f}) best epoch {h['best_epoch']} wall {result['wall_seconds']}s")
    log_file.close()
    return result


def graph_check(steps: int = 20) -> float:
    """Graph replay against eager steps from the same weights on the same batches, dropout off (eval mode), so both
    paths compute the same thing: max |difference| of the parameters after `steps` steps, relative to the largest."""
    dev = torch.device("cuda")
    s = setup(dev, smoke=True)
    M, tokens, cellx = s["Mtr"], s["tokens"], s["cellx"]
    nodes = torch.arange(1, M.G * M.G, device=dev)
    s_all = 100.0 * M.y[:, nodes]
    gain_all = 2.0 ** ((s_all - s_all.min()) / (s_all.max() - s_all.min())) - 1.0
    batches = torch.randperm(M.n, generator=torch.Generator().manual_seed(0))[:steps * BATCH].view(steps, BATCH).to(dev)
    swap = (torch.rand(M.n, generator=torch.Generator().manual_seed(1)) < 0.5).to(dev)
    final = []
    for use_graph in (False, True):
        dd.seed_everything(0)
        model = CDCRRank().to(dev).eval()
        opt = torch.optim.AdamW(model.parameters(), lr=torch.tensor(LR, device=dev), weight_decay=WEIGHT_DECAY,
                                capturable=True)
        step = make_step(model, opt, tokens, cellx, M, 0.5, s_all, gain_all, nodes)
        graph, sm, ssw, _ = capture(model, opt, step, dev) if use_graph else (None,) * 4
        for m in batches:
            if use_graph:
                sm.copy_(m)
                ssw.copy_(swap[m])
                graph.replay()
            else:
                step(m, swap[m])
        final.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    return float((final[0] - final[1]).abs().max() / final[0].abs().max())


def export_check(path: Path) -> None:
    """Weights, inputs and outputs of a freshly initialised model on 64 validation combination cells, for
    keras_check.py (run with TensorFlow outside this environment). CPU, so no TF32 convolutions."""
    device = torch.device("cpu")
    s = setup(device, smoke=True)
    dd.seed_everything(0)
    model = CDCRRank().to(device).eval()
    M = s["Mva"]
    m = torch.arange(64) % M.n
    k = torch.tensor(dd.COMBO)[torch.arange(64) % 9]
    d1, d2, c, u1, u2, p1, p2 = dd.node_inputs(M, m, k // 4, k % 4, torch.zeros(64, dtype=torch.bool))
    with torch.no_grad():
        drug_tab, cell_tab = model.tables(s["tokens"], s["cellx"])
        concat = torch.cat([drug_tab[d1], drug_tab[d2], model.dose(u1, p1), model.dose(u2, p2), cell_tab[c]], -1)
        pred = model(drug_tab, cell_tab, d1, d2, c, u1, u2, p1, p2)
    w = {name.replace(".", "_"): p.detach().numpy() for name, p in model.state_dict().items()}
    np.savez(path, **w, tok1=s["tokens"][d1].numpy(), tok2=s["tokens"][d2].numpy(), cell=s["cellx"][c].numpy(),
             u1=u1.numpy(), u2=u2.numpy(), concat=concat.numpy(), pred=pred.numpy(),
             list_y=(100.0 * M.y[0, 1:]).numpy())                # one 15-cell validation list (percent), for the uRank check
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--lr-patience", type=int, default=LR_PATIENCE, help="0: constant learning rate")
    ap.add_argument("--smoke", action="store_true", help="2,000 training / 1,000 validation matrices")
    ap.add_argument("--export-check", type=Path)
    ap.add_argument("--graph-check", action="store_true", help="CUDA-graph step against the eager step, 20 steps")
    args = ap.parse_args()
    if args.export_check:
        return export_check(args.export_check)
    if args.graph_check:
        torch.use_deterministic_algorithms(True, warn_only=True)
        return print(f"graph vs eager, max relative parameter difference after 20 steps: {graph_check():.2e}")
    if args.seed is None or args.lam is None:
        ap.error("--seed and --lam are required")
    name = (f"lam{args.lam:g}_seed{args.seed}" + (f"_lr{args.lr:g}" if args.lr != LR else "")
            + (f"_lrpat{args.lr_patience}" if args.lr_patience != LR_PATIENCE else "")
            + (f"_pat{args.patience}" if args.patience != 5 else "") + (f"_ep{args.max_epochs}" if args.max_epochs != 100 else "")
            + ("_smoke" if args.smoke else ""))
    run(args.seed, args.lam, RUNS / name, args.max_epochs, args.patience, args.smoke, args.lr, args.lr_patience)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                       # a CUDA error otherwise hangs the process at teardown
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

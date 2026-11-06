"""DeepDRP (adapted) on the public NCI new-combination split: split 1 trains, split 2 selects, no split-3 file is read.

Launch card docs/DEEPDRP_ADAPTATION_CARD_20261103.md. PyTorch re-implementation of DeepDRP (Moragheb et al., PLOS One
21(5) e0348908, 2026). The public code (ParvinRazzaghi/DeepDRP-Dose-Response-Predictions-, commit 37f28d1) does not run
as released, so the layer sizes come from it and the local stage from the paper's S2 (whole pairs held out,
monotherapy in training), where the product of the two monotherapy curves stands in for the held-out matrix.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/deepdrp_adapted/deepdrp.py --seed 41 --global-lr 1e-4
  .venv/Scripts/python.exe -I -B experiments/model_improvement/deepdrp_adapted/deepdrp.py --export-check out.npz
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT.parent / "bisurfgnn"                       # read-only main checkout (CellMiner, drug structures)
S1 = ROOT / "runs/model_improvement/public_split_gate_v1/s1"
RUNS = ROOT / "runs/model_improvement/bisurfgnn_v2_dev/deepdrp_adapted"
DRUGS_CSV = DATA / "data/processed/nci_almanac_unclipped/drugs.csv"
EXPRESSION = DATA / "data/processed/cellminer/expression.parquet"
COLUMNS = ["matrix_key", "drug_a_id", "drug_b_id", "biological_cell_id", "row_i", "col_j", "dose_a_M", "dose_b_M",
           "mono_a_percent", "mono_b_percent", "response_percent"]
# public code, line 255 (0 is padding)
SMILES_DICT = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, "1": 35, "0": 3, "3": 36, "2": 4,
               "5": 37, "4": 5, "7": 38, "6": 6, "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
               "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, "O": 48, "N": 14, "P": 15,
               "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20,
               "a": 55, "c": 56, "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, "l": 25,
               "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
SMILES_LEN, N_GENES, K_NEIGHBOURS = 100, 78, 4
GLOBAL_BATCH, LOCAL_BATCH, LOCAL_LR = 512, 128, 1e-3          # public code: batch 512, GCN learning rate 1e-3


def combo_nodes(G: int) -> list[int]:
    """Node G*r + c of a G x G grid, r / c = dose index of drug a / b, 0 = absent; combinations have r, c >= 1."""
    return [G * r + c for r in range(1, G) for c in range(1, G)]


COMBO = combo_nodes(4)                                          # NCI 4x4 matrices
DEVIATIONS = {
    "D1": "PyTorch re-implementation: the public script does not run (drug indices cover 400,000 of its 800,000 "
          "training rows; the stage-2 model is used before it is defined); its 352-d embedding input is declared "
          "int32 and is kept in floating point here",
    "D2": "local stage as in the paper's S2: the 16 cells of each 4x4 matrix are the nodes, each linked to its 4 "
          "nearest cells by dose index; node input = global embedding (dropout 0.5, public code) + monotherapy-"
          "product (Bliss) value from the split-1 lookup, kept out of the dropout because it plays the role of the "
          "paper's propagated labels; one GCN shared by all matrices, as in the public code",
    "D3": "78 highest-variance genes of CellMiner 2.15 RNA-seq composite over the 59 split-1 cells, z-scored "
          "(paper: top 0.5% of rcellminer expression = 78 genes)",
    "D4": "canonical SMILES with dative-bond arrows removed (outside the authors' dictionary); truncation at 100",
    "D5": "doses: log10 molar scaled to [0, 1] on split-1 doses, sinusoidal code zero for an absent drug "
          "(public code passes raw concentrations of unstated unit)",
    "D6": "split 2 drives early stopping (patience 5, max 100 epochs) and the global learning-rate choice; both drug "
          "orders in training (public code), prediction = mean of the two orders",
    "D7": "local stage trained on batches of 128 matrices (public code: one matrix per step)",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def glorot_(w: torch.Tensor, fan_in: int, fan_out: int) -> None:
    """Keras glorot_uniform."""
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        w.uniform_(-bound, bound)


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------
def load_split(role: str) -> pd.DataFrame:
    return pd.read_parquet(S1 / f"points_{role}.parquet", columns=COLUMNS)


def control_percent() -> float:
    """Control cell value of the split-1 lookup, as in the test frame (evaluate_gate.Lookup.control)."""
    return float(json.loads((S1 / "s1_report.json").read_text(encoding="utf-8"))["lookup"]["control_mean_percent"])


def drug_tokens(drugs: list[str], path: Path = DRUGS_CSV) -> torch.Tensor:
    smiles = pd.read_csv(path, keep_default_na=False).set_index("drug_id").canonical_smiles
    out = np.zeros((len(drugs), SMILES_LEN), np.int64)
    for k, d in enumerate(drugs):
        s = smiles[d].replace("->", "").replace("<-", "")      # D4
        bad = sorted(set(s) - set(SMILES_DICT))
        if bad:
            raise RuntimeError(f"{d}: SMILES characters outside the DeepDRP dictionary: {bad}")
        out[k, :min(len(s), SMILES_LEN)] = [SMILES_DICT[ch] for ch in s[:SMILES_LEN]]
    return torch.tensor(out)


def cell_features(cells: list[str]) -> tuple[torch.Tensor, list[str]]:
    """D3: the 78 highest-variance genes over the split-1 cells, z-scored per gene."""
    e = pd.read_parquet(EXPRESSION)
    x = e[cells].to_numpy(np.float64).T
    top = np.argsort(-x.var(0), kind="stable")[:N_GENES]
    z = x[:, top]
    z = (z - z.mean(0)) / z.std(0)
    genes = [f"{g}|{i}" for g, i in zip(e["Gene name"].iloc[top], e["Entrez gene id"].iloc[top])]
    return torch.tensor(z, dtype=torch.float32), genes


def dose_range(train: pd.DataFrame) -> tuple[float, float]:
    d = np.log10(np.r_[train.dose_a_M.to_numpy(), train.dose_b_M.to_numpy()])
    return float(d.min()), float(d.max())


class Matrices:
    """Points of one split as G x G grids (index 0 = drug absent; NCI G = 4, O'Neil G = 5); matrices in order of
    first appearance."""

    def __init__(self, frame: pd.DataFrame, drug_pos: dict, cell_pos: dict, lohi, control: float, device,
                 codes: np.ndarray | None = None, G: int = 4):
        if codes is None:
            codes = pd.factorize(frame.matrix_key)[0]
        n, i, j, C = int(codes.max()) + 1, frame.row_i.to_numpy(), frame.col_j.to_numpy(), (G - 1) ** 2
        slot = np.bincount(codes * C + (i - 1) * (G - 1) + (j - 1), minlength=C * n)
        if len(frame) != C * n or slot.max() != 1:
            raise RuntimeError("Every matrix must hold its combination cells exactly once")

        def per_matrix(col, pos):
            v = frame[col].to_numpy()
            out = np.empty(n, dtype=object)
            out[codes] = v
            if not (out[codes] == v).all():
                raise RuntimeError(f"{col} varies within a matrix")
            idx = pd.Series(out).map(pos)
            if idx.isna().any():
                raise RuntimeError(f"{col}: entity outside the split-1 index")
            return idx.to_numpy(np.int64)

        def per_axis(col, axis):
            v = frame[col].to_numpy(np.float64)
            out = np.zeros((n, G))
            out[codes, axis] = v
            if not np.array_equal(out[codes, axis], v):
                raise RuntimeError(f"{col} varies along its dose axis")
            return out

        dose_a, dose_b = per_axis("dose_a_M", i), per_axis("dose_b_M", j)
        mono_a, mono_b = per_axis("mono_a_percent", i), per_axis("mono_b_percent", j)
        ref = np.empty((n, G, G))
        ref[:, 0, 0] = control
        ref[:, 1:, 0], ref[:, 0, 1:] = mono_a[:, 1:], mono_b[:, 1:]
        ref[:, 1:, 1:] = mono_a[:, 1:, None] * mono_b[:, None, 1:] / 100.0       # Bliss, as in evaluate_gate
        y = ref.copy()                                     # control and monotherapy cells carry their lookup values
        labelled = "response_percent" in frame.columns     # O'Neil test inputs carry no responses
        if labelled:
            y[codes, i, j] = frame.response_percent.to_numpy(np.float64)
        lo, hi = lohi

        def scaled(d):                                     # D5
            return np.where(d > 0, (np.log10(np.where(d > 0, d, 1.0)) - lo) / (hi - lo), 0.0)

        f32 = dict(dtype=torch.float32, device=device)
        self.n, self.G, self.codes, self.i, self.j = n, G, codes, i, j
        self.combo = torch.tensor(combo_nodes(G), device=device)
        self.da = torch.tensor(per_matrix("drug_a_id", drug_pos), device=device)
        self.db = torch.tensor(per_matrix("drug_b_id", drug_pos), device=device)
        self.cc = torch.tensor(per_matrix("biological_cell_id", cell_pos), device=device)
        self.ua, self.ub = torch.tensor(scaled(dose_a), **f32), torch.tensor(scaled(dose_b), **f32)
        self.ref = torch.tensor(ref.reshape(n, G * G) / 100.0, **f32)       # ratio units
        self.y = torch.tensor(y.reshape(n, G * G) / 100.0, **f32)
        back = 100.0 * self.frame_order(self.y[:, self.combo])
        if labelled and np.abs(back - frame.response_percent.to_numpy(np.float64)).max() > 1e-3:
            raise RuntimeError("Grid layout does not reproduce the point table")
        if not labelled and np.abs(100.0 * self.frame_order(self.ref[:, self.combo])
                                   - frame.mono_a_percent.to_numpy(np.float64)
                                   * frame.mono_b_percent.to_numpy(np.float64) / 100.0).max() > 1e-3:
            raise RuntimeError("Grid layout does not reproduce the monotherapy product")

    def frame_order(self, grid_c: torch.Tensor) -> np.ndarray:
        """[n, (G-1)^2] combination-node values (combo_nodes order) -> one value per row of the source frame."""
        k = (self.i - 1) * (self.G - 1) + (self.j - 1)
        return grid_c.detach().cpu().numpy().astype(np.float64)[self.codes, k]


def node_inputs(M: Matrices, m, r, c, swap):
    """Global-network inputs of node (r, c) of matrices m; drug order swapped where swap is True."""
    ua, ub, pa, pb = M.ua[m, r], M.ub[m, c], r > 0, c > 0
    return (torch.where(swap, M.db[m], M.da[m]), torch.where(swap, M.da[m], M.db[m]), M.cc[m],
            torch.where(swap, ub, ua), torch.where(swap, ua, ub), torch.where(swap, pb, pa), torch.where(swap, pa, pb))


def grid_adjacency(G: int = 4, k: int = K_NEIGHBOURS) -> torch.Tensor:
    """Each cell linked to its k nearest cells by dose index (paper: four nearest by dose), self loops added and
    normalised as in the public code: Adj.dot(d).transpose().dot(d), d = diag(rowsum ** -0.5)."""
    pos = np.array([(r, c) for r in range(G) for c in range(G)], float)
    dist = ((pos[:, None] - pos[None]) ** 2).sum(-1)
    np.fill_diagonal(dist, np.inf)
    a = np.zeros((G * G, G * G))
    for v in range(G * G):
        a[v, np.argsort(dist[v], kind="stable")[:k]] = 1.0          # ties broken by node order
    assert a[G + 1].nonzero()[0].tolist() == [1, G, G + 2, 2 * G + 1], "interior cell must link to its 4 neighbours"
    a += np.eye(G * G)
    d = np.diag(a.sum(1) ** -0.5)
    return torch.tensor(d @ a.T @ d, dtype=torch.float32)


# ---------------------------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------------------------
class GlobalNet(nn.Module):
    """Public code lines 292-346: SMILES 1D-CNN shared by both drugs, 3-layer cell MLP, sinusoidal dose codes,
    352-d embedding, dense 512, output. Keras initialisers."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(SMILES_DICT) + 1, 128)
        self.conv = nn.ModuleList([nn.Conv1d(128, 32, 4), nn.Conv1d(32, 64, 4), nn.Conv1d(64, 96, 4)])
        self.cell = nn.ModuleList([nn.Linear(N_GENES, 32), nn.Linear(32, 64), nn.Linear(64, 96)])
        self.fc, self.out = nn.Linear(352, 512), nn.Linear(512, 1)
        self.register_buffer("omega", 2.0 * math.pi * torch.exp(torch.linspace(0.0, math.log(1000.0), 16)))
        with torch.no_grad():
            self.embed.weight.uniform_(-0.05, 0.05)
            for m in self.conv:
                glorot_(m.weight, m.in_channels * 4, m.out_channels * 4)
                m.bias.zero_()
            for m in [*self.cell, self.fc, self.out]:
                glorot_(m.weight, m.in_features, m.out_features)
                m.bias.zero_()

    def tables(self, tokens, cellx):
        x = self.embed(tokens).transpose(1, 2)
        for m in self.conv:
            x = F.relu(m(x))
        h = cellx
        for m in self.cell:
            h = F.relu(m(h))
        return x.amax(2), h

    def dose(self, u, present):
        a = self.omega * u.unsqueeze(-1)
        return torch.cat([torch.sin(a), torch.cos(a)], -1) * present.unsqueeze(-1).to(a.dtype)

    def forward(self, drug_tab, cell_tab, d1, d2, c, u1, u2, p1, p2):
        emb = torch.cat([drug_tab[d1], drug_tab[d2], cell_tab[c], self.dose(u1, p1), self.dose(u2, p2)], -1)
        return emb, self.out(F.relu(self.fc(emb))).squeeze(-1)


class LocalNet(nn.Module):
    """Public code lines 555-571: dropout, GraphConvolution(16, relu), dropout, GraphConvolution(1), concatenated with
    the global prediction, dense(1). GraphConvolution = adj @ x @ W + b (kegra). Node input: D2."""

    def __init__(self, adj: torch.Tensor):
        super().__init__()
        self.register_buffer("adj", adj)
        self.w1, self.b1 = nn.Parameter(torch.empty(353, 16)), nn.Parameter(torch.zeros(16))
        self.w2, self.b2 = nn.Parameter(torch.empty(16, 1)), nn.Parameter(torch.zeros(1))
        self.agg = nn.Linear(2, 1)
        glorot_(self.w1, 353, 16)
        glorot_(self.w2, 16, 1)
        glorot_(self.agg.weight, 2, 1)
        with torch.no_grad():
            self.agg.bias.zero_()

    def forward(self, emb, ref, gpred):
        x = torch.cat([F.dropout(emb, 0.5, self.training), ref.unsqueeze(-1)], -1)      # D2: ref is not dropped
        h = F.dropout(F.relu(self.adj @ x @ self.w1 + self.b1), 0.5, self.training)
        y = self.adj @ h @ self.w2 + self.b2
        return self.agg(torch.cat([y, gpred.unsqueeze(-1)], -1)).squeeze(-1)


def local_forward(g, local, tables, M, m, swap):
    """[B, G*G] final predictions of matrices m (global network frozen)."""
    N = M.G * M.G
    mm = m.unsqueeze(1).expand(-1, N)
    kk = torch.arange(N, device=m.device).expand_as(mm)
    with torch.no_grad():
        emb, gp = g(*tables, *node_inputs(M, mm, kk // M.G, kk % M.G, swap.unsqueeze(1).expand(-1, N)))
    return local(emb, M.ref[m], gp)


def predict_global(g, tokens, cellx, M, chunk: int = 8192) -> torch.Tensor:
    """[n, (G-1)^2] global predictions of the combination cells, mean of both drug orders (ratio units)."""
    g.eval()
    k = M.combo
    out = torch.empty(M.n, len(k), device=M.y.device)
    with torch.no_grad():
        tables = g.tables(tokens, cellx)
        for lo in range(0, M.n, chunk):
            m = torch.arange(lo, min(lo + chunk, M.n), device=k.device).unsqueeze(1).expand(-1, len(k))
            kk = k.expand_as(m)
            out[lo:lo + len(m)] = sum(g(*tables, *node_inputs(M, m, kk // M.G, kk % M.G,
                                                              torch.full_like(m, s, dtype=torch.bool)))[1]
                                      for s in (False, True)) / 2
    return out


def predict_final(g, local, tokens, cellx, M, chunk: int = 4096) -> torch.Tensor:
    g.eval()
    local.eval()
    out = torch.empty(M.n, len(M.combo), device=M.y.device)
    with torch.no_grad():
        tables = g.tables(tokens, cellx)
        for lo in range(0, M.n, chunk):
            m = torch.arange(lo, min(lo + chunk, M.n), device=out.device)
            out[lo:lo + len(m)] = sum(local_forward(g, local, tables, M, m,
                                                    torch.full_like(m, s, dtype=torch.bool))[:, M.combo]
                                      for s in (False, True)) / 2
    return out


# ---------------------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------------------
def train_global(g, tokens, cellx, Mtr, Mva, lr, seed, max_epochs, patience, log):
    """Every non-control cell of every training matrix in both drug orders (monotherapy stays in training, S2)."""
    dev, G = tokens.device, Mtr.G
    nodes = torch.arange(1, G * G, device=dev)
    em = torch.arange(Mtr.n, device=dev).repeat_interleave(2 * len(nodes))
    ek = nodes.repeat(2 * Mtr.n)
    es = torch.tensor([False] * len(nodes) + [True] * len(nodes), device=dev).repeat(Mtr.n)
    yv = Mva.y[:, Mva.combo]
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    hist, best, best_state, best_epoch, bad, stop = [], float("inf"), None, -1, 0, "max_epochs"
    for epoch in range(max_epochs):
        t0 = time.perf_counter()
        g.train()
        order = torch.randperm(len(em), generator=torch.Generator().manual_seed(seed * 1000 + epoch)).to(dev)
        losses = []
        for lo in range(0, len(order), GLOBAL_BATCH):
            idx = order[lo:lo + GLOBAL_BATCH]
            m, k = em[idx], ek[idx]
            pred = g(*g.tables(tokens, cellx), *node_inputs(Mtr, m, k // G, k % G, es[idx]))[1]
            loss = F.mse_loss(pred, Mtr.y[m, k])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        train_loss = float(torch.stack(losses).mean())
        if not math.isfinite(train_loss):
            raise RuntimeError("Non-finite global training loss")
        val_loss = float(((predict_global(g, tokens, cellx, Mva) - yv) ** 2).mean())
        hist.append(dict(epoch=epoch, train_mse=train_loss, val_mse=val_loss, val_rmse_percent=100 * math.sqrt(val_loss),
                         seconds=round(time.perf_counter() - t0, 1)))
        log(f"global epoch {epoch} train_mse {train_loss:.5f} val_rmse% {100 * math.sqrt(val_loss):.3f} "
            f"{hist[-1]['seconds']}s")
        if val_loss < best:
            best, best_state, best_epoch, bad = val_loss, copy.deepcopy(g.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                stop = "early_stopping"
                break
    g.load_state_dict(best_state)
    return dict(history=hist, best_epoch=best_epoch, best_val_mse=best, stopped_by=stop, train_entries=len(em))


def train_local(g, local, tokens, cellx, Mtr, Mva, seed, max_epochs, patience, log):
    dev = tokens.device
    g.eval()
    with torch.no_grad():
        tables = g.tables(tokens, cellx)
    yv = Mva.y[:, Mva.combo]
    opt = torch.optim.Adam(local.parameters(), lr=LOCAL_LR)
    hist, best, best_state, best_epoch, bad, stop = [], float("inf"), None, -1, 0, "max_epochs"
    for epoch in range(max_epochs):
        t0 = time.perf_counter()
        local.train()
        gen = torch.Generator().manual_seed(seed * 7919 + epoch)
        order = torch.randperm(Mtr.n, generator=gen).to(dev)
        swap = (torch.rand(Mtr.n, generator=gen) < 0.5).to(dev)
        losses = []
        for lo in range(0, Mtr.n, LOCAL_BATCH):
            m = order[lo:lo + LOCAL_BATCH]
            out = local_forward(g, local, tables, Mtr, m, swap[m])
            loss = F.mse_loss(out[:, Mtr.combo], Mtr.y[m][:, Mtr.combo])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        train_loss = float(torch.stack(losses).mean())
        if not math.isfinite(train_loss):
            raise RuntimeError("Non-finite local training loss")
        val_loss = float(((predict_final(g, local, tokens, cellx, Mva) - yv) ** 2).mean())
        hist.append(dict(epoch=epoch, train_mse=train_loss, val_mse=val_loss, val_rmse_percent=100 * math.sqrt(val_loss),
                         seconds=round(time.perf_counter() - t0, 1)))
        log(f"local epoch {epoch} train_mse {train_loss:.5f} val_rmse% {100 * math.sqrt(val_loss):.3f} "
            f"{hist[-1]['seconds']}s")
        if val_loss < best:
            best, best_state, best_epoch, bad = val_loss, copy.deepcopy(local.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                stop = "early_stopping"
                break
    local.load_state_dict(best_state)
    return dict(history=hist, best_epoch=best_epoch, best_val_mse=best, stopped_by=stop)


def metrics(pred_percent: np.ndarray, frame: pd.DataFrame) -> dict:
    y = frame.response_percent.to_numpy(np.float64)
    e = pred_percent - y
    codes = pd.factorize(frame.matrix_key)[0]
    per = np.sqrt(np.bincount(codes, weights=e * e) / np.bincount(codes))
    a, b = frame.mono_a_percent.to_numpy(np.float64), frame.mono_b_percent.to_numpy(np.float64)
    return dict(combination_rmse_percent=float(np.sqrt(np.mean(e * e))), surface_macro_rmse_percent=float(per.mean()),
                pearson=float(np.corrcoef(pred_percent, y)[0, 1]), bias_percent=float(e.mean()),
                lookup_bliss_rmse_percent=float(np.sqrt(np.mean((a * b / 100.0 - y) ** 2))),
                lookup_hsa_rmse_percent=float(np.sqrt(np.mean((np.minimum(a, b) - y) ** 2))),
                points=int(len(y)), matrices=int(codes.max() + 1))


# ---------------------------------------------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------------------------------------------
def setup(device, smoke: bool = False):
    train, val = load_split("train"), load_split("validation")
    if smoke:                                               # first 2,000 / 1,000 matrices
        train = train[train.matrix_key.isin(train.matrix_key.unique()[:2000])].reset_index(drop=True)
        val = val[val.matrix_key.isin(val.matrix_key.unique()[:1000])].reset_index(drop=True)
    drugs = sorted(set(train.drug_a_id) | set(train.drug_b_id))
    cells = sorted(set(train.biological_cell_id))
    if not smoke and (not (set(val.drug_a_id) | set(val.drug_b_id)) <= set(drugs)
                      or not set(val.biological_cell_id) <= set(cells)):
        raise RuntimeError("Validation drugs/cells must all appear in split 1 (new-combination split)")
    if smoke:                                               # smoke subsets need the full entity index
        full = load_split("train")
        drugs, cells = sorted(set(full.drug_a_id) | set(full.drug_b_id)), sorted(set(full.biological_cell_id))
        lohi = dose_range(full)
    else:
        lohi = dose_range(train)
    tokens = drug_tokens(drugs).to(device)
    cellx, genes = cell_features(cells)
    pos_d, pos_c = {d: i for i, d in enumerate(drugs)}, {c: i for i, c in enumerate(cells)}
    control = control_percent()
    return dict(train=train, val=val, drugs=drugs, cells=cells, genes=genes, lohi=lohi, control=control,
                tokens=tokens, cellx=cellx.to(device),
                Mtr=Matrices(train, pos_d, pos_c, lohi, control, device),
                Mva=Matrices(val, pos_d, pos_c, lohi, control, device))


def load_model(path: Path, device, G: int = 4):
    state = torch.load(path, map_location=device, weights_only=False)
    g, local = GlobalNet().to(device), LocalNet(grid_adjacency(G).to(device)).to(device)
    g.load_state_dict(state["global"])
    local.load_state_dict(state["local"])
    return g.eval(), local.eval(), state


def run(seed: int, global_lr: float, out_dir: Path, max_epochs: int, patience: int, smoke: bool) -> dict:
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
    seed_everything(seed)
    s = setup(device, smoke)
    Mtr, Mva, tokens, cellx = s["Mtr"], s["Mva"], s["tokens"], s["cellx"]
    log(f"train matrices {Mtr.n}, validation matrices {Mva.n}, drugs {len(s['drugs'])}, cells {len(s['cells'])}")
    g = GlobalNet().to(device)
    h_global = train_global(g, tokens, cellx, Mtr, Mva, global_lr, seed, max_epochs, patience, log)
    gv = predict_global(g, tokens, cellx, Mva)
    seed_everything(seed + 1)
    local = LocalNet(grid_adjacency().to(device)).to(device)
    h_local = train_local(g, local, tokens, cellx, Mtr, Mva, seed, max_epochs, patience, log)
    pv = predict_final(g, local, tokens, cellx, Mva)
    torch.save({"global": g.state_dict(), "local": local.state_dict(), "drugs": s["drugs"], "cells": s["cells"],
                "genes": s["genes"], "dose_range": s["lohi"], "control_percent": s["control"]},
               out_dir / "model_selected.pt")
    g2, local2, _ = load_model(out_dir / "model_selected.pt", device)
    if float((predict_final(g2, local2, tokens, cellx, Mva) - pv).abs().max()) > 1e-6:
        raise RuntimeError("Reloaded model does not reproduce the validation predictions")
    pred, gpred = 100.0 * Mva.frame_order(pv), 100.0 * Mva.frame_order(gv)
    val = s["val"]
    np.savez_compressed(out_dir / "validation_predictions.npz", prediction=pred, global_only=gpred,
                        matrix_key=val.matrix_key.to_numpy(str), row_i=val.row_i.to_numpy(), col_j=val.col_j.to_numpy())
    result = dict(
        method="DeepDRP (adapted)", seed=seed, global_lr=global_lr, local_lr=LOCAL_LR, smoke=smoke,
        split_read=[1, 2], test_rows_read=0, validation=metrics(pred, val), validation_global_only=metrics(gpred, val),
        global_stage={k: v for k, v in h_global.items() if k != "history"},
        local_stage={k: v for k, v in h_local.items() if k != "history"},
        parameters={"global": sum(p.numel() for p in g.parameters()), "local": sum(p.numel() for p in local.parameters())},
        genes=s["genes"], dose_range_log10=s["lohi"], control_percent=s["control"], deviations=DEVIATIONS,
        source="ParvinRazzaghi/DeepDRP-Dose-Response-Predictions- @ 37f28d119855eb5e3cf07c0da16dcecc1e990fc4",
        wall_seconds=round(time.perf_counter() - t_start, 1),
        gpu=dict(name=torch.cuda.get_device_name(device), peak_allocated_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20)),
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__))
    (out_dir / "history.json").write_text(json.dumps({"global": h_global["history"], "local": h_local["history"]},
                                                     indent=1), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    log(f"validation combination RMSE% {result['validation']['combination_rmse_percent']:.4f} "
        f"(global only {result['validation_global_only']['combination_rmse_percent']:.4f}, "
        f"lookup Bliss {result['validation']['lookup_bliss_rmse_percent']:.4f}) wall {result['wall_seconds']}s")
    log_file.close()
    return result


def export_check(path: Path) -> None:
    """Weights, inputs and outputs of a freshly initialised global network on 64 validation combination cells, for
    keras_check.py (run with TensorFlow outside this environment). CPU, so no TF32 convolutions."""
    device = torch.device("cpu")
    s = setup(device, smoke=True)
    seed_everything(0)
    g = GlobalNet().to(device).eval()
    M = s["Mva"]
    m = torch.arange(64, device=device) % M.n
    k = torch.tensor(COMBO, device=device)[torch.arange(64, device=device) % 9]
    inp = node_inputs(M, m, k // 4, k % 4, torch.zeros(64, dtype=torch.bool, device=device))
    with torch.no_grad():
        emb, pred = g(*g.tables(s["tokens"], s["cellx"]), *inp)
    w = {name: p.detach().cpu().numpy() for name, p in g.state_dict().items()}
    np.savez(path, **{k.replace(".", "_"): v for k, v in w.items()},
             tok1=s["tokens"][inp[0]].cpu().numpy(), tok2=s["tokens"][inp[1]].cpu().numpy(),
             cell=s["cellx"][inp[2]].cpu().numpy(), u1=inp[3].cpu().numpy(), u2=inp[4].cpu().numpy(),
             emb=emb.cpu().numpy(), pred=pred.cpu().numpy())
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--global-lr", type=float)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="2,000 training / 1,000 validation matrices")
    ap.add_argument("--export-check", type=Path)
    args = ap.parse_args()
    if args.export_check:
        return export_check(args.export_check)
    if args.seed is None or args.global_lr is None:
        ap.error("--seed and --global-lr are required")
    name = f"lr{args.global_lr:g}_seed{args.seed}" + ("_smoke" if args.smoke else "")
    run(args.seed, args.global_lr, RUNS / name, args.max_epochs, args.patience, args.smoke)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                       # a CUDA error otherwise hangs the process at teardown
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

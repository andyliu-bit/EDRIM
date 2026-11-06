"""DD-PRiSM measured-monotherapy adapter ("DD-PRiSM-Hallmark-MeasuredMono") on the public NCI new-combination split.

Development use under docs/BISURFGNN_V2_DEV_PROTOCOL.md section 4. Data: PublicSplitGate-v1 S1 point tables only
(split 1 = training, split 2 = validation), hash-checked against s1/manifest.json. No split-3 file is ever opened.

The combination network reads, per combination cell, the two drugs' pathway attentions and their MEASURED monotherapy
viabilities, which are the S1 split-1 lookup anchors (mono_a_percent / 100, mono_b_percent / 100): exactly the anchors
BiSurfGNN, XGBoost and lookup HSA use, so all sit at the same information level.

Author code (GIST-CSBL/DD-PRiSM, commit 8d9c5d5, CC BY-NC-SA 4.0) is executed from the hash-bound local notebook cache;
the only edit is the pathway count (186 KEGG -> 50 Hallmark), via bisurfgnn.comparators.ddprism.

Deviations from the authors' code (also written into every run's config.json):
  D1 measured monotherapy (S1 lookup anchors) replaces the monotherapy model's predicted viability.
  D2 50 MSigDB Hallmark gene sets with CellMiner gene z-scores fit on the split-1 cells replace 186 KEGG sets.
  D3 notebooks 02/03 (NCI-60 pretraining, ALMANAC fine-tuning) are not cached locally and NCI-60 data would add
     external knowledge BiSurfGNN does not have: the monotherapy model is trained from scratch on the split-1
     monotherapy anchors with notebook 04's recipe (reconstructed, not the authors' 02/03 code). It only supplies
     the pathway attention.
  D4 standard eval mode for every validation loss and prediction. The authors' test_comb / predict_comb first run a
     train-mode pass over the evaluation batches, refreshing BatchNorm running statistics with evaluation inputs;
     that refresh is NOT used here.
  D5 split 2 drives the learning-rate schedule, early stopping and epoch selection (protocol section 2: split 2 is the
     selection split), instead of a random 1/8 of the training rows.
  D6 the density-weight grid is extended from [0, 1.5] to cover the training-label range (the release's NoTZ scale
     reaches 1.79); labels are not clipped.
  D7 tensorised data access (same values as the authors' pandas Dataset, checked in tests), fixed validation batch
     composition and seeded shuffles (the authors shuffle validation batches with an unseeded RandomSampler).
Observation kept as in the authors' code: WeightedMSE multiplies a [B] weight vector with a [B, 1] squared error, which
broadcasts to B x B, so the term equals (batch-mean weight) x MSE. It is reproduced literally, not corrected.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
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
from torch import nn  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from bisurfgnn.comparators.ddprism import capture_attention, load_author_classes, verify_source  # noqa: E402

DATA = ROOT.parent / "bisurfgnn"                       # read-only main checkout (CellMiner, drug structures)
S1 = ROOT / "runs/model_improvement/public_split_gate_v1/s1"
CELL_CONFIG = DATA / "configs/cell_context.yaml"
DRUGS_CSV = DATA / "data/processed/nci_almanac_unclipped/drugs.csv"
PROTOCOL = ROOT / "docs/BISURFGNN_V2_DEV_PROTOCOL.md"
PROTOCOL_SHA256 = "155ae4f03ceb8275f68d10f9830f5bcd193c41f70564cc138bc175840788a333"
VAL_OFFSET = 10_000_000
ADAPTER_NAME = "DD-PRiSM-Hallmark-MeasuredMono (adapted; not the authors' original setting)"
COLUMNS = ["matrix_key", "surface_index", "pair_id", "deepsynba_cell", "biological_cell_id", "drug_a_id", "drug_b_id",
           "deepsynba_drug_a", "deepsynba_drug_b", "row_i", "col_j", "dose_a_M", "dose_b_M", "mono_a_percent",
           "mono_b_percent", "response_percent"]
DEVIATIONS = {
    "D1": "measured monotherapy = S1 split-1 lookup anchors (same as BiSurfGNN) instead of predicted viability",
    "D2": "50 MSigDB Hallmark gene sets, CellMiner gene z-scores fit on split-1 cells, instead of 186 KEGG sets",
    "D3": "monotherapy model trained from scratch on split-1 anchors with notebook-04 recipe; no NCI-60 pretraining "
          "or ALMANAC fine-tuning (notebooks 02/03 not cached; NCI-60 would add external knowledge)",
    "D4": "standard eval mode for validation loss and prediction; no BatchNorm refresh with evaluation inputs "
          "(authors' test_comb/predict_comb do a train-mode pass over evaluation batches first)",
    "D5": "split 2 drives LR schedule, early stopping and epoch selection (protocol section 2) instead of a random "
          "1/8 of training rows",
    "D6": "density-weight grid extended from [0, 1.5] to the training-label range; labels not clipped",
    "D7": "tensorised data access, fixed validation batch composition, seeded shuffles",
}
AUTHOR_NOTE = ("authors' WeightedMSE broadcasts a [B] weight vector against a [B,1] squared error (B x B), i.e. "
               "batch-mean weight x MSE; reproduced literally")
FORBIDDEN_PARTS = ("s5",)
FORBIDDEN_NAMES = ("S5_TEST_EVALUATED.marker", "test_predictions.npz", "final_predictions.npz", "test.parquet",
                   "points_test.parquet", "nci_synergy_dose_response.pickle", "points.parquet")
READ_LOG: list[str] = []


# ---------------------------------------------------------------------------------------------------------------
# access guard, hashing, seeding
# ---------------------------------------------------------------------------------------------------------------
def allowed(path: Path) -> Path:
    path = Path(path).resolve()
    if any(p in FORBIDDEN_PARTS for p in path.parts) or path.name in FORBIDDEN_NAMES or "split3" in path.name.lower():
        raise PermissionError(f"Refusing to open a test-side or full-response file: {path}")
    READ_LOG.append(str(path))
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with allowed(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value, mode: str = "x") -> None:
    with Path(path).open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=1, ensure_ascii=False, default=float) + "\n")


# ---------------------------------------------------------------------------------------------------------------
# data: S1 tables, monotherapy records, features
# ---------------------------------------------------------------------------------------------------------------
def load_tables() -> dict[str, pd.DataFrame]:
    manifest = json.loads(allowed(S1 / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for role in ("train", "validation"):
        path = S1 / f"points_{role}.parquet"
        if sha256(path) != manifest[path.name]:
            raise RuntimeError(f"S1 table changed since S1: {path.name}")
        out[role] = pd.read_parquet(allowed(path), columns=COLUMNS)
    return out


def entity_index(tables: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    tr, va = tables["train"], tables["validation"]
    drugs = sorted(set(tr.drug_a_id) | set(tr.drug_b_id))
    cells = sorted(set(tr.biological_cell_id))
    if not (set(va.drug_a_id) | set(va.drug_b_id)) <= set(drugs) or not set(va.biological_cell_id) <= set(cells):
        raise RuntimeError("Validation drugs/cells must all appear in split 1 (new-combination split)")
    return drugs, cells


def monotherapy_records(train: pd.DataFrame) -> pd.DataFrame:
    """Unique (drug, cell, dose) split-1 monotherapy anchors; each key must carry a single value."""
    rows = pd.concat([
        pd.DataFrame({"drug": train.drug_a_id, "cell": train.biological_cell_id, "dose_M": train.dose_a_M,
                      "percent": train.mono_a_percent}),
        pd.DataFrame({"drug": train.drug_b_id, "cell": train.biological_cell_id, "dose_M": train.dose_b_M,
                      "percent": train.mono_b_percent})], ignore_index=True)
    rows["dose_key"] = [f"{float(x):.6g}" for x in rows.dose_M]       # S1 lookup key rule
    g = rows.groupby(["drug", "cell", "dose_key"], sort=True).agg(
        dose_M=("dose_M", "first"), percent=("percent", "first"), pmin=("percent", "min"), pmax=("percent", "max"))
    if float((g.pmax - g.pmin).max()) > 1e-9:
        raise RuntimeError("Split-1 monotherapy anchors are not unique per (drug, cell, dose)")
    return g.drop(columns=["pmin", "pmax"]).reset_index()


def cell_gene_features(cells: list[str]):
    """Hallmark-grouped CellMiner gene z-scores, transform fit on the split-1 cells only (D2)."""
    from bisurfgnn.data.cellminer import (_gene_z_scores, fit_train_only_pathway_transform, load_cached_pathway_topology,
                                          load_cellminer_config, load_hallmark_gene_sets, load_prepared_expression)
    cfg = load_cellminer_config(allowed(CELL_CONFIG))
    expression = load_prepared_expression(cfg)
    catalog = load_hallmark_gene_sets(cfg)
    topology = load_cached_pathway_topology(cfg, expected_catalog_hash=catalog.catalog_hash)
    state = fit_train_only_pathway_transform(
        expression, catalog, topology, cells, "DDPRiSM-MeasuredMono-v1|PublicSplitGate-v1 S1 split-1 cells",
        min_pathway_genes=cfg.min_pathway_genes, clip_value=cfg.clip_value, std_epsilon=cfg.std_epsilon)
    idx = expression.cell_indices(cells)
    z = _gene_z_scores(expression.values[idx], expression.observed_mask[idx], state)
    gene_sets, tensors = {}, []
    for p, name in enumerate(state.pathway_ids):
        sel = torch.where(state.membership[p])[0]
        if len(sel) < 1:
            raise RuntimeError(f"Empty pathway: {name}")
        gene_sets[name] = [state.gene_ids[i] for i in sel.tolist()]
        tensors.append(z[:, sel].float().contiguous())
    info = dict(pathways=len(gene_sets), genes_per_pathway=[len(v) for v in gene_sets.values()],
                fit_cells=len(state.fit_cell_ids), state_hash=state.state_hash, expression_hash=expression.content_hash,
                catalog_hash=catalog.catalog_hash, topology_hash=topology.topology_hash)
    return gene_sets, tensors, info


def morgan512(drugs: list[str]) -> torch.Tensor:
    """512-bit Morgan (radius 2) fingerprints from the frozen canonical SMILES (same generator as the earlier pilot)."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    smiles = pd.read_csv(allowed(DRUGS_CSV), keep_default_na=False).set_index("drug_id").canonical_smiles
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    out = []
    for d in drugs:
        mol = Chem.MolFromSmiles(smiles[d])
        if mol is None:
            raise RuntimeError(f"Invalid frozen canonical SMILES for {d}")
        vec = np.zeros(512, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), vec)
        out.append(vec)
    return torch.tensor(np.stack(out))


def dd_concentration(dose_m) -> np.ndarray:
    """Authors' concentration feature: log10(molar concentration) + 6."""
    return np.log10(np.asarray(dose_m, float)) + 6.0


# ---------------------------------------------------------------------------------------------------------------
# authors' loss and schedule
# ---------------------------------------------------------------------------------------------------------------
def kde_density_grid(data: np.ndarray, grid: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """scipy.stats.gaussian_kde(data)(grid) for 1-D data (Scott's factor, unbiased variance), computed with numpy
    (scipy's BLAS path aborts next to torch's OpenMP runtime in this environment)."""
    data = np.asarray(data, np.float64)
    n = data.size
    cov = float(np.var(data, ddof=1)) * n ** (-2.0 / 5.0)
    out = np.zeros(len(grid))
    for lo in range(0, n, chunk):
        d = data[lo:lo + chunk]
        out += np.exp(-0.5 * (grid[:, None] - d[None, :]) ** 2 / cov).sum(axis=1)
    return out / n / np.sqrt(2.0 * np.pi * cov)


class DensityWeights:
    """Authors' estimate_density + get_weight (precision 2, MinMax-normalised density, weight = 1 - gamma * density)
    with the grid extended to cover the training labels (D6). Labels are rounded like np.around(float64(label), 2)."""

    def __init__(self, train_labels: np.ndarray, gamma: float, precision: int = 2):
        if precision != 2:
            raise ValueError("The authors hard-code precision 2")
        labels = np.asarray(train_labels, np.float64)
        top = max(1.5, float(np.ceil(labels.max() * 100.0) / 100.0))
        self.grid = np.around(np.arange(0, top + 0.001, 0.01), 2)
        dens = kde_density_grid(labels, self.grid)
        self.density = (dens - dens.min()) / (dens.max() - dens.min())
        self.gamma = float(gamma)
        self.grid_top = top

    def index(self, labels) -> np.ndarray:
        x = np.asarray(torch.as_tensor(labels, dtype=torch.float32).cpu().numpy(), np.float64)
        return np.clip(np.rint(x * 100.0).astype(np.int64), 0, len(self.grid) - 1)

    def weights(self, labels) -> np.ndarray:
        return (1.0 - self.gamma * self.density[self.index(labels)]).astype(np.float32)


def author_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, alpha: float, beta: float):
    """Authors' CustomLoss.forward with precomputed weights: alpha * WeightedMSE + beta * (1 - PCC).
    pred, target: [B, 1]; weight: [B] (the [B] x [B, 1] broadcast of the authors' code is kept)."""
    weighted_mse = torch.mean(weight * ((pred - target) ** 2))
    mse = torch.mean((pred - target) ** 2)
    vx, vy = pred - torch.mean(pred), target - torch.mean(target)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return alpha * weighted_mse + beta * (1 - corr), mse, corr


class AuthorSchedule:
    """Notebook 04 loop: improvement = val_loss < best - threshold; LR x factor after `patience_lr` epochs without
    improvement (counter reset), stop after `patience_es` (not reset by LR changes). Selection = plain argmin."""

    def __init__(self, lr: float, threshold: float, patience_lr: int, patience_es: int, factor: float):
        self.lr, self.threshold, self.factor = lr, threshold, factor
        self.max_lr_patience, self.max_es_patience = patience_lr, patience_es
        self.patience_lr, self.patience_es = patience_lr, patience_es
        self.best = float("inf")

    def step(self, val_loss: float) -> tuple[bool, bool]:
        if val_loss < self.best - self.threshold:
            self.best = val_loss
            self.patience_lr, self.patience_es = self.max_lr_patience, self.max_es_patience
        else:
            self.patience_lr -= 1
            self.patience_es -= 1
        lr_changed = False
        if self.patience_lr <= 0:
            self.patience_lr = self.max_lr_patience
            self.lr *= self.factor
            lr_changed = True
        return lr_changed, self.patience_es <= 0


# ---------------------------------------------------------------------------------------------------------------
# model pieces
# ---------------------------------------------------------------------------------------------------------------
def author_classes(n_pathways: int):
    return load_author_classes(ROOT, n_pathways)


def author_functions(names=("MinMaxNormalization", "estimate_density", "get_weight", "PCC", "MSE", "RMSE",
                            "WeightedMSE", "CustomLoss"), device: str = "cpu") -> dict:
    """Execute the inspected authors' helper definitions from 00_Utils.ipynb (tests / fidelity checks only)."""
    import scipy.stats  # noqa: F401  (the authors' estimate_density references scipy.stats)
    path = verify_source(ROOT, "ddprism", "00_Utils.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell.get("source", []))).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                nodes.append(node)
    if sorted(n.name for n in nodes) != sorted(names):
        raise RuntimeError("Expected each authors' helper exactly once")
    import scipy
    namespace = {"np": np, "torch": torch, "nn": nn, "scipy": scipy, "device": device}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "ddprism_author_utils", "exec"),
         namespace)
    return namespace


def model_device(cfg: dict) -> torch.device:
    return torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))


def mono_forward(mono, cell_tensors, fps, cell_idx, drug_idx, conc):
    return mono([[t[cell_idx] for t in cell_tensors], fps[drug_idx], conc])


def attention_table(mono, cell_tensors, fps, n_drugs: int, n_cells: int, device, batch: int = 1024) -> torch.Tensor:
    """Pathway attention per (drug, cell), row = drug * n_cells + cell, in eval mode (dose-independent)."""
    mono.eval()
    drug_idx = torch.arange(n_drugs, device=device).repeat_interleave(n_cells)
    cell_idx = torch.arange(n_cells, device=device).repeat(n_drugs)
    pieces = []
    for lo in range(0, len(drug_idx), batch):
        d, c = drug_idx[lo:lo + batch], cell_idx[lo:lo + batch]
        pieces.append(capture_attention(mono, [[t[c] for t in cell_tensors], fps[d],
                                               torch.zeros(len(d), 1, device=device)]))
    return torch.cat(pieces)


def batches(n: int, size: int, generator: torch.Generator | None):
    order = torch.arange(n) if generator is None else torch.randperm(n, generator=generator)
    return [order[lo:lo + size] for lo in range(0, n, size)]


# ---------------------------------------------------------------------------------------------------------------
# training stages
# ---------------------------------------------------------------------------------------------------------------
def train_monotherapy(Mono, gene_sets, cell_tensors, fps, records, cells, drugs, cfg, seed, device, deadline, log):
    """Reconstructed monotherapy stage (D3): notebook-04 recipe on split-1 monotherapy anchors, internal 7/8 vs 1/8
    random split of the anchor records for its own schedule and selection."""
    m = cfg["mono"]
    if abs(float(m["internal_val_fraction"]) - 0.125) > 1e-12:
        raise ValueError("The reconstructed monotherapy stage uses notebook 04's 7/8 vs 1/8 split")
    seed_everything(seed)
    model = Mono(gene_sets).to(device)
    n = len(records)
    perm = np.random.default_rng(seed).permutation(n)
    n_val = n - int(n * 7 / 8)
    val_rows, tr_rows = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    cell_pos = {c: i for i, c in enumerate(cells)}
    drug_pos = {d: i for i, d in enumerate(drugs)}
    ci = torch.tensor(records.cell.map(cell_pos).to_numpy(), dtype=torch.long, device=device)
    di = torch.tensor(records.drug.map(drug_pos).to_numpy(), dtype=torch.long, device=device)
    conc = torch.tensor(dd_concentration(records.dose_M), dtype=torch.float32, device=device).reshape(-1, 1)
    y64 = records.percent.to_numpy(np.float64) / 100.0
    y = torch.tensor(y64, dtype=torch.float32, device=device).reshape(-1, 1)
    dens = DensityWeights(y64[tr_rows], m["gamma"])                       # KDE on float64 labels, as the authors
    w = torch.tensor(dens.weights(y.cpu().numpy().ravel()), device=device)  # weights from float32 labels, as the authors
    opt = torch.optim.AdamW(model.parameters(), lr=m["learning_rate"])      # authors: AdamW defaults otherwise
    sched = AuthorSchedule(m["learning_rate"], m["threshold"], m["patience_lr"], m["patience_es"], m["lr_factor"])
    tr_t, va_t = torch.tensor(tr_rows, device=device), torch.tensor(val_rows, device=device)
    history, best_loss, best_state, best_epoch, stopped_by = [], float("inf"), None, -1, "max_epochs"
    for epoch in range(m["max_epochs"]):
        if time.monotonic() > deadline:
            stopped_by = "wall_limit"; break
        t0 = time.perf_counter()
        model.train()
        g = torch.Generator().manual_seed(seed * 100_003 + epoch)
        losses = []
        for b in batches(len(tr_t), m["batch_size"], g):
            idx = tr_t[b.to(device)]
            if len(idx) < 2:
                continue                                          # BatchNorm needs >= 2 rows in train mode
            pred = mono_forward(model, cell_tensors, fps, ci[idx], di[idx], conc[idx])
            loss, _, _ = author_loss(pred, y[idx], w[idx], m["alpha"], m["beta"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite monotherapy loss")
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vl, se = [], 0.0
            for b in batches(len(va_t), m["batch_size"], None):
                idx = va_t[b.to(device)]
                pred = mono_forward(model, cell_tensors, fps, ci[idx], di[idx], conc[idx])
                vl.append(float(author_loss(pred, y[idx], w[idx], m["alpha"], m["beta"])[0]))
                se += float(((pred - y[idx]) ** 2).sum())
        val_loss = float(np.mean(vl))
        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch
        lr_used = sched.lr
        lr_changed, stop = sched.step(val_loss)
        if lr_changed:
            opt.param_groups[0]["lr"] = sched.lr
        sec = time.perf_counter() - t0
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=val_loss,
                            val_rmse_percent=100.0 * float(np.sqrt(se / len(va_t))), lr=lr_used, seconds=round(sec, 3)))
        log(f"mono epoch {epoch} train_loss {np.mean(losses):.5f} val_loss {val_loss:.5f} "
            f"val_rmse% {history[-1]['val_rmse_percent']:.3f} lr {lr_used:g} {sec:.1f}s")
        if stop:
            stopped_by = "early_stopping"; break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(history=history, best_epoch=best_epoch, best_val_loss=best_loss, stopped_by=stopped_by,
                       records=n, internal_train=len(tr_rows), internal_val=len(val_rows),
                       density_grid_top=dens.grid_top)


class CombinationData:
    """Tensorised version of the authors' CombinationDataset: [attention1, attention2, viability1, viability2]."""

    def __init__(self, frame: pd.DataFrame, drugs: list[str], cells: list[str], device, swap: bool):
        drug_pos = {d: i for i, d in enumerate(drugs)}
        cell_pos = {c: i for i, c in enumerate(cells)}
        n_cells = len(cells)
        c = frame.biological_cell_id.map(cell_pos).to_numpy()
        a = frame.drug_a_id.map(drug_pos).to_numpy() * n_cells + c
        b = frame.drug_b_id.map(drug_pos).to_numpy() * n_cells + c
        va = frame.mono_a_percent.to_numpy(np.float64) / 100.0
        vb = frame.mono_b_percent.to_numpy(np.float64) / 100.0
        y = frame.response_percent.to_numpy(np.float64) / 100.0
        if swap:                                              # authors: df_Train + copy with drug order swapped
            a, b, va, vb, y = np.r_[a, b], np.r_[b, a], np.r_[va, vb], np.r_[vb, va], np.r_[y, y]
        self.y64 = y                                          # float64 labels (the authors' KDE input)
        self.row_a = torch.tensor(a, dtype=torch.long, device=device)
        self.row_b = torch.tensor(b, dtype=torch.long, device=device)
        self.v_a = torch.tensor(va, dtype=torch.float32, device=device).reshape(-1, 1)
        self.v_b = torch.tensor(vb, dtype=torch.float32, device=device).reshape(-1, 1)
        self.y = torch.tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)
        self.n = len(y)

    def features(self, attention: torch.Tensor, idx: torch.Tensor):
        return [attention[self.row_a[idx]], attention[self.row_b[idx]], self.v_a[idx], self.v_b[idx]]


def evaluate_combination(combo, attention, data: CombinationData, weights, cfg, order: torch.Tensor):
    """Eval mode only (D4). Returns authors' mean batch loss over a fixed batch composition and the predictions."""
    c = cfg["combo"]
    combo.eval()
    pred = torch.empty_like(data.y)
    losses = []
    with torch.no_grad():
        for lo in range(0, data.n, c["batch_size"]):
            idx = order[lo:lo + c["batch_size"]]
            out = combo(data.features(attention, idx))[2]
            pred[idx] = out
            losses.append(float(author_loss(out, data.y[idx], weights[idx], c["alpha"], c["beta"])[0]))
    return float(np.mean(losses)), pred


def split2_metrics(pred_ratio: np.ndarray, frame: pd.DataFrame, codes: np.ndarray | None = None) -> dict:
    y = frame.response_percent.to_numpy(np.float64)
    p = 100.0 * np.asarray(pred_ratio, np.float64)
    e = p - y
    codes = pd.factorize(frame.matrix_key)[0] if codes is None else codes
    per = np.sqrt(np.bincount(codes, weights=e * e) / np.bincount(codes))
    hsa = np.minimum(frame.mono_a_percent.to_numpy(np.float64), frame.mono_b_percent.to_numpy(np.float64))
    return dict(combination_rmse_percent=float(np.sqrt(np.mean(e * e))), surface_macro_rmse_percent=float(per.mean()),
                pearson=float(np.corrcoef(p, y)[0, 1]), bias_percent=float(e.mean()),
                lookup_hsa_rmse_percent=float(np.sqrt(np.mean((hsa - y) ** 2))), points=int(len(y)),
                matrices=int(codes.max() + 1))


def train_combination(Combo, attention, train: CombinationData, val: CombinationData, val_frame, cfg, seed, device,
                      deadline, log, out_dir: Path | None):
    c = cfg["combo"]
    seed_everything(seed)
    combo = Combo().to(device)
    dens = DensityWeights(train.y64, c["gamma"])                     # authors: KDE of the swapped training labels
    w_tr = torch.tensor(dens.weights(train.y.cpu().numpy().ravel()), device=device)
    w_va = torch.tensor(dens.weights(val.y.cpu().numpy().ravel()), device=device)
    codes = pd.factorize(val_frame.matrix_key)[0]
    opt = torch.optim.AdamW(combo.parameters(), lr=c["learning_rate"])
    sched = AuthorSchedule(c["learning_rate"], c["threshold"], c["patience_lr"], c["patience_es"], c["lr_factor"])
    val_order = torch.randperm(val.n, generator=torch.Generator().manual_seed(seed * 7 + 1)).to(device)   # D7
    history, best_loss, best_state, best_epoch, stopped_by = [], float("inf"), None, -1, "max_epochs"
    for epoch in range(c["max_epochs"]):
        if time.monotonic() > deadline:
            stopped_by = "wall_limit"; break
        t0 = time.perf_counter()
        combo.train()
        g = torch.Generator().manual_seed(seed * 100_003 + epoch)
        order = torch.randperm(train.n, generator=g).to(device)
        losses = []
        for lo in range(0, train.n, c["batch_size"]):
            idx = order[lo:lo + c["batch_size"]]
            if len(idx) < 2:
                continue
            _, _, pred = combo(train.features(attention, idx))
            loss, _, _ = author_loss(pred, train.y[idx], w_tr[idx], c["alpha"], c["beta"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite combination loss")
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.detach())
        train_loss = float(torch.stack(losses).mean())
        t_train = time.perf_counter() - t0
        val_loss, pred = evaluate_combination(combo, attention, val, w_va, cfg, val_order)
        metrics = split2_metrics(pred.cpu().numpy().ravel(), val_frame, codes)
        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, copy.deepcopy(combo.state_dict()), epoch
        lr_used = sched.lr
        lr_changed, stop = sched.step(val_loss)
        if lr_changed:
            opt.param_groups[0]["lr"] = sched.lr
        sec = time.perf_counter() - t0
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, lr=lr_used,
                            val_combination_rmse_percent=metrics["combination_rmse_percent"],
                            val_surface_macro_rmse_percent=metrics["surface_macro_rmse_percent"],
                            val_pearson=metrics["pearson"], train_seconds=round(t_train, 3), seconds=round(sec, 3)))
        log(f"combo epoch {epoch} train_loss {train_loss:.5f} val_loss {val_loss:.5f} "
            f"val_rmse% {metrics['combination_rmse_percent']:.3f} macro% {metrics['surface_macro_rmse_percent']:.3f} "
            f"lr {lr_used:g} {sec:.1f}s")
        if out_dir is not None:
            write_json(out_dir / "combination_history.json", dict(history=history, best_epoch=best_epoch), mode="w")
        if stop:
            stopped_by = "early_stopping"; break
    combo.load_state_dict(best_state)
    return combo, dict(history=history, best_epoch=best_epoch, best_val_loss=best_loss, stopped_by=stopped_by,
                       density_grid_top=dens.grid_top, train_rows_with_swap=train.n), w_va, val_order


# ---------------------------------------------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------------------------------------------
def source_hashes() -> dict:
    files = [Path(__file__), HERE / "run_smoke.py", HERE / "launch_validation.py", HERE / "formal_config.json",
             HERE / "smoke_config.json", ROOT / "src/bisurfgnn/comparators/ddprism.py"]
    out = {str(p.relative_to(ROOT)): sha256(p) for p in files if p.exists()}
    for name in ("00_MonotherapyUtils.ipynb", "00_CombinationtherapyUtils.ipynb", "00_Utils.ipynb",
                 "04_CombinationTherapyModel_Training.ipynb"):
        out[f"author:{name}"] = sha256(verify_source(ROOT, "ddprism", name))
    return out


def run_pipeline(cfg: dict, out_dir: Path, *, save_predictions: bool, log) -> dict:
    """Train the monotherapy stage, extract attention, train the combination network, evaluate on split 2 only."""
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("Frozen development protocol changed")
    t_start = time.perf_counter()
    deadline = time.monotonic() + 3600.0 * float(cfg["max_wall_hours"])
    device = model_device(cfg)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seed = int(cfg["seed"])
    tables = load_tables()
    drugs, cells = entity_index(tables)
    records = monotherapy_records(tables["train"])
    gene_sets, cell_tensors, cell_info = cell_gene_features(cells)
    fps = morgan512(drugs).to(device)
    cell_tensors = [t.to(device) for t in cell_tensors]
    Mono, Combo = author_classes(len(gene_sets))
    t0 = time.perf_counter()
    mono, mono_info = train_monotherapy(Mono, gene_sets, cell_tensors, fps, records, cells, drugs, cfg, seed, device,
                                        deadline, log)
    mono_seconds = time.perf_counter() - t0
    attention = attention_table(mono, cell_tensors, fps, len(drugs), len(cells), device)
    train = CombinationData(tables["train"], drugs, cells, device, swap=cfg["combo"]["swap_augment"])
    val = CombinationData(tables["validation"], drugs, cells, device, swap=False)
    t0 = time.perf_counter()
    combo, combo_info, w_va, val_order = train_combination(Combo, attention, train, val, tables["validation"], cfg,
                                                           seed, device, deadline, log, out_dir)
    combo_seconds = time.perf_counter() - t0
    val_loss, pred = evaluate_combination(combo, attention, val, w_va, cfg, val_order)
    if abs(val_loss - combo_info["best_val_loss"]) > 1e-6:
        raise RuntimeError(f"Selected state does not reproduce its validation loss ({val_loss} vs "
                           f"{combo_info['best_val_loss']})")
    pred_np = pred.cpu().numpy().ravel()
    metrics = split2_metrics(pred_np, tables["validation"])
    # reload check: saved state reproduces the predictions exactly
    state = {"mono": mono.state_dict(), "combo": combo.state_dict(), "attention": attention.detach().cpu()}
    if save_predictions:
        torch.save(state, out_dir / "model_selected.pt")
        saved = torch.load(out_dir / "model_selected.pt", map_location=device, weights_only=True)
        Combo2 = Combo().to(device); Combo2.load_state_dict(saved["combo"])
        _, pred2 = evaluate_combination(Combo2, saved["attention"].to(device), val, w_va, cfg, val_order)
        if not torch.equal(pred2, pred):
            raise RuntimeError("Reloaded model does not reproduce the validation predictions")
        frame = tables["validation"]
        pd.DataFrame({"source_row": VAL_OFFSET + np.arange(len(frame)), "surface_id": frame.matrix_key,
                      "target": frame.response_percent.to_numpy(np.float64) / 100.0, "prediction": pred_np}
                     ).to_csv(out_dir / "validation_predictions.csv", index=False)
    combo_epochs = combo_info["history"]
    mono_epochs = mono_info["history"]
    result = dict(
        adapter=ADAPTER_NAME, seed=seed, split_read=[1, 2], test_rows_read=0, evaluation="split 2 (validation) only",
        selected_epoch=combo_info["best_epoch"], selected_val_loss=val_loss, stopped_by=combo_info["stopped_by"],
        validation=metrics, mono=dict(best_epoch=mono_info["best_epoch"], stopped_by=mono_info["stopped_by"],
                                      epochs=len(mono_epochs), records=mono_info["records"],
                                      seconds=round(mono_seconds, 1),
                                      mean_epoch_seconds=float(np.mean([h["seconds"] for h in mono_epochs]))),
        combination=dict(epochs=len(combo_epochs), seconds=round(combo_seconds, 1),
                         mean_epoch_seconds=float(np.mean([h["seconds"] for h in combo_epochs])),
                         mean_train_seconds=float(np.mean([h["train_seconds"] for h in combo_epochs])),
                         train_rows_with_swap=train.n, validation_rows=val.n),
        cell_features=cell_info, drugs=len(drugs), cells=len(cells),
        parameters=dict(mono=sum(p.numel() for p in mono.parameters()), combo=sum(p.numel() for p in combo.parameters())),
        gpu=dict(device=str(device), name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                 peak_allocated_mb=round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1) if device.type == "cuda" else None,
                 peak_reserved_mb=round(torch.cuda.max_memory_reserved(device) / 2 ** 20, 1) if device.type == "cuda" else None),
        wall_seconds=round(time.perf_counter() - t_start, 1), deviations=DEVIATIONS, author_code_note=AUTHOR_NOTE,
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                      pandas=pd.__version__))
    write_json(out_dir / "monotherapy_history.json", mono_info)
    write_json(out_dir / "combination_history.json", combo_info, mode="w")
    return result

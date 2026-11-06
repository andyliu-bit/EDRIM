"""NumericSurface-v1 runner: numeric-only candidates and the staged numeric + graph-surface model.

Data path, point IDs/order, input hashes, test-response guard, primary/repeat
determinism and result tables come unchanged from `run_comparator_pilot.execute`.
Protocol: docs/NUMERIC_SURFACE_V1_PROTOCOL.md.

Model names
  bisurf_num_<arch>_<opt>              numeric-only (arch: ple | tabm)
  bisurf_numsurf_<arch>_<opt>[_cf]     numeric + graph surface corrector
Both are routed through pilot.execute's "bisurf" (per-seed) branch.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits

from bisurfgnn import independent_graph_benchmark as graph
from bisurfgnn import learned_benchmark as learned
from bisurfgnn.models.numeric_surface import (GraphSurfaceCorrector, LowDimSurfaceCorrector, NumericStats,
                                             NumericSurfaceModel, PLEResidualMLP, TabMNumeric, boundary_gate)

spec = importlib.util.spec_from_file_location("ns_search", Path(__file__).with_name("run_algorithm_search.py"))
search = importlib.util.module_from_spec(spec); spec.loader.exec_module(search)
pilot = search.pilot
NUM = re.compile(r"bisurf_num_(ple|tabm)_(o[12])")
FULL = re.compile(r"bisurf_numsurf_(ple|tabm)_(o[12])(_cf)?(_lowdim)?")


# ----------------------------------------------------------------------------- data
class Data:
    """Point tensors for one role, in the frozen table order."""

    def __init__(self, table, device):
        self.mono_a = table.mono_a.to(device); self.mono_b = table.mono_b.to(device)
        self.doses = table.doses.to(device); self.target = table.target.to(device)
        self.a = table.drug_a_index.long().to(device); self.b = table.drug_b_index.long().to(device)
        self.c = table.cell_index.long().to(device)
        self.surface_ids = np.asarray(table.surface_ids)
        codes, sizes = np.unique(self.surface_ids, return_inverse=True, return_counts=True)[1:]
        self.point_weight = torch.as_tensor(1.0 / sizes[codes], dtype=torch.float32, device=device)
        self.table = table

    def __len__(self):
        return len(self.surface_ids)


def surface_loss(pred, target, weight):
    """Surface-equal MSE for batches made of complete surfaces (weights sum to #surfaces)."""
    return torch.sum(weight * (pred - target) ** 2) / torch.sum(weight)


def metric(data, pred):
    return pilot.metric(data.target.cpu().numpy(), np.asarray(pred), data.surface_ids)["surface_macro_rmse"]


def build_numeric(arch, stats, cfg):
    spec_ = cfg["numeric_arch"][arch]
    return PLEResidualMLP(stats, **spec_) if arch == "ple" else TabMNumeric(stats, **spec_)


def fit_stats(train: Data):
    t = lambda x: x.detach().cpu().float()
    return NumericStats.fit(t(train.mono_a), t(train.mono_b), t(train.doses[:, 0]), t(train.doses[:, 1]), t(train.target))


def batches(data: Data, batch_surfaces: int, seed: int):
    return graph._surface_batch_indices(data.table, batch_surfaces, shuffle_seed=seed)


def member_predictions(model, data: Data, idx, banks_fn=None):
    """[B] or [B, k] predictions; TabM members keep their own losses during training."""
    i = idx.to(data.mono_a.device)
    ma, mb, d = data.mono_a[i], data.mono_b[i], data.doses[i]
    r_num = model.numeric.members(ma, mb, d) if isinstance(model.numeric, TabMNumeric) and model.training else \
        model.numeric(ma, mb, d)
    if model.corrector is not None:
        drugs, cells, a, b, c = banks_fn(idx)
        r_graph = model.corrector.forward_banks(drugs, cells, a, b, c, d)
        r_num = r_num + (r_graph[:, None] if r_num.ndim == 2 else r_graph)
    anchor, gate = torch.minimum(ma, mb), boundary_gate(d)
    return (anchor[:, None] + gate[:, None] * r_num) if r_num.ndim == 2 else anchor + gate * r_num


def batch_loss(pred, data: Data, idx):
    i = idx.to(data.mono_a.device); y, w = data.target[i], data.point_weight[i]
    if pred.ndim == 2:  # TabM: mean of member losses
        return torch.stack([surface_loss(pred[:, j], y, w) for j in range(pred.shape[1])]).mean()
    return surface_loss(pred, y, w)


@torch.no_grad()
def predict_full(model, data: Data, banks, device, offset=None):
    model.eval()
    drugs = cells = None
    if model.corrector is not None:
        drugs, cells = model.corrector.encode(graph.pad_graph_items(banks.drug_graphs).to(device),
                                              graph.pad_graph_items(banks.cell_graphs).to(device))
    out = []
    for s in range(0, len(data), 4096):
        i = torch.arange(s, min(s + 4096, len(data)), device=device)
        r = model.numeric(data.mono_a[i], data.mono_b[i], data.doses[i]) if offset is None else offset[i]
        if model.corrector is not None:
            r = r + model.corrector.forward_banks(drugs, cells, data.a[i], data.b[i], data.c[i], data.doses[i])
        out.append((torch.minimum(data.mono_a[i], data.mono_b[i]) + boundary_gate(data.doses[i]) * r).cpu())
    return torch.cat(out).numpy()


def batch_banks_fn(model, data: Data, banks, device):
    if isinstance(model.corrector, LowDimSurfaceCorrector):  # fixed codes indexed by global bank ids
        def fixed(idx):
            i = idx.to(device)
            drugs, cells = model.corrector.encode()
            return drugs, cells, data.a[i], data.b[i], data.c[i]
        return fixed
    def fn(idx):
        idx_cpu = idx.cpu()
        a, b = data.table.drug_a_index[idx_cpu].long(), data.table.drug_b_index[idx_cpu].long()
        drug_graphs, inv = graph._unique_graph_batch(torch.cat([a, b]), banks.drug_graphs, device)
        cell_graphs, cinv = graph._unique_graph_batch(data.table.cell_index[idx_cpu].long(), banks.cell_graphs, device)
        drugs, cells = model.corrector.encode(drug_graphs, cell_graphs)
        n = len(idx_cpu)
        return drugs, cells, inv[:n], inv[n:], cinv
    return fn


# ----------------------------------------------------------------------------- optimisation loop
def optimise(model, params, train, val, opt, steps, seed, deadline, history, tag, banks=None, device=None,
             offset_train=None, best=None, log_train=True):
    """Constant-lr AdamW over complete-surface batches; validation every eval_every steps."""
    optimizer = torch.optim.AdamW(params, lr=opt["lr"], weight_decay=opt["weight_decay"])
    if opt.get("schedule", "constant") == "cosine":  # decays to 0 at the frozen step budget
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    else:
        scheduler = None
    fn = batch_banks_fn(model, train, banks, device) if model.corrector is not None else None
    step, epoch = 0, 0
    while step < steps:
        for idx in batches(train, opt["batch_surfaces"], seed * 1000003 + epoch + 7919 * len(history)):
            if step >= steps:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Frozen NumericSurface wall budget exceeded")
            model.train()
            if offset_train is not None:
                model.numeric.eval()
            optimizer.zero_grad(set_to_none=True)
            if offset_train is not None:  # stage 2 with cross-fitted numeric residuals
                i = idx.to(device); d = train.doses[i]
                drugs, cells, a, b, c = fn(idx)
                r = offset_train[i] + model.corrector.forward_banks(drugs, cells, a, b, c, d)
                pred = torch.minimum(train.mono_a[i], train.mono_b[i]) + boundary_gate(d) * r
            else:
                pred = member_predictions(model, train, idx, fn)
            loss = batch_loss(pred, train, idx)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite NumericSurface loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, opt["gradient_clip"])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite NumericSurface gradient")
            optimizer.step(); step += 1
            if scheduler is not None:
                scheduler.step()
            if step % opt["eval_every"] == 0 or step == steps:
                score = metric(val, predict_full(model, val, banks, device))
                row = dict(stage=tag, step=step, val_rmse=score, grad_norm=float(norm))
                if log_train:
                    row["train_eval_rmse"] = metric(train, predict_full(model, train, banks, device))
                history.append(row)
                if score < best["score"] - 1e-5:
                    best.update(score=score, state=copy.deepcopy(model.state_dict()), where=f"{tag}:{step}")
        epoch += 1
    return best


def crossfit_numeric_offsets(arch, opt, train: Data, stats, cfg, seed, folds, deadline, device):
    """Out-of-fold numeric residuals on training points, grouped by prediction group (pair x cell)."""
    groups = np.asarray(train.table.group_ids)
    unique = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    assignment = dict(zip(unique, rng.permutation(len(unique)) % folds))
    fold_of = np.array([assignment[g] for g in groups])
    offsets = torch.zeros(len(train), device=device)
    for k in range(folds):
        rows = np.where(fold_of != k)[0]; held = np.where(fold_of == k)[0]
        sub = Data(search_sub(train.table, rows), device)
        learned._seed_everything(seed + 101 * (k + 1))
        model = NumericSurfaceModel(build_numeric(arch, stats, cfg)).to(device)
        best = dict(score=np.inf, state=copy.deepcopy(model.state_dict()), where="init")
        optimise(model, list(model.parameters()), sub, sub, opt, opt["steps"], seed + k, deadline, [], f"cf{k}",
                 device=device, best=best, log_train=False)
        model.load_state_dict(best["state"]); model.eval()
        h = torch.as_tensor(held, device=device)
        with torch.no_grad():
            offsets[h] = model.numeric(train.mono_a[h], train.mono_b[h], train.doses[h])
    return offsets


def search_sub(table, rows):
    rows_t = torch.as_tensor(rows)
    kw = {}
    for field in table.__dataclass_fields__:
        v = getattr(table, field)
        if isinstance(v, torch.Tensor) and v.shape[:1] == (len(table),):
            kw[field] = v[rows_t]
        elif isinstance(v, np.ndarray) and v.shape[:1] == (len(table),):
            kw[field] = v[rows]
        else:
            kw[field] = v
    return type(table)(**kw)


def lowdim_codes(banks, train: Data, cfg):
    """Full-XGBoost information: Morgan-512 PCA-16 (training drugs) and cell pathway PCA-16 (training cells)."""
    from sklearn.decomposition import PCA
    fp = pilot.morgan512((ROOT / cfg["data_root"]).resolve(), banks).numpy().astype(float)
    train_drugs = sorted(set(train.a.cpu().tolist()) | set(train.b.cpu().tolist()))
    drug = PCA(16, random_state=0).fit(fp[train_drugs]).transform(fp)
    cell_vec = np.stack([g.node_features[:, 0].numpy() for g in banks.cell_graphs]).astype(float)
    cell = PCA(16, random_state=0).fit(cell_vec[sorted(set(train.c.cpu().tolist()))]).transform(cell_vec)
    return torch.as_tensor(drug, dtype=torch.float32), torch.as_tensor(cell, dtype=torch.float32)


def build_corrector(full, banks, stats, cfg, train):
    if full is None:
        return None
    if full.group(4):
        drug, cell = lowdim_codes(banks, train, cfg)
        return LowDimSurfaceCorrector(drug, cell, stats, dropout=cfg["graph"]["dropout"])
    return GraphSurfaceCorrector(banks.atom_feature_dim, banks.cell_feature_dim, stats, **cfg["graph"])


def run_model(name, tables, banks, cfg, seed, out, deadline):
    device = torch.device(cfg["device"])
    train, val = Data(tables["train"], device), Data(tables["validation"], device)
    full = FULL.fullmatch(name); num = NUM.fullmatch(name)
    arch, opt_key = (full or num).group(1), (full or num).group(2)
    stats = fit_stats(train)
    history, info = [], {"numeric_stats": dict(residual_scale=stats.residual_scale, dose_knots=stats.dose_knots,
                                                arm_bins=[len(b) for b in stats.arm_bins])}
    learned._seed_everything(seed)
    numeric = build_numeric(arch, stats, cfg)
    corrector = build_corrector(full, banks, stats, cfg, train)
    model = NumericSurfaceModel(numeric, corrector).to(device)
    np.testing.assert_allclose(predict_full(model, val, banks, device), np.minimum(val.mono_a.cpu(), val.mono_b.cpu()), atol=1e-6)
    best = dict(score=metric(val, predict_full(model, val, banks, device)), state=copy.deepcopy(model.state_dict()), where="init")
    history.append(dict(stage="init", step=0, val_rmse=best["score"]))
    if num:
        opt = cfg["numeric_opt"][arch][opt_key]
        optimise(model, list(model.parameters()), train, val, opt, opt["steps"], seed, deadline, history, "numeric",
                 banks, device, best=best)
    else:
        # Stage 1: reuse the frozen numeric-only checkpoint of the same seed and verify it bit-for-bit.
        # relative paths resolve inside the running code snapshot
        src = ROOT / cfg["numeric_source_runs"][f"{arch}_{opt_key}"] / f"bisurf_num_{arch}_{opt_key}_seed_{seed}_primary"
        state = torch.load(src / "model.pt", weights_only=True, map_location="cpu")
        model.numeric.load_state_dict({k[len("numeric."):]: v for k, v in state.items() if k.startswith("numeric.")})
        saved = pd.read_csv(src / "predictions.csv").prediction.to_numpy()
        reproduced = predict_full(NumericSurfaceModel(model.numeric).to(device), val, None, device)
        if np.max(np.abs(reproduced - saved)) > 1e-6:
            raise RuntimeError("Stage-1 numeric checkpoint does not reproduce its saved predictions")
        info["stage1_source"] = learned._file_record(src / "model.pt")
        info["stage1_train_rmse"] = metric(train, predict_full(NumericSurfaceModel(model.numeric).to(device), train, None, device))
        info["stage1_val_rmse"] = metric(val, reproduced)
        best = dict(score=info["stage1_val_rmse"], state=copy.deepcopy(model.state_dict()), where="stage1")
        history.append(dict(stage="stage1", step=0, val_rmse=best["score"], train_eval_rmse=info["stage1_train_rmse"]))
        offsets = None
        if full.group(3):
            offsets = crossfit_numeric_offsets(arch, cfg["numeric_opt"][arch][opt_key], train, stats, cfg, seed,
                                               cfg["stage2"]["crossfit_folds"], deadline, device)
        for p in model.numeric.parameters():
            p.requires_grad_(False)
        s2 = cfg["stage2"]
        optimise(model, list(model.corrector.parameters()), train, val, s2, s2["steps"], seed, deadline, history,
                 "stage2", banks, device, offset_train=offsets, best=best)
        for p in model.numeric.parameters():
            p.requires_grad_(True)
        s3 = cfg["stage3"]
        optimise(model, list(model.parameters()), train, val, s3, s3["steps"], seed + 1, deadline, history,
                 "stage3", banks, device, best=best)
    model.load_state_dict(best["state"])
    prediction = predict_full(model, val, banks, device)
    torch.save(best["state"], out / "model.pt")
    clone = NumericSurfaceModel(build_numeric(arch, stats, cfg), build_corrector(full, banks, stats, cfg, train)).to(device)
    clone.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(predict_full(clone, val, banks, device), prediction, atol=0, rtol=0)
    info["selected"] = best["where"]
    info["train_eval_rmse_selected"] = metric(train, predict_full(model, train, banks, device))
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pilot.write_history(out, history, selected=best["where"])
    return prediction, {"best_epoch": best["where"], "history": history, **info,
                        "parameters": sum(p.numel() for p in model.parameters())}


def train_graph_route(tables, banks, cfg, seed, mono, out, deadline):
    del mono
    name = out.name.split("_seed_")[0]
    if not (FULL.fullmatch(name) or NUM.fullmatch(name)):
        raise ValueError(f"Unregistered NumericSurface model {name}")
    search.check_reference(out, cfg, banks)
    return run_model(name, tables, banks, cfg, seed, out, deadline)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); config_path = args.config.resolve(); output = args.output_dir.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not all(NUM.fullmatch(m) or FULL.fullmatch(m) for m in cfg["models"]) or cfg["fold"] != 0 \
            or cfg["claim_scope"] != "engineering_pilot_no_performance_ranking":
        raise ValueError("Unregistered or non-development NumericSurface configuration")
    allowed = ROOT / "runs/model_improvement"
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Output must be a fresh experiment subdirectory")
    output.mkdir(parents=True, exist_ok=False)
    reference = ROOT / cfg["reference_run"]
    sources = [learned._file_record(reference / n) for n in ("train_ids.csv", "validation_ids.csv", "data_binding.json", "summary.json")]
    learned._write_json_atomic(output / "reference_binding.json", {"sources": sources, "written_before_response_loading": True})
    accesses = []; original = pd.read_parquet
    def guard(path, *a, **k):
        resolved = Path(path).resolve()
        if resolved.name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError(f"Forbidden response source: {resolved}")
        accesses.append(str(resolved)); return original(path, *a, **k)
    try:
        with threadpool_limits(limits=cfg["threads"]), patch.object(pd, "read_parquet", guard), \
                patch.object(pilot, "train_graph", train_graph_route):
            result = pilot.execute(output, cfg, config_path)
        for record in sources:
            if learned._file_record(Path(record["path"]))["sha256"] != record["sha256"]:
                raise RuntimeError("Reference run drifted during training")
        search.assess(output, cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    except Exception as error:
        learned._write_json_atomic(output / "failure.json", {"status": "failed", "error": repr(error)})
        learned._write_json_atomic(output / "progress.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        learned._write_json_atomic(output / "input_access.json", {"parquet_reads": accesses, "test_response_rows_read": 0})


if __name__ == "__main__":
    main()

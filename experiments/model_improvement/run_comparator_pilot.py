"""Frozen, validation-only development runs with explicit comparator adaptations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.optimize import least_squares
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

from bisurfgnn import independent_graph_benchmark as graph
from bisurfgnn import learned_benchmark as learned
from bisurfgnn.benchmark_config import load_benchmark_config
from bisurfgnn.comparators.combokr2 import load_author_module, bind_small_kernel_backend
from bisurfgnn.comparators.ddprism import load_author_classes, capture_attention
from bisurfgnn.data.benchmark_context import load_benchmark_cell_context
from bisurfgnn.data.cellminer import load_cellminer_config, _gene_z_scores
from bisurfgnn.data.chemical_splits import load_one_new_scaffold_benchmark
from bisurfgnn.models.compact_bisurf import CompactBiSurfGNN

CONFIG = ROOT / "configs/model_improvement/comparator_pilot.yaml"
RESULT_COLUMNS = ["run", "model", "seed", "best_epoch", "surface_macro_rmse",
                  "point_rmse", "mae", "above_one_rmse", "wall_seconds", "status"]


def sample_complete_groups(frame, maximum, seed, salt):
    sizes = frame.groupby("prediction_group_id", sort=True).size()
    order = sorted(sizes.index, key=lambda x: hashlib.sha256(f"{salt}|{seed}|{x}".encode()).hexdigest())
    chosen, total = [], 0
    for key in order:
        if total + int(sizes[key]) <= maximum:
            chosen.append(key)
            total += int(sizes[key])
    if not chosen:
        raise ValueError("No complete group fits the point budget")
    return frame.loc[frame.prediction_group_id.isin(chosen)].copy().reset_index(drop=True)


def sample_supported_frames(train, validation, surfaces, cfg):
    """Select complete groups by IDs, covering validation partner/cell contexts."""
    va = sample_complete_groups(validation, cfg["validation_max_points"],
                                cfg["sampling_seed"], cfg["sampling_salt"])
    metadata = surfaces[surfaces.surface_replicate_id.isin(va.surface_replicate_id)]
    required = set(zip(metadata.cell_id, metadata.partner_drug_id))
    sizes = train.groupby("prediction_group_id").size()
    order = sorted(sizes.index, key=lambda x: hashlib.sha256(
        f"{cfg['sampling_salt']}|{cfg['sampling_seed']}|{x}".encode()).hexdigest())
    contexts = {}
    for row in train[["prediction_group_id", "cell_id", "drug_a_id", "drug_b_id"]].drop_duplicates().itertuples(index=False):
        contexts.setdefault(row.prediction_group_id, set()).update(
            ((row.cell_id, row.drug_a_id), (row.cell_id, row.drug_b_id)))
    selected, total, missing = set(), 0, required.copy()
    for key in order:
        if missing & contexts[key] and total + int(sizes[key]) <= cfg["train_max_points"]:
            selected.add(key); total += int(sizes[key]); missing -= contexts[key]
    if missing:
        raise ValueError(f"Frozen training budget cannot cover {len(missing)} partner/cell contexts")
    for key in order:
        if key not in selected and total + int(sizes[key]) <= cfg["train_max_points"]:
            selected.add(key); total += int(sizes[key])
    tr = train[train.prediction_group_id.isin(selected)].copy().reset_index(drop=True)
    return {"train": tr, "validation": va}


def audit_sample(frames, surfaces):
    tr, va = frames["train"], frames["validation"]
    support = set(zip(tr.cell_id, tr.drug_a_id)) | set(zip(tr.cell_id, tr.drug_b_id))
    meta = {r: surfaces[surfaces.surface_replicate_id.isin(f.surface_replicate_id)]
            for r, f in frames.items()}
    required = set(zip(meta["validation"].cell_id, meta["validation"].partner_drug_id))
    training_clusters = set(meta["train"].heldout_cluster_id) | set(meta["train"].partner_cluster_id)
    checks = {"partner_cell_contexts_seen_in_sampled_training": required <= support,
        "validation_cells_seen_in_sampled_training": set(va.cell_id) <= set(tr.cell_id),
        "validation_heldout_clusters_absent_sampled_training":
            not (set(meta["validation"].heldout_cluster_id) & training_clusters)}
    for key in ("prediction_group_id", "surface_replicate_id", "source_combodrugseq"):
        checks[f"disjoint_{key}"] = not (set(tr[key]) & set(va[key]))
    return {"checks": checks, "required_partner_cell_contexts": len(required),
            "missing_partner_cell_contexts": len(required-support)}


def frozen_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def write_history(out, history, **extra):
    learned._write_json_atomic(out / "training_progress.json", {"history": history, **extra})


def metric(target, pred, surfaces):
    target, pred = np.asarray(target), np.asarray(pred)
    if not np.isfinite(pred).all() or target.shape != pred.shape:
        raise RuntimeError("Invalid predictions or incomplete coverage")
    tail = target > 1
    return dict(surface_macro_rmse=learned.surface_macro_rmse(target, pred, surfaces),
                point_rmse=float(np.sqrt(np.mean((target - pred) ** 2))),
                mae=float(np.mean(np.abs(target - pred))),
                above_one_rmse=float(np.sqrt(np.mean((target[tail]-pred[tail])**2))) if tail.any() else None)


def loss_equal_surface(pred, target, surfaces):
    return graph._surface_mean_mse(pred.reshape(-1), target.reshape(-1), surfaces)


def train_graph(tables, banks, cfg, seed, mono, out, deadline):
    learned._seed_everything(seed)
    device = torch.device(cfg["device"])
    model = CompactBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim, 136,
                            graph_layers=2, attention_heads=4, dropout=0.1,
                            monotherapy_upper_bound=3.0, residual_monotherapy_input=mono).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                 weight_decay=cfg["training"]["weight_decay"])
    tr, va = tables["train"], tables["validation"]
    def predict():
        return graph.predict_graph_compact(model, va, banks,
            batch_size=cfg["training"].get("evaluation_batch_size", 128), device=device)[0]
    initial = predict()
    np.testing.assert_allclose(initial, np.minimum(va.mono_a.numpy(), va.mono_b.numpy()),
                               atol=3e-6, rtol=3e-6)
    best = metric(va.target.numpy(), initial, va.surface_ids)["surface_macro_rmse"]
    state, epoch_best = frozen_state(model), -1
    history = [{"epoch": -1, "surface_macro_rmse": best}]
    for epoch in range(cfg["training"]["epochs"]):
        if time.monotonic() > deadline:
            raise TimeoutError("Frozen development wall budget exceeded")
        model.train()
        values = []
        for idx in graph._surface_batch_indices(tr, cfg["training"]["batch_surfaces"],
                                               shuffle_seed=seed*1000003+epoch):
            optimizer.zero_grad(set_to_none=True)
            if time.monotonic() > deadline:
                raise TimeoutError("Frozen development wall budget exceeded")
            pred = graph._model_forward(model, tr, banks, idx, device).prediction
            loss = loss_equal_surface(pred, tr.target[idx].to(device), tr.surface_ids[idx.numpy()])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite graph loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite graph gradient")
            optimizer.step()
            values.append(float(loss.detach()))
        score = metric(va.target.numpy(), predict(), va.surface_ids)["surface_macro_rmse"]
        if score < best - cfg["training"]["min_delta"]:
            best, state, epoch_best = score, frozen_state(model), epoch
        history.append({"epoch": epoch, "train_batch_mean_mse": float(np.mean(values)),
                        "surface_macro_rmse": score})
        print(f"BiSurf mono={mono} seed={seed} epoch={epoch} val={score:.7f}", flush=True)
        write_history(out, history, device=str(device), best_epoch=epoch_best)
    model.load_state_dict(state)
    prediction = predict()
    torch.save(state, out / "model.pt")
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(predict(), prediction, atol=0, rtol=0)
    return prediction, {"best_epoch": epoch_best, "history": history,
                        "parameters": sum(p.numel() for p in model.parameters())}


def morgan512(data_root, banks):
    frame = pd.read_csv(data_root / "data/processed/nci_almanac_unclipped/drugs.csv", keep_default_na=False)
    lookup = frame.set_index("drug_id").canonical_smiles.to_dict()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    result = []
    for drug in banks.drug_ids:
        molecule = Chem.MolFromSmiles(lookup[drug])
        if molecule is None:
            raise RuntimeError("Invalid frozen canonical SMILES")
        vector = np.zeros(512, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), vector)
        result.append(vector)
    return torch.tensor(np.stack(result))


def dd_features(context, banks, fold):
    state = context.folds[fold].state
    expression = context.expression
    indices = expression.cell_indices(banks.biological_cell_ids)
    z = _gene_z_scores(expression.values[indices], expression.observed_mask[indices], state)
    genes, tensors = {}, []
    for index, name in enumerate(state.pathway_ids):
        selected = torch.where(state.membership[index])[0]
        if len(selected) < 1:
            raise RuntimeError(f"Empty pathway: {name}")
        genes[name] = [state.gene_ids[i] for i in selected.tolist()]
        tensors.append(z[:, selected])
    return genes, tensors


def train_dd(tables, banks, fingerprints, genes, expressions, cfg, seed, out, deadline):
    learned._seed_everything(seed)
    device = torch.device(cfg["device"])
    Mono, Combo = load_author_classes(ROOT, len(genes))
    mono, combo = Mono(genes).to(device), Combo().to(device)
    expressions = [e.to(device) for e in expressions]
    fingerprints = fingerprints.to(device)
    tr, va = tables["train"], tables["validation"]
    # Only training-role monotherapy covariates train the representation.
    rows = pd.DataFrame({
        "drug": torch.cat([tr.drug_a_index, tr.drug_b_index]).numpy(),
        "cell": torch.cat([tr.cell_index, tr.cell_index]).numpy(),
        "dose": torch.cat([tr.doses[:, 0], tr.doses[:, 1]]).numpy(),
        "target": torch.cat([tr.mono_a, tr.mono_b]).numpy(),
    }).groupby(["drug", "cell", "dose"], as_index=False).target.median()
    di = torch.tensor(rows.drug.to_numpy(), dtype=torch.long, device=device)
    ci = torch.tensor(rows.cell.to_numpy(), dtype=torch.long, device=device)
    raw = (torch.pow(10.0, torch.tensor(rows.dose.to_numpy(), device=device))-1.0)*1e-12
    doses = torch.log10(raw.clamp_min(1e-16)).reshape(-1, 1)+6
    target = torch.tensor(rows.target.to_numpy(), dtype=torch.float32, device=device).reshape(-1, 1)
    opt = torch.optim.AdamW(mono.parameters(), lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])
    mono_history = []
    for epoch in range(cfg["training"]["mono_epochs"]):
        mono.train(); losses = []
        for idx in minibatches(len(rows), cfg["training"]["batch_size"], seed+epoch):
            idx = idx.to(device)
            if time.monotonic() > deadline:
                raise TimeoutError("Frozen development wall budget exceeded")
            opt.zero_grad(set_to_none=True)
            pred = mono([[e[ci[idx]] for e in expressions], fingerprints[di[idx]], doses[idx]])
            loss = torch.mean((pred-target[idx])**2)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite DD monotherapy loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(mono.parameters(), cfg["training"]["gradient_clip"])
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite DD monotherapy gradient")
            opt.step(); losses.append(float(loss.detach()))
        mono_history.append({"epoch": epoch, "mse": float(np.mean(losses))})
        print(f"DD mono seed={seed} epoch={epoch} mse={np.mean(losses):.6f}", flush=True)
        write_history(out, mono_history, phase="monotherapy", device=str(device))
    mono.eval()
    def attention(table):
        result = []
        for drugs in (table.drug_a_index, table.drug_b_index):
            pieces = []
            for idx in minibatches(len(table), cfg["training"].get("evaluation_batch_size", 128), None):
                cells = table.cell_index[idx].long().to(device)
                pieces.append(capture_attention(mono, [[e[cells] for e in expressions],
                    fingerprints[drugs[idx].long().to(device)], torch.zeros(len(idx), 1, device=device)]))
            result.append(torch.cat(pieces))
        return result
    att = {role: attention(table) for role, table in tables.items()}
    observed_mono = {r: (t.mono_a.to(device), t.mono_b.to(device)) for r, t in tables.items()}
    def forward(role, idx):
        idx = idx.to(device)
        return combo([att[role][0][idx], att[role][1][idx], observed_mono[role][0][idx, None],
                      observed_mono[role][1][idx, None]])[2].reshape(-1)
    def predict():
        combo.eval()
        with torch.no_grad():
            return np.concatenate([forward("validation", idx).cpu().numpy()
                for idx in minibatches(len(va), cfg["training"].get("evaluation_batch_size", 128), None)])
    initial = predict()
    best = metric(va.target.numpy(), initial, va.surface_ids)["surface_macro_rmse"]
    state, best_epoch = frozen_state(combo), -1
    history = [{"epoch": -1, "surface_macro_rmse": best}]
    opt = torch.optim.AdamW(combo.parameters(), lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])
    for epoch in range(cfg["training"]["epochs"]):
        if time.monotonic() > deadline:
            raise TimeoutError("Frozen development wall budget exceeded")
        combo.train(); values = []
        for idx in graph._surface_batch_indices(tr, cfg["training"]["batch_surfaces"],
                                               shuffle_seed=seed*1000003+epoch):
            if time.monotonic() > deadline:
                raise TimeoutError("Frozen development wall budget exceeded")
            if len(idx) < 2:
                raise RuntimeError("DD BatchNorm requires at least two training rows")
            opt.zero_grad(set_to_none=True)
            loss = loss_equal_surface(forward("train", idx), tr.target[idx].to(device), tr.surface_ids[idx.numpy()])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite DD combination loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(combo.parameters(), cfg["training"]["gradient_clip"])
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite DD combination gradient")
            opt.step(); values.append(float(loss.detach()))
        score = metric(va.target.numpy(), predict(), va.surface_ids)["surface_macro_rmse"]
        if score < best-cfg["training"]["min_delta"]:
            best, state, best_epoch = score, frozen_state(combo), epoch
        history.append({"epoch": epoch, "surface_macro_rmse": score,
                        "train_batch_mean_mse": float(np.mean(values))})
        print(f"DD combo seed={seed} epoch={epoch} val={score:.7f}", flush=True)
        write_history(out, history, mono_history=mono_history, phase="combination", device=str(device))
    combo.load_state_dict(state); prediction = predict()
    torch.save({"mono": frozen_state(mono), "combo": state}, out / "model.pt")
    saved = torch.load(out / "model.pt", weights_only=True, map_location="cpu")
    mono.load_state_dict(saved["mono"]); combo.load_state_dict(saved["combo"])
    reloaded_att = attention(va)
    for actual, expected in zip(reloaded_att, att["validation"], strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    np.testing.assert_allclose(predict(), prediction, atol=0, rtol=0)
    return prediction, {"best_epoch": best_epoch, "mono_history": mono_history,
                        "history": history, "mono_training_points": len(rows),
                        "parameters": sum(p.numel() for m in (mono, combo) for p in m.parameters())}


def minibatches(n, batch_size, seed):
    idx = np.arange(n) if seed is None else np.random.default_rng(seed).permutation(n)
    chunks = list(torch.as_tensor(x) for x in np.array_split(idx, max(1, int(np.ceil(n/batch_size)))))
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        final = chunks.pop()
        chunks[-1] = torch.cat([chunks[-1], final])
    return chunks


def mono_covariates(frames):
    pieces = []
    for frame in frames.values():
        for arm in ("a", "b"):
            pieces.append(pd.DataFrame({"drug": frame[f"drug_{arm}_id"], "cell": frame.cell_id,
                "dose": (10.0**frame[f"dose_{arm}_model"]-1.0)*1e-12,
                "mono": frame[f"mono_response_{arm}"]*100.0}))
    return pd.concat(pieces).groupby(["cell", "drug", "dose"], as_index=False).mono.median()


def train_combo(frames, banks, fingerprints, cfg, out, deadline):
    author = load_author_module(ROOT)
    braid = author.braid_model_with_raw_c_input
    cov = mono_covariates({k: v.drop(columns="response") for k, v in frames.items()})
    hills, lookup, diagnostics = [], {}, []
    for (cell, drug), group in cov.groupby(["cell", "drug"], sort=True):
        if time.monotonic() > deadline:
            raise TimeoutError("Frozen development wall budget exceeded")
        dose, response = group.dose.to_numpy(), group.mono.to_numpy()
        def residual(p):
            emax, h, log_c = p
            power = (dose/(10.0**log_c))**h
            return 100.0 + (emax-100.0)*power/(1.0+power)-response
        fit = least_squares(residual, [min(float(response.min()), 99.0), 1.0, np.log10(np.median(dose))],
                            bounds=([0, 0.2, -16], [99.999, 3, -2]),
                            max_nfev=cfg["combokr"]["hill_max_nfev"])
        emax, h, log_c = fit.x
        lookup[(cell, drug)] = (100.0, emax, h, 10.0**log_c)
        hills.append(dict(cell=cell, drug=drug, Emax=emax, h=h, C=10.0**log_c))
        diagnostics.append(dict(kind="hill", cell=cell, drug=drug, success=bool(fit.success),
            nfev=int(fit.nfev), status=int(fit.status), unique_doses=len(np.unique(dose)),
            rmse=float(np.sqrt(np.mean(fit.fun**2)))/100.0))
    braids, cdd = [], []
    for (cell, a, b), group in frames["train"].groupby(["cell_id", "drug_a_id", "drug_b_id"], sort=True):
        if time.monotonic() > deadline:
            raise TimeoutError("Frozen development wall budget exceeded")
        h1, h2 = lookup[(cell, a)], lookup[(cell, b)]
        doses = [(10.0**group[f"dose_{arm}_model"].to_numpy()-1.0)*1e-12 for arm in ("a", "b")]
        params = [100.0, h1[1], h2[1], min(h1[1], h2[1])*0.5, h1[2], h2[2], h1[3], h2[3], 0.0]
        def residual(p):
            current = list(params); current[3], current[8] = p
            return braid(doses, *current)-group.response.to_numpy()*100.0
        fit = least_squares(residual, [params[3], 0.0],
                            bounds=([0, -1.9], [max(min(h1[1], h2[1]), 1e-6), 20]),
                            max_nfev=cfg["combokr"]["braid_max_nfev"])
        params[3], params[8] = fit.x
        braids.append(params); cdd.append([cell, a, b])
        diagnostics.append(dict(kind="braid_train_only", cell=cell, drug_a=a, drug_b=b,
                                success=bool(fit.success), rmse=float(np.sqrt(np.mean(fit.fun**2)))/100.0))
    original_cdd, braids = np.array(cdd), np.array(braids)
    cdd = np.vstack([original_cdd, original_cdd[:, [0, 2, 1]]])
    braids = np.vstack([braids, braids[:, [0, 2, 1, 3, 5, 4, 7, 6, 8]]])
    fp = fingerprints.numpy().astype(float)
    dot = fp @ fp.T; union = fp.sum(1)[:, None]+fp.sum(1)[None, :]-dot
    kd = np.divide(dot, union, out=np.zeros_like(dot), where=union > 0)
    np.fill_diagonal(kd, 1.0)
    cell = np.stack([g.node_features[:, 0].numpy() for g in banks.cell_graphs])
    kc = rbf_kernel(cell, gamma=cfg["combokr"]["kernel_gamma"])
    approximation = author.SurfaceKernelNystromDifference(tol=cfg["combokr"]["nystrom_tol"],
                                                          kernel="rbf", max_is_100=True)
    approximation.create_approximation()
    model = author.ScalablecomboKRdiff(kc, kd, np.array(banks.cell_ids), np.array(banks.drug_ids),
                                     approximation, hills=pd.DataFrame(hills), max_is_100=True)
    bind_small_kernel_backend(model, cfg["combokr"].get("max_training_surfaces", 1000))
    model.train(cdd, braids, cfg["combokr"]["maxiter"])
    pred = pd.Series(index=frames["validation"].index, dtype=float)
    for key, group in frames["validation"].groupby(["cell_id", "drug_a_id", "drug_b_id"], sort=True):
        if time.monotonic() > deadline:
            raise TimeoutError("Frozen development wall budget exceeded")
        doses = [(10.0**group[f"dose_{arm}_model"].to_numpy()-1.0)*1e-12 for arm in ("a", "b")]
        values, params = model.predict_with_candidates(np.array([key]), doses[0][None], doses[1][None])
        pred.loc[group.index] = values[0]/100.0
    pd.DataFrame(diagnostics).to_csv(out / "fit_diagnostics.csv", index=False)
    np.savez(out / "fitted_training.npz", braids=braids, cdd=cdd, kd=kd, kc=kc)
    pd.DataFrame(hills).to_csv(out / "hills.csv", index=False)
    return pred.to_numpy(), {"best_epoch": None, "training_surfaces": len(original_cdd),
                            "nystrom_components": model.Phiy_tr.shape[1],
                            "fit_failures": sum(not row["success"] for row in diagnostics),
        "solver_diagnostics": model.dense_fit_diagnostics,
        "backend": f"dense sampled-Kronecker MINRES; regparam=0; maxiter={cfg['combokr']['maxiter']}"}


def execute(output, cfg, config_path=CONFIG):
    start = time.monotonic(); deadline = start+cfg["max_wall_seconds"]
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable; no silent CPU fallback")
    data_root = (ROOT / cfg["data_root"]).resolve()
    jobs = [(m, s, "primary") for m in cfg["models"] if not m.startswith("combokr") for s in cfg["seeds"]]
    jobs += [(m, None, "primary") for m in cfg["models"] if m.startswith("combokr")]
    jobs += [(m, cfg["seeds"][0] if not m.startswith("combokr") else None, "repeat") for m in cfg["models"]]
    pd.DataFrame([dict(run=r, model=m, seed=s, status="pending") for m, s, r in jobs],
                 columns=RESULT_COLUMNS).to_csv(output / "results.csv", index=False)
    snapshots = [config_path, ROOT / cfg.get("protocol_document", "docs/COMPARATOR_PILOT_PROTOCOL.md"),
                 ROOT / "data/raw_manifests/model_comparators.json"]+list((ROOT / "src/bisurfgnn").rglob("*.py"))
    snapshots += list((ROOT / "experiments/model_improvement").glob("*.py"))
    records = []
    for path in snapshots:
        record = learned._file_record(path); records.append(record)
        dest = output / "source_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
    learned._write_json_atomic(output / "launch.json", {"config": cfg, "sources": records,
        "written_before_response_loading": True, "python": sys.version,
        "torch": torch.__version__, "numpy": np.__version__, "claim_scope": cfg["claim_scope"],
        "gpu": torch.cuda.get_device_name(0) if cfg["device"] == "cuda" else None})
    benchmark = load_benchmark_config(data_root / cfg["benchmark_config"])
    cell_cfg = load_cellminer_config(data_root / cfg["cell_config"])
    split = load_one_new_scaffold_benchmark(benchmark)
    context = load_benchmark_cell_context(benchmark, cell_cfg)
    shards = learned.load_learned_point_shards(benchmark, verify_roles=("train", "validation"))
    full_frames = {role: learned.load_learned_role_frame(shards, split.surfaces,
        cfg["fold"], role, sample_seed=cfg["sampling_seed"]) for role in ("train", "validation")}
    if cfg.get("preserve_partner_cell_support", False):
        frames = sample_supported_frames(full_frames["train"], full_frames["validation"], split.surfaces, cfg)
    else:
        frames = {role: sample_complete_groups(f, cfg[f"{role}_max_points"],
            cfg["sampling_seed"], cfg["sampling_salt"]) for role, f in full_frames.items()}
    del full_frames
    sample_audit = audit_sample(frames, split.surfaces)
    learned._write_json_atomic(output / "sample_split_audit.json", sample_audit)
    if cfg.get("preserve_partner_cell_support", False) and not all(sample_audit["checks"].values()):
        raise RuntimeError(f"Sample violates frozen task: {sample_audit}")
    for key in ("prediction_group_id", "surface_replicate_id", "source_combodrugseq"):
        if set(frames["train"][key]) & set(frames["validation"][key]):
            raise RuntimeError(f"Training/validation overlap: {key}")
    frame = pd.concat(frames.values())
    drugs = sorted(set(frame.drug_a_id) | set(frame.drug_b_id)); cells = sorted(set(frame.cell_id))
    banks = graph.build_graph_feature_banks(drugs, cells, context, cfg["fold"], benchmark)
    tables = {role: learned.build_compact_point_table(f, banks, role=role, fold=cfg["fold"],
               monotherapy_upper_bound=3.0) for role, f in frames.items()}
    for role, f in frames.items():
        f[["source_combodrugseq", "prediction_group_id", "surface_replicate_id"]].to_csv(
            output / f"{role}_ids.csv", index=False)
    inputs = [learned._file_record(shards.role_path(cfg["fold"], role)) for role in frames]
    inputs += [learned._file_record(data_root / cfg["benchmark_config"]),
               learned._file_record(data_root / cfg["cell_config"])]
    inputs += [learned._file_record(path) for path in (
        benchmark.output_dir / "benchmark_manifest.json",
        benchmark.output_dir / "surface_assignments.parquet",
        shards.root / "point_shard_manifest.json",
        benchmark.nci_processed_dir / "drugs.csv",
        cell_cfg.processed_dir / "expression_manifest.json",
        cell_cfg.processed_dir / "expression_tensors.pt")]
    inputs += list(banks.source_assets.values())
    learned._write_json_atomic(output / "data_binding.json", {"split_hash": split.split_hash,
        "context_state_hash": banks.context_state_hash, "inputs": inputs,
        "rows": {role: len(f) for role, f in frames.items()},
        "groups": {role: f.prediction_group_id.nunique() for role, f in frames.items()},
        "full_split_audit": split.manifest["split_audit"], "test_response_rows_read": 0})
    fingerprints = morgan512(data_root, banks)
    genes, expressions = dd_features(context, banks, cfg["fold"])
    result, primary, repeat_checks = [], {}, []
    for name, seed, repeat in jobs:
        job_start = time.monotonic()
        print(f"START {name} seed={seed} {repeat}", flush=True)
        out = output / f"{name}_seed_{seed}_{repeat}"; out.mkdir()
        learned._write_json_atomic(output / "progress.json", {"status": "running",
            "model": name, "seed": seed, "run": repeat, "completed_jobs": len(result),
            "total_jobs": len(jobs), "elapsed_seconds": time.monotonic()-start})
        if name.startswith("bisurf"):
            pred, detail = train_graph(tables, banks, cfg, seed, name.endswith("v2"), out, deadline)
        elif name.startswith("ddprism"):
            pred, detail = train_dd(tables, banks, fingerprints, genes, expressions, cfg, seed, out, deadline)
        else:
            pred, detail = train_combo(frames, banks, fingerprints, cfg, out, deadline)
        va = tables["validation"]
        row = dict(run=repeat, model=name, seed=seed, best_epoch=detail["best_epoch"],
                   **metric(va.target.numpy(), pred, va.surface_ids),
                   wall_seconds=time.monotonic()-job_start, status="passed")
        prediction_frame = pd.DataFrame(dict(source_row=va.source_rows, surface_id=va.surface_ids,
                                             target=va.target.numpy(), prediction=pred))
        prediction_frame.to_csv(out / "predictions.csv", index=False)
        prediction_frame.assign(abs_error=lambda f: abs(f.target-f.prediction)).nlargest(20, "abs_error").to_csv(
            out / "largest_errors.csv", index=False)
        learned._write_json_atomic(out / "metrics.json", {**row, **detail})
        if repeat == "primary":
            primary[(name, seed)] = pred.copy()
        else:
            difference = float(np.max(np.abs(primary[(name, seed)]-pred)))
            if difference > 1e-7:
                raise RuntimeError(f"Repeated prediction drift: {name}, {difference}")
            repeat_checks.append({"model": name, "max_abs_difference": difference})
        result.append(row)
        pd.DataFrame(result, columns=RESULT_COLUMNS).to_csv(output / "results.csv", index=False)
        print(f"DONE {name} seed={seed} rmse={row['surface_macro_rmse']:.7f}", flush=True)
        if cfg["device"] == "cuda":
            torch.cuda.empty_cache()
    for record in records+inputs:
        if learned._file_record(Path(record["path"]))["sha256"] != record["sha256"]:
            raise RuntimeError(f"Input/source drift: {record['path']}")
    summary = {"status": "passed", "claim_scope": cfg["claim_scope"], "results": result,
               "repeat_checks": repeat_checks, "test_response_rows_read": 0,
               "wall_seconds": time.monotonic()-start}
    learned._write_json_atomic(output / "summary.json", summary)
    learned._write_json_atomic(output / "progress.json", {"status": "passed",
        "completed_jobs": len(result), "total_jobs": len(jobs), "wall_seconds": summary["wall_seconds"]})
    if cfg.get("report_prefix"):
        prefix = cfg["report_prefix"]
        if Path(prefix).name != prefix:
            raise ValueError("Report prefix must be a filename stem")
        report_dir = ROOT / "reports/tables"; report_dir.mkdir(parents=True, exist_ok=True)
        results_frame = pd.DataFrame(result, columns=RESULT_COLUMNS)
        results_frame.to_csv(report_dir / f"{prefix}.csv", index=False)
        results_frame[results_frame.run == "primary"].groupby("model").surface_macro_rmse.agg(
            ["count", "mean", "std", "min", "max"]).to_csv(report_dir / f"{prefix}_summary.csv")
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(); output = args.output_dir.resolve()
    allowed = (ROOT / "runs/model_improvement").resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Output must be a fresh subdirectory of runs/model_improvement")
    output.mkdir(parents=True, exist_ok=False)
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg["device"] not in {"cpu", "cuda"} or cfg["fold"] != 0 or cfg["claim_scope"] != "engineering_pilot_no_performance_ranking":
        raise ValueError("Runner requires a validation-only fold-0 development configuration")
    accesses = []
    original = pd.read_parquet
    def guard(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved.name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError(f"Forbidden response source: {resolved}")
        accesses.append(str(resolved)); return original(path, *args, **kwargs)
    try:
        with threadpool_limits(limits=cfg["threads"]), patch.object(pd, "read_parquet", guard):
            summary = execute(output, cfg, config_path)
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
    except Exception as error:
        learned._write_json_atomic(output / "failure.json", {"status": "failed",
            "error": repr(error), "claim_scope": cfg["claim_scope"]})
        learned._write_json_atomic(output / "progress.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        learned._write_json_atomic(output / "input_access.json", {"parquet_reads": accesses,
                                                                 "test_response_rows_read": 0})


if __name__ == "__main__":
    main()

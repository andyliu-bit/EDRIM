"""AlgorithmSearch-v1: candidate architectures, auxiliary monotherapy loss, XGBoost comparator.

Reuses the frozen comparator pilot data path (`run_comparator_pilot.execute`):
point IDs/order, data binding, repeat-determinism, checkpoint reload and the
test-response guard are all unchanged. Protocol: docs/ALGORITHM_SEARCH_V1_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
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
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

from bisurfgnn import independent_graph_benchmark as graph
from bisurfgnn import learned_benchmark as learned
from bisurfgnn.models.compact_bisurf import CompactBiSurfGNN
from bisurfgnn.models.pathway_readout_bisurf import SearchBiSurfGNN

spec = importlib.util.spec_from_file_location("search_pilot", Path(__file__).with_name("run_comparator_pilot.py"))
pilot = importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)

VARIANTS = {
    "bisurf_mono_v2": None,
    # execute() routes names starting with "bisurf" to the graph trainer.
    "bisurf_search_pr_v2": dict(pathway_readout=True),
    "bisurf_search_aux_v2": dict(monotherapy_auxiliary=True),
    "bisurf_search_dcd_v2": dict(drug_cell_drug=True),
    "bisurf_search_pr_aux_v2": dict(pathway_readout=True, monotherapy_auxiliary=True),
    "bisurf_search_dcd_aux_v2": dict(drug_cell_drug=True, monotherapy_auxiliary=True),
    "bisurf_search_pr_dcd_aux_v2": dict(pathway_readout=True, drug_cell_drug=True, monotherapy_auxiliary=True),
    "bisurf_search_mdx_v2": dict(monotherapy_dose_branch=True),
    "bisurf_search_mdx_pr_v2": dict(monotherapy_dose_branch=True, pathway_readout=True),
    "bisurf_search_mdx_aux_v2": dict(monotherapy_dose_branch=True, monotherapy_auxiliary=True),
    "bisurf_search_mdx_pr_aux_v2": dict(monotherapy_dose_branch=True, pathway_readout=True,
                                        monotherapy_auxiliary=True),
}
XGB_PATTERN = re.compile(r"xgb_residual_d(\d+)_m(\d+)")


def registered(name: str) -> bool:
    return name in VARIANTS or XGB_PATTERN.fullmatch(name) is not None


def build_model(banks, variant, seed, cfg):
    learned._seed_everything(seed)
    kwargs = dict(graph_layers=2, attention_heads=4, dropout=0.1,
                  monotherapy_upper_bound=3.0, residual_monotherapy_input=True)
    flags = VARIANTS[variant]
    if flags is None:
        model = CompactBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim, 136, **kwargs)
    else:
        model = SearchBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim, 136, **flags, **kwargs)
    learned._seed_everything(seed)
    return model.to(torch.device(cfg["device"]))


def check_reference(out, cfg, banks):
    """Byte-compare point IDs/order and data inputs with the frozen GPU v2 run."""
    reference = ROOT / cfg["reference_run"]; bindings = []
    for role in ("train", "validation"):
        old, current = reference / f"{role}_ids.csv", out.parent / f"{role}_ids.csv"
        record = learned._file_record(old)
        if learned._file_record(current)["sha256"] != record["sha256"]:
            raise RuntimeError(f"Point ID/order drift relative to GPU v2: {role}")
        bindings.append(record)
    old_binding = json.loads((reference / "data_binding.json").read_text())
    if banks.context_state_hash != old_binding["context_state_hash"]:
        raise RuntimeError("Training-fitted expression state drift")
    current = {r["path"]: r["sha256"] for r in json.loads((out.parent / "data_binding.json").read_text())["inputs"]}
    if any(current.get(r["path"]) != r["sha256"] for r in old_binding["inputs"]):
        raise RuntimeError("Frozen data or preprocessing changed since GPU v2")
    return bindings


def predict(model, table, banks, cfg):
    return graph.predict_graph_compact(model, table, banks,
        batch_size=cfg["training"]["evaluation_batch_size"], device=torch.device(cfg["device"]))[0]


def forward_with_auxiliary(model, table, banks, idx, device):
    """Same batching as `graph._model_forward`, plus per-arm monotherapy predictions."""
    index = idx.to(dtype=torch.int64, device="cpu")
    a, b = table.drug_a_index[index].long(), table.drug_b_index[index].long()
    drug_graphs, drug_inverse = graph._unique_graph_batch(torch.cat([a, b]), banks.drug_graphs, device)
    n = len(index)
    cell_graphs, cell_inverse = graph._unique_graph_batch(table.cell_index[index].long(), banks.cell_graphs, device)
    masks = torch.ones(n, dtype=torch.bool, device=device)
    return model.forward_with_auxiliary(drug_graphs, cell_graphs, drug_inverse[:n], drug_inverse[n:],
        cell_inverse, table.doses[index].to(device), table.mono_a[index].to(device),
        table.mono_b[index].to(device), masks, masks)


def new_module_norms(model):
    norms = {}
    for name in ("pathway_readout", "atom_pathway"):
        if hasattr(model, name):
            norms[f"{name}_output_weight_norm"] = float(getattr(model, name).output.weight.norm())
    if hasattr(model, "monotherapy_dose_branch"):
        norms["monotherapy_dose_head_weight_norm"] = float(model.monotherapy_dose_branch.head.weight.norm())
    return norms


def train_variant(tables, banks, cfg, seed, mono, out, deadline):
    del mono
    variant = out.name.split("_seed_")[0]
    if variant not in VARIANTS:
        raise ValueError(f"Unregistered search variant: {variant}")
    bindings = check_reference(out, cfg, banks)
    model = build_model(banks, variant, seed, cfg)
    auxiliary = getattr(model, "use_monotherapy_auxiliary", False)
    weight = float(cfg.get("auxiliary_weight", 0.5))
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    tr, va = tables["train"], tables["validation"]
    initial = predict(model, va, banks, cfg)
    np.testing.assert_allclose(initial, np.minimum(va.mono_a.numpy(), va.mono_b.numpy()), atol=3e-6, rtol=3e-6)
    best = pilot.metric(va.target.numpy(), initial, va.surface_ids)["surface_macro_rmse"]
    state, best_epoch = pilot.frozen_state(model), -1
    history = [{"epoch": -1, "surface_macro_rmse": best, **new_module_norms(model)}]
    for epoch in range(cfg["training"]["epochs"]):
        model.train(); combo_losses, aux_losses = [], []; epoch_start = time.monotonic()
        for idx in graph._surface_batch_indices(tr, cfg["training"]["batch_surfaces"], shuffle_seed=seed*1000003+epoch):
            if time.monotonic() > deadline:
                raise TimeoutError("Frozen search wall budget exceeded")
            optimizer.zero_grad(set_to_none=True)
            target = tr.target[idx].to(device)
            if auxiliary:
                output, (mono_pred_a, mono_pred_b) = forward_with_auxiliary(model, tr, banks, idx, device)
                combo = pilot.loss_equal_surface(output.prediction, target, tr.surface_ids[idx.numpy()])
                # Training-role observed monotherapy only; never an input of the auxiliary head.
                aux = 0.5 * (torch.mean((mono_pred_a - tr.mono_a[idx].to(device)) ** 2)
                             + torch.mean((mono_pred_b - tr.mono_b[idx].to(device)) ** 2))
                loss = combo + weight * aux
                aux_losses.append(float(aux.detach()))
            else:
                output = graph._model_forward(model, tr, banks, idx, device)
                combo = pilot.loss_equal_surface(output.prediction, target, tr.surface_ids[idx.numpy()])
                loss = combo
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite search loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite search gradient")
            optimizer.step(); combo_losses.append(float(combo.detach()))
        score = pilot.metric(va.target.numpy(), predict(model, va, banks, cfg), va.surface_ids)["surface_macro_rmse"]
        if score < best-cfg["training"]["min_delta"]:
            best, state, best_epoch = score, pilot.frozen_state(model), epoch
        history.append({"epoch": epoch, "surface_macro_rmse": score,
            "train_batch_mean_mse": float(np.mean(combo_losses)),
            "train_auxiliary_mse": float(np.mean(aux_losses)) if aux_losses else None,
            "wall_seconds": time.monotonic()-epoch_start, **new_module_norms(model)})
        pilot.write_history(out, history, best_epoch=best_epoch, variant=variant, device=str(device))
        print(f"SEARCH {variant} seed={seed} epoch={epoch} val={score:.7f}", flush=True)
    model.load_state_dict(state)
    prediction = predict(model, va, banks, cfg)
    torch.save(state, out / "model.pt")
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(predict(model, va, banks, cfg), prediction, atol=0, rtol=0)
    return prediction, {"best_epoch": best_epoch, "history": history, "variant": variant,
        "auxiliary_weight": weight if auxiliary else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "reference_ids": bindings,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


# ----- XGBoost residual comparator (protocol section 1, C2) -------------------
def xgb_features(frame, drug_feat, cell_feat, drug_index, cell_index, swap):
    a, b = ("b", "a") if swap else ("a", "b")
    ma, mb = frame[f"mono_response_{a}"].to_numpy(float), frame[f"mono_response_{b}"].to_numpy(float)
    da, db = frame[f"dose_{a}_model"].to_numpy(float), frame[f"dose_{b}_model"].to_numpy(float)
    fa = drug_feat[frame[f"drug_{a}_id"].map(drug_index).to_numpy()]
    fb = drug_feat[frame[f"drug_{b}_id"].map(drug_index).to_numpy()]
    cell = cell_feat[frame.cell_id.map(cell_index).to_numpy()]
    return np.column_stack([ma, mb, da, db, np.minimum(ma, mb), ma*mb, np.abs(ma-mb), fa, fb, cell])


def train_xgb(frames, banks, fingerprints, cfg, out, deadline):
    import xgboost as xgb
    from sklearn.decomposition import PCA
    name = out.name.split("_seed_")[0]; seed = int(out.name.split("_seed_")[1].split("_")[0])
    match = XGB_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unregistered comparator: {name}")
    depth, min_child = int(match.group(1)), int(match.group(2))
    bindings = check_reference(out, cfg, banks)
    tr, va = frames["train"], frames["validation"]
    ids = pd.read_csv(out.parent / "validation_ids.csv")
    if not (ids.source_combodrugseq.to_numpy() == va.source_combodrugseq.to_numpy()).all():
        raise RuntimeError("Validation frame order differs from the frozen ID file")
    xc = cfg["xgboost"]
    drug_index = {d: i for i, d in enumerate(banks.drug_ids)}
    cell_index = {c: i for i, c in enumerate(banks.cell_ids)}
    fp = fingerprints.numpy().astype(float)
    train_drugs = sorted(set(tr.drug_a_id) | set(tr.drug_b_id))
    drug_feat = PCA(xc["drug_pca"], random_state=0).fit(fp[[drug_index[d] for d in train_drugs]]).transform(fp)
    cell_vec = np.stack([g.node_features[:, 0].numpy() for g in banks.cell_graphs]).astype(float)
    train_cells = sorted(set(tr.cell_id))
    cell_feat = PCA(xc["cell_pca"], random_state=0).fit(cell_vec[[cell_index[c] for c in train_cells]]).transform(cell_vec)
    build = lambda f, s: xgb_features(f, drug_feat, cell_feat, drug_index, cell_index, s)
    hsa_tr = np.minimum(tr.mono_response_a, tr.mono_response_b).to_numpy(float)
    hsa_va = np.minimum(va.mono_response_a, va.mono_response_b).to_numpy(float)
    weight = 1.0 / tr.groupby("surface_replicate_id").surface_replicate_id.transform("size").to_numpy(float)
    x = np.vstack([build(tr, False), build(tr, True)])
    y = np.concatenate([tr.response.to_numpy(float) - hsa_tr] * 2)
    w = np.concatenate([weight, weight])
    params = dict(objective="reg:squarederror", eta=xc["learning_rate"], max_depth=depth,
                  min_child_weight=min_child, subsample=xc["subsample"], colsample_bytree=xc["colsample"],
                  tree_method="hist", seed=seed, nthread=xc["threads"])
    booster = xgb.train(params, xgb.DMatrix(x, label=y, weight=w), num_boost_round=xc["max_trees"])
    if time.monotonic() > deadline:
        raise TimeoutError("Frozen search wall budget exceeded")
    dva = (xgb.DMatrix(build(va, False)), xgb.DMatrix(build(va, True)))
    target, surfaces = va.response.to_numpy(float), va.surface_replicate_id.to_numpy()
    def predict_at(k):
        if k == 0:
            return hsa_va.copy()
        return hsa_va + 0.5 * (booster.predict(dva[0], iteration_range=(0, k))
                               + booster.predict(dva[1], iteration_range=(0, k)))
    history, best_k, best = [], 0, pilot.metric(target, hsa_va, surfaces)["surface_macro_rmse"]
    history.append({"trees": 0, "surface_macro_rmse": best})
    for k in range(xc["eval_every"], xc["max_trees"] + 1, xc["eval_every"]):
        score = pilot.metric(target, predict_at(k), surfaces)["surface_macro_rmse"]
        history.append({"trees": k, "surface_macro_rmse": score})
        if score < best - cfg["training"]["min_delta"]:
            best, best_k = score, k
    prediction = predict_at(best_k)
    booster.save_model(out / "model.json")
    reloaded = xgb.Booster(); reloaded.load_model(out / "model.json")
    if best_k:
        again = hsa_va + 0.5 * (reloaded.predict(dva[0], iteration_range=(0, best_k))
                                + reloaded.predict(dva[1], iteration_range=(0, best_k)))
        np.testing.assert_allclose(again, prediction, atol=0, rtol=0)
    pilot.write_history(out, history, best_trees=best_k, variant=name)
    return prediction, {"best_epoch": best_k, "history": history, "variant": name,
        "reference_ids": bindings, "features": int(x.shape[1]), "augmented_training_rows": int(len(x)),
        "explained_variance_drug_pca": None, "params": params}


def assess(output, cfg):
    frame = pd.read_csv(output / "results.csv"); frame = frame[frame.run == "primary"]
    rows = []
    for name, group in frame.groupby("model"):
        values = group.surface_macro_rmse.to_numpy(); n = len(values)
        sd = float(values.std(ddof=1)) if n > 1 else float("nan")
        radius = float(student_t.ppf(.975, n-1)*sd/np.sqrt(n)) if n > 1 else float("nan")
        rows.append(dict(model=name, seeds=n, mean=float(values.mean()), sd=sd,
                         seed_mean_ci95_low=float(values.mean())-radius, seed_mean_ci95_high=float(values.mean())+radius))
    statistics = pd.DataFrame(rows).sort_values("mean")
    statistics.to_csv(output / "seed_statistics.csv", index=False)
    if cfg.get("report_prefix"):
        statistics.to_csv(ROOT / "reports/tables" / f"{cfg['report_prefix']}_seed_statistics.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); config_path = args.config.resolve(); output = args.output_dir.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not all(registered(m) for m in cfg["models"]) or cfg["fold"] != 0 \
            or cfg["claim_scope"] != "engineering_pilot_no_performance_ranking":
        raise ValueError("Unregistered or non-development search configuration")
    allowed = ROOT / "runs/model_improvement"
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Output must be a fresh experiment subdirectory")
    output.mkdir(parents=True, exist_ok=False)
    reference = ROOT / cfg["reference_run"]
    sources = [learned._file_record(reference / name) for name in (
        "train_ids.csv", "validation_ids.csv", "data_binding.json", "summary.json")]
    learned._write_json_atomic(output / "reference_binding.json", {"sources": sources, "written_before_response_loading": True})
    accesses = []; original = pd.read_parquet
    def guard(path, *a, **k):
        resolved = Path(path).resolve()
        if resolved.name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError(f"Forbidden response source: {resolved}")
        accesses.append(str(resolved)); return original(path, *a, **k)
    try:
        with threadpool_limits(limits=cfg["threads"]), patch.object(pd, "read_parquet", guard), \
                patch.object(pilot, "train_graph", train_variant), patch.object(pilot, "train_combo", train_xgb):
            result = pilot.execute(output, cfg, config_path)
        for record in sources:
            if learned._file_record(Path(record["path"]))["sha256"] != record["sha256"]:
                raise RuntimeError("Reference run drifted during training")
        assess(output, cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    except Exception as error:
        learned._write_json_atomic(output / "failure.json", {"status": "failed", "error": repr(error)})
        learned._write_json_atomic(output / "progress.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        learned._write_json_atomic(output / "input_access.json", {"parquet_reads": accesses, "test_response_rows_read": 0})


if __name__ == "__main__":
    main()

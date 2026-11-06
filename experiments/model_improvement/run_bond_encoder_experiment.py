"""Validation-only bond ablation using the established frozen pilot data path."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
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
from bisurfgnn.data.batch import pad_graph_items
from bisurfgnn.data.bond_graphs import load_bond_graph_bank, pad_bond_graphs
from bisurfgnn.data.chemistry import BOND_FEATURE_DIM
from bisurfgnn.models.bond_bisurf import BondGraphEncoder
from bisurfgnn.models.compact_bisurf import CompactBiSurfGNN

spec = importlib.util.spec_from_file_location("bond_trial_pilot", Path(__file__).with_name("run_comparator_pilot.py"))
pilot = importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
VARIANTS = {"bisurf_mono_v2": None, "bisurf_node_gate_v2": "node_control",
            "bisurf_uniform_bond_v2": "uniform_bonds", "bisurf_full_bond_v2": "full_bonds"}


def build_model(banks, mode, seed, cfg):
    learned._seed_everything(seed)
    model = CompactBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim, 136,
        graph_layers=2, attention_heads=4, dropout=0.1, monotherapy_upper_bound=3.0,
        residual_monotherapy_input=True)
    if mode is not None:
        encoder = BondGraphEncoder(banks.atom_feature_dim, 136, 2, 0.1, mode=mode)
        result = encoder.load_state_dict(model.drug_encoder.state_dict(), strict=False)
        if result.unexpected_keys or set(result.missing_keys) != {
            f"bond_gates.{i}.{p}" for i in range(2) for p in ("weight", "bias")}:
            raise RuntimeError("Common encoder initialisation did not align")
        model.drug_encoder = encoder
    # Construction consumes different RNG amounts; align all training randomness.
    learned._seed_everything(seed)
    return model.to(torch.device(cfg["device"]))


def forward_points(model, table, banks, molecules, idx, device):
    a, b = table.drug_a_index[idx].long(), table.drug_b_index[idx].long()
    unique, inverse = torch.unique(torch.cat([a, b]), sorted=True, return_inverse=True)
    drug_graphs = pad_bond_graphs([molecules[int(i)] for i in unique]).to(device)
    cell_graphs, cell_inverse = graph._unique_graph_batch(table.cell_index[idx], banks.cell_graphs, device)
    inverse = inverse.to(device); masks = torch.ones(len(idx), dtype=torch.bool, device=device)
    return model(drug_graphs, cell_graphs, inverse[:len(idx)], inverse[len(idx):], cell_inverse,
        table.doses[idx].to(device), table.mono_a[idx].to(device), table.mono_b[idx].to(device), masks, masks)


def predict(model, table, banks, molecules, cfg):
    model.eval(); device = torch.device(cfg["device"])
    predictions = []
    with torch.no_grad():
        drugs = model.encode_drugs(pad_bond_graphs(molecules).to(device))
        context = model.encode_context(pad_graph_items(banks.cell_graphs).to(device))
        for start in range(0, len(table), cfg["training"]["evaluation_batch_size"]):
            idx = torch.arange(start, min(len(table), start+cfg["training"]["evaluation_batch_size"]))
            masks = torch.ones(len(idx), dtype=torch.bool, device=device)
            result = model.forward_pooled(drugs, context, table.drug_a_index[idx].long().to(device),
                table.drug_b_index[idx].long().to(device), table.cell_index[idx].long().to(device),
                table.doses[idx].to(device), table.mono_a[idx].to(device), table.mono_b[idx].to(device), masks, masks)
            predictions.append(result.prediction.cpu().numpy())
    return np.concatenate(predictions)


def train_variant(tables, banks, cfg, seed, mono, out, deadline):
    del mono
    variant = out.name.split("_seed_")[0]
    if variant not in VARIANTS:
        raise ValueError("Unregistered bond variant")
    # Compare full point IDs, not just group counts. A changed task stops the run.
    reference = ROOT / cfg["reference_run"]
    bindings = []
    for role in tables:
        old, current = reference / f"{role}_ids.csv", out.parent / f"{role}_ids.csv"
        record = learned._file_record(old)
        if learned._file_record(current)["sha256"] != record["sha256"]:
            raise RuntimeError(f"Point ID/order drift relative to GPU v2: {role}")
        bindings.append(record)
    old_binding = json.loads((reference / "data_binding.json").read_text())
    if banks.context_state_hash != old_binding["context_state_hash"]:
        raise RuntimeError("Training-fitted expression state drift")
    current_binding = json.loads((out.parent / "data_binding.json").read_text())
    current_inputs = {r["path"]: r["sha256"] for r in current_binding["inputs"]}
    if any(current_inputs.get(r["path"]) != r["sha256"] for r in old_binding["inputs"]):
        raise RuntimeError("Frozen data or preprocessing changed since GPU v2")
    data_root = (ROOT / cfg["data_root"]).resolve()
    structure_path = data_root / "data/processed/nci_almanac_unclipped/structures.pt"
    molecules = load_bond_graph_bank(banks, structure_path)
    learned._write_json_atomic(out.parent / "bond_feature_binding.json", {
        "structures": learned._file_record(structure_path), "bond_feature_dim": BOND_FEATURE_DIM,
        "graphs": len(molecules), "max_atoms": max(len(g.node_features) for g in molecules),
        "atoms_and_adjacency_equal_existing_bank": True,
        "directed_edges": sum(g.edge_index.shape[1] for g in molecules)})
    model = build_model(banks, VARIANTS[variant], seed, cfg)
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    tr, va = tables["train"], tables["validation"]
    initial = predict(model, va, banks, molecules, cfg)
    np.testing.assert_allclose(initial, np.minimum(va.mono_a.numpy(), va.mono_b.numpy()), atol=3e-6, rtol=3e-6)
    best = pilot.metric(va.target.numpy(), initial, va.surface_ids)["surface_macro_rmse"]
    state, best_epoch = pilot.frozen_state(model), -1
    history = [{"epoch": -1, "surface_macro_rmse": best}]
    for epoch in range(cfg["training"]["epochs"]):
        model.train(); values = []; epoch_start = time.monotonic()
        for idx in graph._surface_batch_indices(tr, cfg["training"]["batch_surfaces"], shuffle_seed=seed*1000003+epoch):
            if time.monotonic() > deadline:
                raise TimeoutError("Frozen bond development budget exceeded")
            optimizer.zero_grad(set_to_none=True)
            output = forward_points(model, tr, banks, molecules, idx, device)
            loss = pilot.loss_equal_surface(output.prediction, tr.target[idx].to(device), tr.surface_ids[idx.numpy()])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite bond loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite bond gradient")
            optimizer.step(); values.append(float(loss.detach()))
        score = pilot.metric(va.target.numpy(), predict(model, va, banks, molecules, cfg), va.surface_ids)["surface_macro_rmse"]
        if score < best-cfg["training"]["min_delta"]:
            best, state, best_epoch = score, pilot.frozen_state(model), epoch
        history.append({"epoch": epoch, "surface_macro_rmse": score,
            "train_batch_mean_mse": float(np.mean(values)), "wall_seconds": time.monotonic()-epoch_start})
        pilot.write_history(out, history, best_epoch=best_epoch, variant=variant, device=str(device))
        print(f"BOND {variant} seed={seed} epoch={epoch} val={score:.7f}", flush=True)
    model.load_state_dict(state)
    prediction = predict(model, va, banks, molecules, cfg)
    torch.save(state, out / "model.pt")
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(predict(model, va, banks, molecules, cfg), prediction, atol=0, rtol=0)
    return prediction, {"best_epoch": best_epoch, "history": history, "mode": VARIANTS[variant],
        "parameters": sum(p.numel() for p in model.parameters()), "bond_feature_dim": BOND_FEATURE_DIM,
        "additional_parameters": 0 if VARIANTS[variant] is None else 2*(BOND_FEATURE_DIM+2),
        "reference_ids": bindings, "structures": learned._file_record(structure_path),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


def assess(output, cfg):
    if cfg.get("stage") == "preflight":
        return
    frame = pd.read_csv(output / "results.csv"); frame = frame[frame.run == "primary"]
    paired = frame.pivot(index="seed", columns="model", values="surface_macro_rmse")
    base, full = paired.bisurf_mono_v2, paired.bisurf_full_bond_v2
    checks = {"four_of_five_seeds_improve": int((full < base).sum()) >= cfg["go_required_seeds"],
        "mean_relative_reduction_at_least_one_percent": float((base.mean()-full.mean())/base.mean()) >= cfg["go_relative_improvement"],
        "mean_better_than_node_gate": bool(full.mean() < paired.bisurf_node_gate_v2.mean()),
        "mean_better_than_uniform_bonds": bool(full.mean() < paired.bisurf_uniform_bond_v2.mean())}
    statistics = []
    for name in paired:
        values = paired[name].dropna(); n = len(values)
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        radius = float(student_t.ppf(.975, n-1)*sd/np.sqrt(n))
        statistics.append(dict(model=name, seeds=n, mean=mean, sd=sd,
            seed_mean_ci95_low=mean-radius, seed_mean_ci95_high=mean+radius))
    statistics = pd.DataFrame(statistics)
    statistics.to_csv(output / "seed_statistics.csv", index=False)
    if cfg.get("report_prefix"):
        statistics.to_csv(ROOT / "reports/tables" / f"{cfg['report_prefix']}_seed_statistics.csv", index=False)
    difference = base-full
    radius = float(student_t.ppf(.975, len(difference)-1)*difference.std(ddof=1)/np.sqrt(len(difference)))
    learned._write_json_atomic(output / "development_decision.json", {
        "scope": "fixed_fold0_development_only", "checks": checks, "advance_to_multifold": all(checks.values()),
        "relative_rmse_reduction_percent": float(100*(base.mean()-full.mean())/base.mean()),
        "improved_seeds": int((full < base).sum()), "not_independent_confirmation": True,
        "paired_seed_mean_rmse_reduction": float(difference.mean()),
        "paired_seed_ci95_low": float(difference.mean()-radius),
        "paired_seed_ci95_high": float(difference.mean()+radius),
        "ci_scope": "training_seed_variability_conditional_on_fixed_development_sample"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model_improvement/bond_encoder_v1.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); config_path = args.config.resolve(); output = args.output_dir.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if set(cfg["models"]) != set(VARIANTS) or cfg["fold"] != 0 or cfg["claim_scope"] != "engineering_pilot_no_performance_ranking":
        raise ValueError("This experiment requires the four registered validation-only variants")
    allowed = ROOT / "runs/model_improvement"
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Output must be a fresh experiment subdirectory")
    output.mkdir(parents=True, exist_ok=False)
    reference = ROOT / cfg["reference_run"]
    sources = [learned._file_record(reference / name) for name in (
        "train_ids.csv", "validation_ids.csv", "data_binding.json", "summary.json")]
    learned._write_json_atomic(output / "reference_binding.json", {"sources": sources, "written_before_response_loading": True})
    accesses = []; original = pd.read_parquet
    def guard(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved.name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError(f"Forbidden response source: {resolved}")
        accesses.append(str(resolved)); return original(path, *args, **kwargs)
    try:
        with threadpool_limits(limits=cfg["threads"]), patch.object(pd, "read_parquet", guard), patch.object(pilot, "train_graph", train_variant):
            result = pilot.execute(output, cfg, config_path)
        for record in sources:
            if learned._file_record(Path(record["path"]))["sha256"] != record["sha256"]:
                raise RuntimeError("Reference run drifted during training")
        assess(output, cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    except Exception as error:
        learned._write_json_atomic(output / "failure.json", {"status": "failed", "error": repr(error)})
        learned._write_json_atomic(output / "progress.json", {"status": "failed", "error": repr(error)})
        if (output / "summary.json").exists():
            summary = json.loads((output / "summary.json").read_text())
            summary.update(status="failed", postprocessing_error=repr(error))
            learned._write_json_atomic(output / "summary.json", summary)
        raise
    finally:
        learned._write_json_atomic(output / "input_access.json", {"parquet_reads": accesses, "test_response_rows_read": 0})


if __name__ == "__main__":
    main()

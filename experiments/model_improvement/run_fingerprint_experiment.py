"""Frozen validation-only fingerprint fusion experiment, using existing shards."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys
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
from bisurfgnn.models.compact_bisurf import CompactBiSurfGNN
from bisurfgnn.models.fingerprint_bisurf import FingerprintBiSurfGNN, pad_fingerprint_graphs


def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_helper("fingerprint_remaining_helper", "run_remaining_variant_experiment.py")
pilot = helper.pilot
VARIANTS = {"bisurf_mono_v2": "base", "bisurf_fp_only_v2": "fingerprint_only",
            "bisurf_fp_fused_v2": "fused", "bisurf_graph_summary_v2": "graph_summary_control"}
_fingerprint_cache = {}


def fingerprints_for(banks, cfg):
    key = tuple(banks.drug_ids)
    if key not in _fingerprint_cache:
        value = pilot.morgan512((ROOT / cfg["data_root"]).resolve(), banks)
        if value.shape != (len(key), 512) or not bool(((value == 0) | (value == 1)).all()):
            raise RuntimeError("Invalid aligned Morgan fingerprint bank")
        _fingerprint_cache[key] = value
    return _fingerprint_cache[key]


def build_model(banks, mode, seed, cfg):
    learned._seed_everything(seed)
    kwargs = dict(graph_layers=2, attention_heads=4, dropout=.1,
                  monotherapy_upper_bound=3., residual_monotherapy_input=True)
    if mode == "base":
        model = CompactBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim, 136, **kwargs)
    else:
        model = FingerprintBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim, 136, mode=mode, **kwargs)
    learned._seed_everything(seed)
    return model.to(torch.device(cfg["device"]))


def forward_with_fingerprints(model, table, banks, indices, device, fingerprints):
    ix = indices.to(dtype=torch.int64, device="cpu")
    count = len(ix)
    drugs = torch.cat([table.drug_a_index[ix], table.drug_b_index[ix]]).long()
    unique, inverse = torch.unique(drugs, sorted=True, return_inverse=True)
    batch = pad_fingerprint_graphs([banks.drug_graphs[int(i)] for i in unique], fingerprints[unique]).to(device)
    cells, cell_inverse = graph._unique_graph_batch(table.cell_index[ix], banks.cell_graphs, device)
    flags = torch.ones(count, dtype=torch.bool, device=device)
    inverse = inverse.to(device)
    return model(batch, cells, inverse[:count], inverse[count:], cell_inverse,
                 table.doses[ix].to(device), table.mono_a[ix].to(device), table.mono_b[ix].to(device), flags, flags)


def predict_with_fingerprints(model, table, banks, *, batch_size, device, fingerprints):
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        drug_bank = model.encode_drugs(pad_fingerprint_graphs(banks.drug_graphs, fingerprints).to(device))
        cells = model.encode_context(graph.pad_graph_items(banks.cell_graphs).to(device))
        for ix in learned._batch_indices(len(table), batch_size, shuffle_seed=None):
            flags = torch.ones(len(ix), dtype=torch.bool, device=device)
            output = model.forward_pooled(drug_bank, cells, table.drug_a_index[ix].to(device),
                table.drug_b_index[ix].to(device), table.cell_index[ix].to(device), table.doses[ix].to(device),
                table.mono_a[ix].to(device), table.mono_b[ix].to(device), flags, flags)
            predictions.append(output.prediction.cpu().numpy())
            references.append(output.reference.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(references)


def train_variant(tables, banks, cfg, seed, mono, out, deadline):
    fp = fingerprints_for(banks, cfg)
    binding = {"drug_ids": list(banks.drug_ids), "shape": list(fp.shape),
        "sha256": hashlib.sha256(fp.contiguous().numpy().tobytes()).hexdigest(),
        "rdkit": __import__("rdkit").__version__, "radius": 2, "fp_size": 512,
        "source": "canonical_smiles_no_response_labels", "pretrained_model": None}
    binding_path = out.parent / "fingerprint_binding.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise RuntimeError("Fingerprint bank changed between tasks")
    else:
        learned._write_json_atomic(binding_path, binding)
        np.savez(out.parent / "fingerprints.npz", drug_ids=np.array(banks.drug_ids), fingerprints=fp.numpy())
    with patch.object(helper, "VARIANTS", VARIANTS), patch.object(helper, "build_model", build_model), \
         patch.object(graph, "_model_forward", lambda m,t,b,i,d: forward_with_fingerprints(m,t,b,i,d,fp)), \
         patch.object(graph, "predict_graph_compact", lambda m,t,b,**kw: predict_with_fingerprints(m,t,b,fingerprints=fp,**kw)):
        prediction, detail = helper.train_variant(tables, banks, cfg, seed, mono, out, deadline)
    detail.update(fingerprint_sha256=binding["sha256"], input_variant=VARIANTS[out.name.split("_seed_")[0]],
                  same_decoder_and_cell_encoder=True)
    return prediction, detail


def assess(output, cfg):
    helper.assess(output, cfg)
    if cfg["stage"] == "preflight":
        return
    path = output / "development_decision.json"
    decision = json.loads(path.read_text())
    scores = pd.read_csv(output / "results.csv")
    means = scores[scores.run == "primary"].groupby("model").surface_macro_rmse.mean()
    if "bisurf_fp_fused_v2" in means and "bisurf_graph_summary_v2" in means:
        for row in decision["candidates"]:
            if row["model"] == "bisurf_fp_fused_v2":
                row["checks"]["mean_better_than_same_capacity_graph_summary"] = bool(means[row["model"]] < means.bisurf_graph_summary_v2)
                row["advance_to_multifold"] = all(row["checks"].values())
    decision["multiple_comparisons"] = "Three exploratory candidates; no significance claim. Seed intervals descriptive, not multiplicity-adjusted."
    learned._write_json_atomic(path, decision)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    if cfg["protocol"] != "FingerprintFusion-v1" or cfg["fold"] != 0 or cfg["claim_scope"] != "engineering_pilot_no_performance_ranking":
        raise ValueError("Frozen fold-0 development protocol required")
    if set(cfg["models"]) != set(VARIANTS) or cfg["stage"] not in ("preflight", "development"):
        raise ValueError("All four frozen fingerprint variants are required")
    if cfg["stage"] == "development" and cfg["seeds"] != [41,42,43,44,45]:
        raise ValueError("All five registered development seeds are required")
    allowed = ROOT / "runs/model_improvement"
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Fresh experiment output directory required")
    output.mkdir(parents=True, exist_ok=False)
    reference = ROOT / cfg["reference_run"]
    sources = [learned._file_record(reference / name) for name in ("train_ids.csv", "validation_ids.csv", "data_binding.json", "summary.json")]
    learned._write_json_atomic(output / "reference_binding.json", {"sources":sources,"written_before_response_loading":True})
    accesses = []
    original = pd.read_parquet
    def guard(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved.name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError(f"Forbidden response source: {resolved}")
        accesses.append(str(resolved))
        return original(path, *args, **kwargs)
    try:
        with threadpool_limits(limits=cfg["threads"]), patch.object(pd, "read_parquet", guard), patch.object(pilot, "train_graph", train_variant):
            result = pilot.execute(output, cfg, args.config.resolve())
        for record in sources:
            if learned._file_record(Path(record["path"]))["sha256"] != record["sha256"]:
                raise RuntimeError("Reference run changed")
        assess(output, cfg)
        print(json.dumps({k:v for k,v in result.items() if k != "results"}), flush=True)
    except Exception as error:
        learned._write_json_atomic(output / "failure.json", {"status":"failed","error":repr(error)})
        learned._write_json_atomic(output / "progress.json", {"status":"failed","error":repr(error)})
        if (output / "summary.json").exists():
            result = json.loads((output / "summary.json").read_text())
            result.update(status="failed", postprocessing_error=repr(error))
            learned._write_json_atomic(output / "summary.json", result)
        raise
    finally:
        learned._write_json_atomic(output / "input_access.json", {"parquet_reads":accesses,"test_response_rows_read":0})


if __name__ == "__main__":
    main()

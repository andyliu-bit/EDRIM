"""PublicSplitGate-v1 development runs (S2 timing, S3/S4 training). Reads only split-1/2 point tables.

  timing   S2: time-only pilot on 5% of training surfaces (no accuracy is computed)
  bisurf   S3/S4: BiSurfGNN (--form anchored|additive) for the given seeds
  xgb      S4: frozen XGBoost residual comparator (xgb_residual_d10_m1) for the given seeds

Usage (repository root):
  .venv/Scripts/python.exe experiments/model_improvement/public_split_gate/run_gate.py timing --output-dir runs/model_improvement/public_split_gate_v1/s2_timing
  .venv/Scripts/python.exe experiments/model_improvement/public_split_gate/run_gate.py bisurf --form anchored --seeds 41 --output-dir runs/model_improvement/public_split_gate_v1/s3_anchored
  .venv/Scripts/python.exe experiments/model_improvement/public_split_gate/run_gate.py xgb --seeds 41 42 --output-dir runs/model_improvement/public_split_gate_v1/s4_xgb_a
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gate  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from gate_common import graph, learned, pilot, search  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs/gate.yaml"


def code_records() -> dict[str, str]:
    paths = sorted(set((gate.ROOT / "src/bisurfgnn").rglob("*.py")) | set((gate.ROOT / "experiments/model_improvement").glob("*.py"))
                   | {HERE / "gate_common.py", HERE / "run_gate.py", CONFIG})
    return {p.relative_to(gate.ROOT).as_posix(): gate.sha256(p) for p in paths}


def prepare(cfg):
    frames = {r: gate.load_points(r) for r in ("train", "validation")}
    banks = gate.build_banks(frames)
    tables = gate.build_tables(frames, banks)
    return frames, banks, tables


def predict(model, table, banks, cfg):
    return search.predict(model, table, banks, cfg)


def train_epoch(model, optimizer, table, banks, cfg, seed, epoch, device):
    model.train(); losses = []
    for idx in graph._surface_batch_indices(table, cfg["training"]["batch_surfaces"], shuffle_seed=seed * 1000003 + epoch):
        optimizer.zero_grad(set_to_none=True)
        target = table.target[idx].to(device)
        output = graph._model_forward(model, table, banks, idx, device)
        loss = pilot.loss_equal_surface(output.prediction, target, table.surface_ids[idx.numpy()])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("Non-finite gradient")
        optimizer.step(); losses.append(float(loss.detach()))
    return float(np.mean(losses))


def train_bisurf(tables, banks, cfg, form, seed, out):
    """run_algorithm_search.train_variant plus early stopping and a per-run wall cap (FROZEN_RECORD item 4)."""
    device = torch.device(cfg["device"]); t = cfg["training"]
    model = gate.build_model(banks, form, seed, cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    tr, va = tables["train"], tables["validation"]
    start = time.monotonic()
    initial = predict(model, va, banks, cfg)
    np.testing.assert_allclose(initial, np.minimum(va.mono_a.numpy(), va.mono_b.numpy()), atol=3e-6, rtol=3e-6)
    best = gate.surface_macro(va, initial); state, best_epoch = pilot.frozen_state(model), -1
    history = [{"epoch": -1, "surface_macro_rmse": best, "pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), initial)}]
    stale, stopped_by, last = 0, "max_epochs", 0.0
    for epoch in range(t["max_epochs"]):
        if time.monotonic() - start + last > cfg["wall_cap_seconds"]:
            stopped_by = "wall_cap"; break
        epoch_start = time.monotonic()
        loss = train_epoch(model, optimizer, tr, banks, cfg, seed, epoch, device)
        pred = predict(model, va, banks, cfg)
        score = gate.surface_macro(va, pred)
        improved = score < best - t["min_delta"]
        if improved:
            best, state, best_epoch, stale = score, pilot.frozen_state(model), epoch, 0
        else:
            stale += 1
        last = time.monotonic() - epoch_start
        history.append({"epoch": epoch, "surface_macro_rmse": score, "pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), pred),
                        "train_batch_mean_loss": loss, "wall_seconds": last, "improved": improved})
        pilot.write_history(out, history, best_epoch=best_epoch, form=form, device=str(device))
        print(f"BISURF {form} seed={seed} epoch={epoch} val_surface_macro={score:.7f} best={best:.7f} ({last:.0f}s)", flush=True)
        if stale >= t["patience"]:
            stopped_by = "early_stopping"; break
    model.load_state_dict(state)
    prediction = predict(model, va, banks, cfg)
    torch.save(state, out / "model.pt")
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(predict(model, va, banks, cfg), prediction, atol=0, rtol=0)
    return prediction, {"best_epoch": best_epoch, "stopped_by": stopped_by, "epochs_run": len(history) - 1,
                        "history": history, "parameters": sum(p.numel() for p in model.parameters()),
                        "wall_seconds": time.monotonic() - start,
                        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


def xgb_inputs(frames, banks, fingerprints, cfg):
    from sklearn.decomposition import PCA
    xc = cfg["xgboost"]; tr = frames["train"]
    drug_index = {d: i for i, d in enumerate(banks.drug_ids)}
    cell_index = {c: i for i, c in enumerate(banks.cell_ids)}
    fp = fingerprints.numpy().astype(float)
    train_drugs = sorted(set(tr.drug_a_id) | set(tr.drug_b_id))
    drug_feat = PCA(xc["drug_pca"], random_state=0).fit(fp[[drug_index[d] for d in train_drugs]]).transform(fp)
    cell_vec = np.stack([g.node_features[:, 0].numpy() for g in banks.cell_graphs]).astype(float)
    train_cells = sorted(set(tr.cell_id))
    cell_feat = PCA(xc["cell_pca"], random_state=0).fit(cell_vec[[cell_index[c] for c in train_cells]]).transform(cell_vec)
    return lambda f, s: search.xgb_features(f, drug_feat, cell_feat, drug_index, cell_index, s)


def base_margin(booster) -> float:
    raw = str(json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"])
    values = [float(v) for v in raw.strip("[]").split(",") if v.strip()]
    if len(values) != 1:
        raise RuntimeError(f"Unexpected base_score: {raw}")
    return values[0]


def train_xgb(frames, banks, fingerprints, cfg, seed, out):
    """run_algorithm_search.train_xgb (xgb_residual_d10_m1) with block-wise validation margins (FROZEN_RECORD item 6)."""
    import xgboost as xgb
    xc = cfg["xgboost"]; tr, va = frames["train"], frames["validation"]
    start = time.monotonic()
    build = xgb_inputs(frames, banks, fingerprints, cfg)
    hsa_tr = np.minimum(tr.mono_response_a, tr.mono_response_b).to_numpy(float)
    hsa_va = np.minimum(va.mono_response_a, va.mono_response_b).to_numpy(float)
    weight = 1.0 / tr.groupby("surface_replicate_id").surface_replicate_id.transform("size").to_numpy(float)
    x = np.vstack([build(tr, False), build(tr, True)])
    y = np.concatenate([tr.response.to_numpy(float) - hsa_tr] * 2)
    w = np.concatenate([weight, weight])
    params = dict(objective="reg:squarederror", eta=xc["learning_rate"], max_depth=10, min_child_weight=1,
                  subsample=xc["subsample"], colsample_bytree=xc["colsample"], tree_method="hist", seed=seed,
                  nthread=xc["threads"])
    booster = xgb.train(params, xgb.DMatrix(x, label=y, weight=w), num_boost_round=xc["max_trees"])
    train_seconds = time.monotonic() - start
    del x, y, w
    dva = (xgb.DMatrix(build(va, False)), xgb.DMatrix(build(va, True)))
    base = base_margin(booster)
    margin = [np.zeros(len(va)), np.zeros(len(va))]
    target, surfaces = va.response.to_numpy(float), va.surface_replicate_id.to_numpy()
    best_k, best = 0, pilot.metric(target, hsa_va, surfaces)["surface_macro_rmse"]
    history = [{"trees": 0, "surface_macro_rmse": best}]
    step = xc["eval_every"]
    for k in range(step, xc["max_trees"] + 1, step):
        for o in (0, 1):
            margin[o] += booster.predict(dva[o], iteration_range=(k - step, k), output_margin=True) - base
        pred = hsa_va + 0.5 * ((base + margin[0]) + (base + margin[1]))
        score = pilot.metric(target, pred, surfaces)["surface_macro_rmse"]
        history.append({"trees": k, "surface_macro_rmse": score, "pooled_rmse_percent": gate.pooled_rmse_percent(target, pred)})
        if score < best - cfg["training"]["min_delta"]:
            best, best_k, best_pred = score, k, pred.copy()
    prediction = best_pred if best_k else hsa_va.copy()
    booster.save_model(out / "model.json")
    reloaded = xgb.Booster(); reloaded.load_model(out / "model.json")
    direct_diff = 0.0
    if best_k:
        direct = hsa_va + 0.5 * (reloaded.predict(dva[0], iteration_range=(0, best_k))
                                 + reloaded.predict(dva[1], iteration_range=(0, best_k)))
        direct_diff = float(np.max(np.abs(direct - prediction)))
        if direct_diff > 1e-5:
            raise RuntimeError(f"Block-wise validation prediction differs from the direct prediction: {direct_diff}")
        prediction = direct
    pilot.write_history(out, history, best_trees=best_k, variant="xgb_residual_d10_m1")
    return prediction, {"best_epoch": best_k, "history": history, "params": params, "base_score": base,
                        "blockwise_vs_direct_max_abs": direct_diff, "train_seconds": train_seconds,
                        "wall_seconds": time.monotonic() - start}


def timing(frames, banks, tables, cfg, out):
    """S2: time only. Validation predictions are discarded without computing any metric."""
    import xgboost as xgb
    tr = frames["train"]
    surfaces = np.array(sorted(tr.surface_replicate_id.unique()))
    rng = np.random.default_rng(cfg["timing"]["subset_seed"])
    keep = set(rng.choice(surfaces, size=int(round(cfg["timing"]["fraction"] * len(surfaces))), replace=False))
    sub = tr[tr.surface_replicate_id.isin(keep)].reset_index(drop=True)
    sub_table = learned.build_compact_point_table(sub, banks, role="train", fold=0, monotherapy_upper_bound=3.0)
    device = torch.device(cfg["device"])
    model = gate.build_model(banks, "anchored", 41, cfg["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    t0 = time.monotonic(); train_epoch(model, optimizer, sub_table, banks, cfg, 41, 0, device); t_sub = time.monotonic() - t0
    t0 = time.monotonic(); _ = predict(model, tables["validation"], banks, cfg); t_val = time.monotonic() - t0
    del _
    scale = len(tr) / len(sub)
    epoch = t_sub * scale + t_val
    cap = cfg["wall_cap_seconds"]
    max_epochs = cfg["training"]["max_epochs"]
    if max_epochs * epoch > cap:
        max_epochs = max(cfg["timing"]["minimum_epochs"], int(cap // epoch))
    build = xgb_inputs({"train": sub}, banks, gate.morgan(banks), cfg)
    hsa = np.minimum(sub.mono_response_a, sub.mono_response_b).to_numpy(float)
    x = np.vstack([build(sub, False), build(sub, True)]); y = np.concatenate([sub.response.to_numpy(float) - hsa] * 2)
    params = dict(objective="reg:squarederror", eta=cfg["xgboost"]["learning_rate"], max_depth=10, min_child_weight=1,
                  subsample=cfg["xgboost"]["subsample"], colsample_bytree=cfg["xgboost"]["colsample"], tree_method="hist",
                  seed=41, nthread=cfg["xgboost"]["threads"])
    t0 = time.monotonic(); booster = xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=cfg["xgboost"]["eval_every"])
    t_xgb = time.monotonic() - t0
    dva = xgb.DMatrix(build(frames["validation"], False))
    t0 = time.monotonic(); _ = booster.predict(dva, output_margin=True); t_xgb_val = time.monotonic() - t0
    blocks = cfg["xgboost"]["max_trees"] / cfg["xgboost"]["eval_every"]
    result = dict(subset_surfaces=len(keep), subset_points=len(sub), full_train_points=len(tr),
                  bisurf=dict(subset_epoch_seconds=t_sub, validation_pass_seconds=t_val, projected_epoch_seconds=epoch,
                              projected_run_seconds_at_cap=max_epochs * epoch, max_epochs_decided=max_epochs,
                              rule="reduce only if max_epochs * projected epoch > wall cap; never below minimum_epochs"),
                  xgb=dict(subset_50_tree_seconds=t_xgb, projected_training_seconds=t_xgb * scale * blocks,
                           validation_block_seconds_one_order=t_xgb_val,
                           projected_validation_seconds=2 * t_xgb_val * blocks),
                  accuracy_computed=False)
    learned._write_json_atomic(out / "timing.json", result)
    print(json.dumps(result, indent=1), flush=True)
    smoke_test(frames, banks, cfg, out / "smoke")


def tiny_frames(frames, seed):
    rng = np.random.default_rng(seed)
    tiny = {}
    for role, frame in frames.items():
        surfaces = np.array(sorted(frame.surface_replicate_id.unique()))
        keep = set(rng.choice(surfaces, size=max(8, int(0.005 * len(surfaces))), replace=False))
        tiny[role] = frame[frame.surface_replicate_id.isin(keep)].reset_index(drop=True)
    return tiny


def smoke_test(frames, banks, cfg, out):
    """Engineering check (FROZEN_RECORD 2026-10-13 01:00): run the full training functions end to end on tiny
    subsets (0.5% of train and validation surfaces, 1 epoch, 100 trees). Nothing here informs any decision."""
    import copy
    tiny_cfg = copy.deepcopy(cfg)
    tiny_cfg["training"]["max_epochs"] = 1
    tiny_cfg["xgboost"]["max_trees"] = 100
    rng = np.random.default_rng(cfg["timing"]["subset_seed"] + 1)
    tiny = {}
    for role, frame in frames.items():
        surfaces = np.array(sorted(frame.surface_replicate_id.unique()))
        keep = set(rng.choice(surfaces, size=max(8, int(0.005 * len(surfaces))), replace=False))
        tiny[role] = frame[frame.surface_replicate_id.isin(keep)].reset_index(drop=True)
    tables = gate.build_tables(tiny, banks)
    out.mkdir()
    for form in sorted(gate.FORMS):
        (out / form).mkdir()
        train_bisurf(tables, banks, tiny_cfg, form, 41, out / form)
    (out / "xgb").mkdir()
    _, detail = train_xgb(tiny, banks, gate.morgan(banks), tiny_cfg, 41, out / "xgb")
    learned._write_json_atomic(out / "smoke.json", {"status": "passed", "xgb_blockwise_vs_direct_max_abs": detail["blockwise_vs_direct_max_abs"],
                                                    "xgb_base_score": detail["base_score"], "note": "tiny subsets; not for any decision"})
    print("SMOKE passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["timing", "bisurf", "xgb"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--form", choices=sorted(gate.FORMS))
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument("--tiny", action="store_true",
                        help="engineering smoke test of the whole main path on 0.5%% subsets (output dir must start with smoke_)")
    args = parser.parse_args()
    cfg = gate.load_config(CONFIG)
    output = args.output_dir.resolve()
    if not output.is_relative_to(gate.RUN) or output.exists():
        raise ValueError("Output must be a fresh directory under runs/model_improvement/public_split_gate_v1")
    if args.tiny != output.name.startswith("smoke_"):
        raise ValueError("Smoke outputs (and only smoke outputs) go to directories named smoke_*")
    if args.mode == "bisurf" and (args.form is None or not args.seeds):
        raise ValueError("bisurf mode needs --form and --seeds")
    if args.mode == "xgb" and not args.seeds:
        raise ValueError("xgb mode needs --seeds")
    if any(s not in (41, 42, 43, 44, 45) for s in args.seeds):
        raise ValueError("Only the registered seeds 41-45 are allowed")
    if args.mode == "bisurf":
        timing_file = gate.RUN / "s2_timing/timing.json"
        if not timing_file.exists():
            raise RuntimeError("S2 timing must run first")
        cfg["training"]["max_epochs"] = json.loads(timing_file.read_text())["bisurf"]["max_epochs_decided"]
    if cfg["device"] == "cuda" and args.mode != "xgb" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    bindings = gate.check_bindings()
    output.mkdir(parents=True)
    code = code_records()
    learned._write_json_atomic(output / "launch.json", {"mode": args.mode, "form": args.form, "seeds": args.seeds, "config": cfg,
                                                        "bindings": bindings, "code_sha256": code, "python": sys.version,
                                                        "torch": torch.__version__, "test_response_rows_read": 0})
    threads = cfg["xgboost"]["threads"] if args.mode == "xgb" else cfg["threads"]
    with threadpool_limits(limits=threads):
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frames, banks, tables = prepare(cfg)
        if args.tiny:
            frames = tiny_frames(frames, cfg["timing"]["subset_seed"] + 2)
            tables = gate.build_tables(frames, banks)
            cfg["training"]["max_epochs"] = 1
            cfg["xgboost"]["max_trees"] = 100
        if args.mode == "timing":
            timing(frames, banks, tables, cfg, output)
        else:
            fingerprints = gate.morgan(banks) if args.mode == "xgb" else None
            va = tables["validation"]
            for seed in args.seeds:
                name = f"bisurf_{args.form}" if args.mode == "bisurf" else "xgb_residual_d10_m1"
                out = output / f"{name}_seed_{seed}"; out.mkdir()
                print(f"START {name} seed={seed}", flush=True)
                if args.mode == "bisurf":
                    pred, detail = train_bisurf(tables, banks, cfg, args.form, seed, out)
                else:
                    pred, detail = train_xgb(frames, banks, fingerprints, cfg, seed, out)
                metrics = {"model": name, "seed": seed,
                           "validation_surface_macro_rmse": gate.surface_macro(va, pred),
                           "validation_pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), pred),
                           **{k: v for k, v in detail.items() if k != "history"}}
                pd.DataFrame(dict(source_row=va.source_rows, surface_id=va.surface_ids, target=va.target.numpy(),
                                  prediction=pred)).to_csv(out / "validation_predictions.csv", index=False)
                learned._write_json_atomic(out / "metrics.json", metrics)
                print(f"DONE {name} seed={seed} val_pooled={metrics['validation_pooled_rmse_percent']:.4f} "
                      f"val_surface_macro={metrics['validation_surface_macro_rmse']:.6f} best={detail['best_epoch']}", flush=True)
                if cfg["device"] == "cuda" and args.mode == "bisurf":
                    torch.cuda.empty_cache()
    drift = [r for r, h in code.items() if gate.sha256(gate.ROOT / r) != h]
    if drift:
        raise RuntimeError(f"Source drift during the run: {drift}")
    learned._write_json_atomic(output / "summary.json", {"status": "passed", "test_response_rows_read": 0})


if __name__ == "__main__":
    main()

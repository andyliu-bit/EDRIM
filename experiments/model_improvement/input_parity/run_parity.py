"""BiSurfGNN v2 phase 1 (input parity): +MoLFormer / +976 genes / +both on the gate's additive BiSurfGNN.

Protocol: docs/BISURFGNN_V2_DEV_PROTOCOL.md section 3 (frozen, runs/model_improvement/bisurfgnn_v2_dev/
PROTOCOL_FREEZE.json); design: docs/INPUT_PARITY_FEASIBILITY_20261013.md section 4.

Everything except the injected inputs is the PublicSplitGate-v1 additive run (run_gate.train_bisurf, copied below
with the external rows threaded through): configs/gate.yaml with the S2 max_epochs, checked equal to the configs
recorded by the seed-41/42 baseline launches; the S1 split-1 training and split-2 validation point tables
(gate_common.load_points; MDA-MB-468 is not in S1 and is refused again by the feature loader); the same graph banks,
data order, AdamW settings, surface-macro early stopping (patience 4, min_delta 1e-5), 20-epoch cap, 3 h wall cap,
determinism settings and checkpoint-reload check. Every repository module this process loads and the gate recorded
is checked byte-identical to the gate's record. Only split-1/2 tables are read; split 3 is never opened.

Modes (output directories must not exist):
  train    formal run, --arm molformer|genes976|both, --output-dir runs/model_improvement/bisurfgnn_v2_dev/phase1/<arm>_seed_<seed>
  smoke    whole main path on the gate's 0.5% tiny subsets (run_gate.tiny_frames), 1 epoch; phase1_smoke/smoke_*
  harness  --arm none --seed 41, full data, 1 epoch: epoch -1 and 0 must equal the gate seed-41 history bit for bit;
           phase1_smoke/harness_*
  timing   one full training epoch plus one validation pass; predictions discarded, no metric computed;
           phase1_smoke/timing_*

Repository root, isolated interpreter (the MoLFormer file is a torch/pickle container), no bytecode:
  .venv/Scripts/python.exe -I -B experiments/model_improvement/input_parity/run_parity.py --mode train --arm molformer --seed 41 --output-dir runs/model_improvement/bisurfgnn_v2_dev/phase1/molformer_seed_41
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/model_improvement/public_split_gate"))
sys.path.insert(0, str(ROOT / "src"))
import run_gate  # noqa: E402  (frozen gate runner; importing it only defines functions)
import gate_common as gate  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from bisurfgnn.data.batch import pad_graph_items  # noqa: E402
from bisurfgnn.data.external_features import CELL_DIM, DRUG_DIM, load_deepsynba_features  # noqa: E402
from bisurfgnn.models.compact_bisurf_parity import ParityAdditiveCompactBiSurfGNN  # noqa: E402

graph, learned, pilot = gate.graph, gate.learned, gate.pilot

V2 = ROOT / "runs/model_improvement/bisurfgnn_v2_dev"
PHASE1 = V2 / "phase1"
SMOKE = V2 / "phase1_smoke"
PROTOCOL = ROOT / "docs/BISURFGNN_V2_DEV_PROTOCOL.md"
PROTOCOL_SHA256 = "155ae4f03ceb8275f68d10f9830f5bcd193c41f70564cc138bc175840788a333"
FEATURE_DIR = gate.DATA / "data/raw/external_benchmarks/deepsynba/data/features"
ARMS = {"none": {}, "molformer": {"drug_ext_dim": DRUG_DIM}, "genes976": {"cell_ext_dim": CELL_DIM},
        "both": {"drug_ext_dim": DRUG_DIM, "cell_ext_dim": CELL_DIM}}
FORMAL_ARMS = ("molformer", "genes976", "both")
SEEDS = (41, 42, 43, 44, 45)
# gate additive baselines whose launch records bind the configuration and the code (seed -> run directory)
BASELINE_RUNS = {41: gate.RUN / "s3_additive_r2", 42: gate.RUN / "s4_bisurf_additive_a"}
HARNESS_REFERENCE = gate.RUN / "s3_additive_r2/bisurf_additive_seed_41/training_progress.json"
HARNESS_KEYS = ("epoch", "surface_macro_rmse", "pooled_rmse_percent", "train_batch_mean_loss", "improved")
PROTECTED = ("public_split_gate_v1/s5", "test_evaluated", "test_predictions", "final_predictions", "test.parquet",
             "points_test", "jaaks", "intensity", "az_sanger", "reserve")
MODE_PREFIX = {"smoke": "smoke_", "harness": "harness_", "timing": "timing_"}


def guard(path: Path) -> Path:
    text = Path(path).resolve().as_posix().lower()
    if any(token in text for token in PROTECTED):
        raise RuntimeError(f"refusing to open a protected path: {text}")
    return Path(path)


def read_json(path: Path) -> dict:
    return json.loads(guard(path).read_text(encoding="utf-8"))


# ----- bindings -------------------------------------------------------------------------------------------------
def protocol_binding() -> dict:
    digest = gate.sha256(guard(PROTOCOL))
    frozen = read_json(V2 / "PROTOCOL_FREEZE.json")["sha256"]
    if not digest == frozen == PROTOCOL_SHA256:
        raise RuntimeError("docs/BISURFGNN_V2_DEV_PROTOCOL.md differs from its freeze record")
    return dict(protocol_sha256=digest, freeze_record=str(V2 / "PROTOCOL_FREEZE.json"))


def effective_config() -> dict:
    """configs/gate.yaml with the S2-decided max_epochs, exactly as run_gate.main builds it for BiSurfGNN."""
    cfg = gate.load_config(guard(run_gate.CONFIG))
    cfg["training"]["max_epochs"] = read_json(gate.RUN / "s2_timing/timing.json")["bisurf"]["max_epochs_decided"]
    for seed, run in BASELINE_RUNS.items():
        launch = read_json(run / "launch.json")
        if launch["config"] != cfg or launch["form"] != "additive" or seed not in launch["seeds"]:
            raise RuntimeError(f"Configuration differs from the gate additive baseline launch {run.name}")
    return cfg


def loaded_repo_files() -> set[str]:
    """Repository sources executed by this process (src/ and experiments/ modules, plus the gate config)."""
    loaded = {Path(run_gate.CONFIG).resolve().relative_to(ROOT).as_posix()}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        resolved = Path(path).resolve()
        if resolved.is_relative_to(ROOT / "src") or resolved.is_relative_to(ROOT / "experiments"):
            loaded.add(resolved.relative_to(ROOT).as_posix())
    return loaded


def gate_code_binding() -> dict:
    """Every repository source file loaded by this process that the baseline launches recorded is unchanged."""
    records = [read_json(run / "launch.json")["code_sha256"] for run in BASELINE_RUNS.values()]
    loaded = loaded_repo_files()
    checked, new = [], []
    for rel in sorted(loaded):
        expected = {r[rel] for r in records if rel in r}
        if not expected:
            new.append(rel)
            continue
        if len(expected) != 1 or gate.sha256(ROOT / rel) not in expected:
            raise RuntimeError(f"{rel} differs from the gate baseline code record")
        checked.append(rel)
    other_changed = sorted(rel for r in records for rel, h in r.items()
                           if rel not in loaded and (not (ROOT / rel).exists() or gate.sha256(ROOT / rel) != h))
    return dict(loaded_files_identical_to_gate_record=len(checked), loaded_files_not_in_gate_record=new,
                gate_recorded_files_changed_but_not_loaded=other_changed)


def code_records() -> dict[str, str]:
    """run_gate.code_records() (all of src/bisurfgnn, experiments/model_improvement/*.py, gate files) plus this
    directory's scripts. Recorded at launch; drift is checked at the end for the files this process executed."""
    paths = {ROOT / rel for rel in run_gate.code_records()} | set(HERE.glob("*.py")) | set(HERE.glob("*.sh"))
    return {p.relative_to(ROOT).as_posix(): gate.sha256(p) for p in sorted(paths)}


# ----- model and external rows -----------------------------------------------------------------------------------
def build_parity_model(banks, arm: str, seed: int, device: str):
    """The gate's additive builder (gate_common.build_model) plus the arm's zero-initialised projections.

    Parent parameters must equal the gate model's for the same seed; with arm 'none' the state dict and module
    structure must be identical. The RNG is re-seeded last, so training starts from the gate's RNG state.
    """
    learned._seed_everything(seed)
    model = ParityAdditiveCompactBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim, 136,
                                           **ARMS[arm], **gate.MODEL_KWARGS)
    reference = gate.build_model(banks, "additive", seed, "cpu")      # itself checked against the frozen v2 builder
    expected, actual = reference.state_dict(), model.state_dict()
    base = list(expected)
    if list(actual)[:len(base)] != base or not all(torch.equal(expected[k], actual[k]) for k in base):
        raise RuntimeError(f"Shared parameters differ from the gate additive model (seed {seed})")
    extra = [k for k in actual if k not in expected]
    allowed = tuple(p for p, key in (("drug_ext_proj.", "drug_ext_dim"), ("cell_ext_proj.", "cell_ext_dim"))
                    if key in ARMS[arm])
    if any(not k.startswith(allowed) for k in extra) or not all(any(k.startswith(p) for k in extra) for p in allowed):
        raise RuntimeError(f"Unexpected parameters for arm {arm}: {extra}")
    if arm == "none" and repr(reference).split("(", 1)[1] != repr(model).split("(", 1)[1]:
        raise RuntimeError("Module structure differs from the gate additive model")
    for projection in (model.drug_ext_proj, model.cell_ext_proj):
        if projection is not None and (projection[-1].weight.abs().sum() != 0 or projection[-1].bias.abs().sum() != 0):
            raise RuntimeError("External projections must start at exactly zero")
    del reference
    learned._seed_everything(seed)
    return model.to(torch.device(device))


class DeviceExternal:
    """The arm's external rows on the training device, in bank order (drug rows = banks.drug_ids)."""

    def __init__(self, features, arm: str, device):
        dims = ARMS[arm]
        if dims and features is None:
            raise ValueError(f"arm {arm} needs external features")
        self.arm = arm
        self.drug = features.drug.to(device) if "drug_ext_dim" in dims else None
        self.cell = features.cell.to(device) if "cell_ext_dim" in dims else None

    def drug_rows(self, global_indices=None) -> dict:
        """Rows for the unique, sorted bank indices exactly as graph._unique_graph_batch orders its graphs."""
        if self.drug is None:
            return {}
        if global_indices is None:
            return {"drug_ext": self.drug}
        unique = torch.unique(global_indices.to(dtype=torch.int64, device="cpu"), sorted=True)
        return {"drug_ext": self.drug[unique.to(self.drug.device)]}

    def cell_rows(self, global_indices=None) -> dict:
        if self.cell is None:
            return {}
        if global_indices is None:
            return {"cell_ext": self.cell}
        unique = torch.unique(global_indices.to(dtype=torch.int64, device="cpu"), sorted=True)
        return {"cell_ext": self.cell[unique.to(self.cell.device)]}


def forward_ext(model, table, banks, ext: DeviceExternal, indices, device):
    """graph._model_forward plus the external rows of the batch's unique drugs and cells."""
    index_cpu = indices.to(dtype=torch.int64, device="cpu")
    drug_a = table.drug_a_index[index_cpu].to(dtype=torch.int64)
    drug_b = table.drug_b_index[index_cpu].to(dtype=torch.int64)
    drug_values = torch.cat([drug_a, drug_b])
    drug_graphs, drug_inverse = graph._unique_graph_batch(drug_values, banks.drug_graphs, device)
    point_count = len(index_cpu)
    cell_values = table.cell_index[index_cpu].to(dtype=torch.int64)
    cell_graphs, cell_inverse = graph._unique_graph_batch(cell_values, banks.cell_graphs, device)
    masks = torch.ones(point_count, dtype=torch.bool, device=device)
    return model(drug_graphs, cell_graphs, drug_inverse[:point_count], drug_inverse[point_count:], cell_inverse,
                 table.doses[index_cpu].to(device), table.mono_a[index_cpu].to(device),
                 table.mono_b[index_cpu].to(device), masks, masks,
                 **ext.drug_rows(drug_values), **ext.cell_rows(cell_values))


def predict_ext(model, table, banks, ext: DeviceExternal, cfg) -> np.ndarray:
    """graph.predict_graph_compact (via run_algorithm_search.predict) plus the full-bank external rows."""
    device = torch.device(cfg["device"])
    model.eval()
    predictions = []
    with torch.no_grad():
        drug_embeddings = model.encode_drugs(pad_graph_items(banks.drug_graphs).to(device), **ext.drug_rows())
        context_embeddings = model.encode_context(pad_graph_items(banks.cell_graphs).to(device), **ext.cell_rows())
        for indices in graph._batch_indices(len(table), cfg["training"]["evaluation_batch_size"], shuffle_seed=None):
            index_cpu = indices.to(dtype=torch.int64, device="cpu")
            masks = torch.ones(len(index_cpu), dtype=torch.bool, device=device)
            output = model.forward_pooled(
                drug_embeddings, context_embeddings,
                table.drug_a_index[index_cpu].to(device=device, dtype=torch.int64),
                table.drug_b_index[index_cpu].to(device=device, dtype=torch.int64),
                table.cell_index[index_cpu].to(device=device, dtype=torch.int64),
                table.doses[index_cpu].to(device), table.mono_a[index_cpu].to(device),
                table.mono_b[index_cpu].to(device), masks, masks)
            predictions.append(output.prediction.detach().cpu().numpy())
    return np.concatenate(predictions)


# ----- training (run_gate.train_epoch / train_bisurf with the external rows) -------------------------------------
def train_epoch_ext(model, optimizer, table, banks, ext, cfg, seed, epoch, device):
    model.train(); losses = []
    for idx in graph._surface_batch_indices(table, cfg["training"]["batch_surfaces"], shuffle_seed=seed * 1000003 + epoch):
        optimizer.zero_grad(set_to_none=True)
        target = table.target[idx].to(device)
        output = forward_ext(model, table, banks, ext, idx, device)
        loss = pilot.loss_equal_surface(output.prediction, target, table.surface_ids[idx.numpy()])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("Non-finite gradient")
        optimizer.step(); losses.append(float(loss.detach()))
    return float(np.mean(losses))


def train_parity(tables, banks, ext, cfg, arm, seed, out):
    """run_gate.train_bisurf with the parity model: early stopping on validation surface-macro and the wall cap."""
    device = torch.device(cfg["device"]); t = cfg["training"]
    model = build_parity_model(banks, arm, seed, cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    tr, va = tables["train"], tables["validation"]
    start = time.monotonic()
    initial = predict_ext(model, va, banks, ext, cfg)
    np.testing.assert_allclose(initial, np.minimum(va.mono_a.numpy(), va.mono_b.numpy()), atol=3e-6, rtol=3e-6)
    best = gate.surface_macro(va, initial); state, best_epoch = pilot.frozen_state(model), -1
    history = [{"epoch": -1, "surface_macro_rmse": best, "pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), initial)}]
    stale, stopped_by, last = 0, "max_epochs", 0.0
    for epoch in range(t["max_epochs"]):
        if time.monotonic() - start + last > cfg["wall_cap_seconds"]:
            stopped_by = "wall_cap"; break
        epoch_start = time.monotonic()
        loss = train_epoch_ext(model, optimizer, tr, banks, ext, cfg, seed, epoch, device)
        pred = predict_ext(model, va, banks, ext, cfg)
        score = gate.surface_macro(va, pred)
        improved = score < best - t["min_delta"]
        if improved:
            best, state, best_epoch, stale = score, pilot.frozen_state(model), epoch, 0
        else:
            stale += 1
        last = time.monotonic() - epoch_start
        history.append({"epoch": epoch, "surface_macro_rmse": score, "pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), pred),
                        "train_batch_mean_loss": loss, "wall_seconds": last, "improved": improved})
        pilot.write_history(out, history, best_epoch=best_epoch, arm=arm, device=str(device))
        print(f"PARITY {arm} seed={seed} epoch={epoch} val_surface_macro={score:.7f} best={best:.7f} ({last:.0f}s)", flush=True)
        if stale >= t["patience"]:
            stopped_by = "early_stopping"; break
    model.load_state_dict(state)
    prediction = predict_ext(model, va, banks, ext, cfg)
    torch.save(state, out / "model.pt")
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(predict_ext(model, va, banks, ext, cfg), prediction, atol=0, rtol=0)
    return prediction, {"best_epoch": best_epoch, "stopped_by": stopped_by, "epochs_run": len(history) - 1,
                        "history": history, "parameters": sum(p.numel() for p in model.parameters()),
                        "wall_seconds": time.monotonic() - start,
                        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


def timing_epoch(tables, banks, ext, cfg, arm, seed, formal_max_epochs: int) -> dict:
    """One full training epoch and one validation pass; the predictions are discarded without any metric."""
    device = torch.device(cfg["device"]); t = cfg["training"]
    model = build_parity_model(banks, arm, seed, cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    tr, va = tables["train"], tables["validation"]
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    t0 = time.monotonic(); initial = predict_ext(model, va, banks, ext, cfg); sync(); t_initial = time.monotonic() - t0
    np.testing.assert_allclose(initial, np.minimum(va.mono_a.numpy(), va.mono_b.numpy()), atol=3e-6, rtol=3e-6)
    del initial                                                         # untrained == HSA check only
    t0 = time.monotonic(); train_epoch_ext(model, optimizer, tr, banks, ext, cfg, seed, 0, device); sync()
    t_train = time.monotonic() - t0
    t0 = time.monotonic(); prediction = predict_ext(model, va, banks, ext, cfg); sync(); t_val = time.monotonic() - t0
    finite = bool(np.isfinite(prediction).all()); del prediction
    epoch = t_train + t_val
    return dict(arm=arm, seed=seed, train_points=len(tr), validation_points=len(va),
                train_epoch_seconds=t_train, validation_pass_seconds=t_val, epoch_seconds=epoch,
                initial_validation_pass_seconds=t_initial, formal_max_epochs=formal_max_epochs,
                projected_run_seconds_at_max_epochs=t_initial + formal_max_epochs * epoch + 2 * t_val,
                parameters=sum(p.numel() for p in model.parameters()),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                validation_predictions_finite=finite, accuracy_computed=False)


def harness_comparison(history: list[dict]) -> dict:
    """Epoch -1 and 0 of the all-off arm against the frozen gate seed-41 history (exact equality)."""
    reference = read_json(HARNESS_REFERENCE)["history"][:2]
    rows = []
    for mine, gate_row in zip(history[:2], reference):
        keys = [k for k in HARNESS_KEYS if k in gate_row]
        rows.append(dict(epoch=gate_row["epoch"], gate={k: gate_row[k] for k in keys},
                         this_run={k: mine.get(k) for k in keys},
                         identical=all(mine.get(k) == gate_row[k] for k in keys) and set(keys) <= set(mine)))
    return dict(reference=str(HARNESS_REFERENCE), reference_sha256=gate.sha256(HARNESS_REFERENCE), epochs=rows,
                identical=len(rows) == 2 and all(r["identical"] for r in rows))


# ----- entry point ---------------------------------------------------------------------------------------------
def validate_target(mode: str, arm: str, seed: int, output: Path) -> None:
    if seed not in SEEDS:
        raise ValueError("Only the registered seeds 41-45 are allowed")
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    if mode == "train":
        if arm not in FORMAL_ARMS or output != PHASE1 / f"{arm}_seed_{seed}":
            raise ValueError("Formal runs: --arm molformer|genes976|both and output phase1/<arm>_seed_<seed>")
        return
    if output.parent != SMOKE or not output.name.startswith(MODE_PREFIX[mode]):
        raise ValueError(f"{mode} outputs go to {SMOKE.as_posix()}/{MODE_PREFIX[mode]}*")
    if mode == "harness" and (arm != "none" or seed != 41):
        raise ValueError("The harness check compares the all-off arm with the gate's seed-41 history")


def main():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B` (torch/pickle feature file; no bytecode next to frozen sources)")
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "smoke", "harness", "timing"], default="train")
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    validate_target(args.mode, args.arm, args.seed, output)
    cfg = effective_config()
    formal_max_epochs = cfg["training"]["max_epochs"]
    if args.mode != "train":
        cfg["training"]["max_epochs"] = 1
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    protocol = protocol_binding()
    bindings = gate.check_bindings()
    gate_code = gate_code_binding()
    output.mkdir(parents=True)
    code = code_records()
    learned._write_json_atomic(output / "launch.json", {
        "protocol": "BiSurfGNN-v2-dev phase 1 (input parity)", "mode": args.mode, "arm": args.arm, "seed": args.seed,
        "arm_inputs": ARMS[args.arm], "config": cfg, "baseline_launches": {s: str(r / "launch.json") for s, r in BASELINE_RUNS.items()},
        "protocol_binding": protocol, "bindings": bindings, "gate_code_binding": gate_code, "code_sha256": code,
        "python": sys.version, "torch": torch.__version__, "test_response_rows_read": 0})
    try:
        run(args, cfg, formal_max_epochs, output)
    except BaseException as error:
        learned._write_json_atomic(output / "failure.json", {"status": "failed", "error": repr(error)})
        raise
    final_binding = gate_code_binding()                         # includes modules imported lazily during the run
    executed = loaded_repo_files()
    drift = [r for r, h in code.items() if r in executed and gate.sha256(ROOT / r) != h]
    if drift:
        raise RuntimeError(f"Source drift during the run: {drift}")
    learned._write_json_atomic(output / "summary.json", {"status": "passed", "mode": args.mode, "arm": args.arm,
                                                         "seed": args.seed, "gate_code_binding_at_end": final_binding,
                                                         "executed_repository_files": len(executed),
                                                         "executed_files_not_hash_recorded": sorted(executed - set(code)),
                                                         "test_response_rows_read": 0})


def run(args, cfg, formal_max_epochs, output):
    device = torch.device(cfg["device"])
    with threadpool_limits(limits=cfg["threads"]):
        torch.set_num_threads(cfg["threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frames, banks, tables = run_gate.prepare(cfg)
        features = load_deepsynba_features(banks.drug_ids, banks.cell_ids, frames,
                                           feature_dir=guard(FEATURE_DIR), mapping_dir=guard(gate.S1))
        learned._write_json_atomic(output / "feature_binding.json", features.record)
        if args.mode == "smoke":
            frames = run_gate.tiny_frames(frames, cfg["timing"]["subset_seed"] + 2)
            tables = gate.build_tables(frames, banks)
        ext = DeviceExternal(features if ARMS[args.arm] else None, args.arm, device)
        print(f"START parity mode={args.mode} arm={args.arm} seed={args.seed} train_points={len(tables['train'])} "
              f"validation_points={len(tables['validation'])}", flush=True)
        if args.mode == "timing":
            result = timing_epoch(tables, banks, ext, cfg, args.arm, args.seed, formal_max_epochs)
            learned._write_json_atomic(output / "timing.json", result)
            print(json.dumps(result, indent=1), flush=True)
        else:
            va = tables["validation"]
            pred, detail = train_parity(tables, banks, ext, cfg, args.arm, args.seed, output)
            metrics = {"model": f"bisurf_additive_parity_{args.arm}", "arm": args.arm, "seed": args.seed, "mode": args.mode,
                       "validation_surface_macro_rmse": gate.surface_macro(va, pred),
                       "validation_pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), pred),
                       **{k: v for k, v in detail.items() if k != "history"}}
            if args.mode == "smoke":
                metrics["note"] = "tiny subsets, 1 epoch: engineering check only, not for any decision"
            pd.DataFrame(dict(source_row=va.source_rows, surface_id=va.surface_ids, target=va.target.numpy(),
                              prediction=pred)).to_csv(output / "validation_predictions.csv", index=False)
            learned._write_json_atomic(output / "metrics.json", metrics)
            print(f"DONE parity arm={args.arm} seed={args.seed} val_pooled={metrics['validation_pooled_rmse_percent']:.4f} "
                  f"val_surface_macro={metrics['validation_surface_macro_rmse']:.6f} best={detail['best_epoch']}", flush=True)
            if args.mode == "harness":
                comparison = harness_comparison(detail["history"])
                learned._write_json_atomic(output / "harness_check.json", comparison)
                print(f"HARNESS identical_to_gate_seed41_history={comparison['identical']}", flush=True)
                if not comparison["identical"]:
                    raise RuntimeError("All-off harness differs from the gate history (see harness_check.json)")


if __name__ == "__main__":
    main()

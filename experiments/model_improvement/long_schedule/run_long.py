"""BiSurfGNN v2 phase 2, candidate C: long training schedule (DEVIATIONS.md D4, registered before any run).

Model, data, inputs and selection rule are exactly the gate additive BiSurfGNN (input-parity arm 'none'); only the
schedule changes: up to 60 epochs, per-epoch cosine learning-rate decay 3e-4 -> 3e-6, no early stopping, best
validation surface-macro epoch selected (as in the gate), 8 h wall cap. Epochs -1 and 0 must equal the gate's
same-seed history exactly (epoch 0 uses the gate's learning rate), otherwise the run aborts.

This file only imports run_parity.py (used by the running phase-1 batch); it never modifies it.
Split 1 trains, split 2 evaluates; split 3 is never opened.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/long_schedule/run_long.py --mode train --seed 41
  .venv/Scripts/python.exe -I -B experiments/model_improvement/long_schedule/run_long.py --mode smoke --seed 41
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "input_parity"))
import run_parity as rp  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch is imported)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

gate, run_gate, learned, pilot = rp.gate, rp.run_gate, rp.learned, rp.pilot

OUT = rp.V2 / "phase2_long"
SMOKE = rp.V2 / "phase2_long_smoke"
SEEDS = (41, 42)
SCHEDULE = {"max_epochs": 60, "lr_start": 3e-4, "lr_end": 3e-6, "early_stopping": False, "wall_cap_seconds": 28800}
GATE_HISTORY = {41: gate.RUN / "s3_additive_r2/bisurf_additive_seed_41/training_progress.json",
                42: gate.RUN / "s4_bisurf_additive_a/bisurf_additive_seed_42/training_progress.json"}
CHECK_KEYS = ("epoch", "surface_macro_rmse", "pooled_rmse_percent", "train_batch_mean_loss", "improved")


def check_against_gate(history: list[dict], seed: int) -> dict:
    reference = rp.read_json(rp.guard(GATE_HISTORY[seed]))["history"][:2]
    rows = []
    for mine, ref in zip(history[:2], reference):
        keys = [k for k in CHECK_KEYS if k in ref]
        rows.append(dict(epoch=ref["epoch"], identical=all(mine.get(k) == ref[k] for k in keys) and set(keys) <= set(mine),
                         gate={k: ref[k] for k in keys}, this_run={k: mine.get(k) for k in keys}))
    return dict(reference=str(GATE_HISTORY[seed]), epochs=rows, identical=len(rows) == 2 and all(r["identical"] for r in rows))


def train_long(tables, banks, ext, cfg, seed, out, check: bool):
    """run_parity.train_parity (arm 'none') with the registered long schedule."""
    device = torch.device(cfg["device"]); t = cfg["training"]
    model = rp.build_parity_model(banks, "none", seed, cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    if t["learning_rate"] != SCHEDULE["lr_start"]:
        raise RuntimeError("Gate learning rate differs from the registered schedule start")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULE["max_epochs"], eta_min=SCHEDULE["lr_end"])
    tr, va = tables["train"], tables["validation"]
    start = time.monotonic()
    initial = rp.predict_ext(model, va, banks, ext, cfg)
    np.testing.assert_allclose(initial, np.minimum(va.mono_a.numpy(), va.mono_b.numpy()), atol=3e-6, rtol=3e-6)
    best = gate.surface_macro(va, initial); state, best_epoch = pilot.frozen_state(model), -1
    history = [{"epoch": -1, "surface_macro_rmse": best, "pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), initial)}]
    stopped_by, last, gate_check = "max_epochs", 0.0, None
    for epoch in range(t["max_epochs"]):
        if time.monotonic() - start + last > cfg["wall_cap_seconds"]:
            stopped_by = "wall_cap"; break
        epoch_start = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        loss = rp.train_epoch_ext(model, optimizer, tr, banks, ext, cfg, seed, epoch, device)
        pred = rp.predict_ext(model, va, banks, ext, cfg)
        score = gate.surface_macro(va, pred)
        improved = score < best - t["min_delta"]
        if improved:
            best, state, best_epoch = score, pilot.frozen_state(model), epoch
        scheduler.step()
        last = time.monotonic() - epoch_start
        history.append({"epoch": epoch, "surface_macro_rmse": score, "pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), pred),
                        "train_batch_mean_loss": loss, "wall_seconds": last, "improved": improved, "learning_rate": lr})
        pilot.write_history(out, history, best_epoch=best_epoch, arm="none", device=str(device))
        print(f"LONG seed={seed} epoch={epoch} lr={lr:.2e} val_surface_macro={score:.7f} "
              f"val_pooled={history[-1]['pooled_rmse_percent']:.4f} best={best:.7f} ({last:.0f}s)", flush=True)
        if check and epoch == 0:
            gate_check = check_against_gate(history, seed)
            learned._write_json_atomic(out / "gate_check.json", gate_check)
            if not gate_check["identical"]:
                raise RuntimeError("Epochs -1/0 differ from the gate history (see gate_check.json)")
            print("GATE_CHECK identical epochs -1 and 0", flush=True)
    model.load_state_dict(state)
    prediction = rp.predict_ext(model, va, banks, ext, cfg)
    torch.save(state, out / "model.pt")
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(rp.predict_ext(model, va, banks, ext, cfg), prediction, atol=0, rtol=0)
    return prediction, {"best_epoch": best_epoch, "stopped_by": stopped_by, "epochs_run": len(history) - 1,
                        "history": history, "gate_check": gate_check, "wall_seconds": time.monotonic() - start,
                        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


def main():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B`")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "smoke"], required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = ap.parse_args()
    output = (OUT if args.mode == "train" else SMOKE) / f"{'' if args.mode == 'train' else 'smoke_'}long_seed_{args.seed}"
    if output.exists():
        raise SystemExit(f"Output directory already exists: {output}")
    cfg = rp.effective_config()                              # gate config, checked against the baseline launches
    cfg["training"]["max_epochs"] = SCHEDULE["max_epochs"] if args.mode == "train" else 2
    cfg["wall_cap_seconds"] = SCHEDULE["wall_cap_seconds"]
    cfg["schedule"] = dict(SCHEDULE, registered="runs/model_improvement/bisurfgnn_v2_dev/DEVIATIONS.md D4")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    protocol = rp.protocol_binding()
    bindings = gate.check_bindings()
    gate_code = rp.gate_code_binding()
    output.mkdir(parents=True)
    code = dict(rp.code_records(), **{"experiments/model_improvement/long_schedule/run_long.py": gate.sha256(Path(__file__))})
    learned._write_json_atomic(output / "launch.json", {
        "protocol": "BiSurfGNN-v2-dev phase 2 candidate C (long schedule)", "mode": args.mode, "seed": args.seed,
        "config": cfg, "gate_history": str(GATE_HISTORY[args.seed]), "protocol_binding": protocol, "bindings": bindings,
        "gate_code_binding": gate_code, "code_sha256": code, "python": sys.version, "torch": torch.__version__,
        "test_response_rows_read": 0})
    try:
        with threadpool_limits(limits=cfg["threads"]):
            torch.set_num_threads(cfg["threads"])
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            frames, banks, tables = run_gate.prepare(cfg)
            if args.mode == "smoke":
                frames = run_gate.tiny_frames(frames, cfg["timing"]["subset_seed"] + 2)
                tables = gate.build_tables(frames, banks)
            ext = rp.DeviceExternal(None, "none", torch.device(cfg["device"]))
            print(f"START long mode={args.mode} seed={args.seed} train_points={len(tables['train'])} "
                  f"validation_points={len(tables['validation'])}", flush=True)
            va = tables["validation"]
            pred, detail = train_long(tables, banks, ext, cfg, args.seed, output, check=args.mode == "train")
            metrics = {"model": "bisurf_additive_long_schedule", "seed": args.seed, "mode": args.mode,
                       "validation_surface_macro_rmse": gate.surface_macro(va, pred),
                       "validation_pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), pred),
                       **{k: v for k, v in detail.items() if k != "history"}}
            if args.mode == "smoke":
                metrics["note"] = "tiny subsets, 2 epochs: engineering check only, not for any decision"
            pd.DataFrame(dict(source_row=va.source_rows, surface_id=va.surface_ids, target=va.target.numpy(),
                              prediction=pred)).to_csv(output / "validation_predictions.csv", index=False)
            learned._write_json_atomic(output / "metrics.json", metrics)
            print(f"DONE long seed={args.seed} val_pooled={metrics['validation_pooled_rmse_percent']:.4f} "
                  f"best_epoch={detail['best_epoch']}", flush=True)
    except BaseException as error:
        learned._write_json_atomic(output / "failure.json", {"status": "failed", "error": repr(error)})
        raise
    executed = rp.loaded_repo_files()
    drift = [r for r, h in code.items() if r in executed and gate.sha256(rp.ROOT / r) != h]
    if drift:
        raise RuntimeError(f"Source drift during the run: {drift}")
    learned._write_json_atomic(output / "summary.json", {"status": "passed", "mode": args.mode, "seed": args.seed,
                                                         "test_response_rows_read": 0})


if __name__ == "__main__":
    main()

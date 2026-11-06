"""BiSurfGNN v2 phase 2, combined candidate (DEVIATIONS.md D6): arm 'both' (MoLFormer 768-d + 976 genes, input parity)
with the D4 long schedule. Model, data and inputs are exactly the phase-1 'both' arm; only the schedule changes:
up to 60 epochs, per-epoch cosine learning-rate decay 3e-4 -> 3e-6, no early stopping, best validation surface-macro
epoch selected, 8 h wall cap. Epochs -1 and 0 must equal the phase-1 'both' same-seed history exactly (epoch 0 uses
the same learning rate), otherwise the run aborts.

Imports run_parity.py and run_long.py; neither is modified. Split 1 trains, split 2 evaluates; split 3 is never opened.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/long_schedule/run_long_both.py --seed 41
  .venv/Scripts/python.exe -I -B experiments/model_improvement/long_schedule/run_long_both.py --seed 41 --smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "input_parity"))
import run_parity as rp  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch is imported)
import run_long  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

gate, run_gate, learned, pilot = rp.gate, rp.run_gate, rp.learned, rp.pilot

ARM = "both"
SCHEDULE = run_long.SCHEDULE
OUT = rp.V2 / "phase2_long_both"
REFERENCE = {s: rp.PHASE1 / f"both_seed_{s}/training_progress.json" for s in rp.SEEDS}


def check_first_epochs(history: list[dict], seed: int) -> dict:
    reference = rp.read_json(rp.guard(REFERENCE[seed]))["history"][:2]
    rows = []
    for mine, ref in zip(history[:2], reference):
        keys = [k for k in run_long.CHECK_KEYS if k in ref]
        rows.append(dict(epoch=ref["epoch"], identical=set(keys) <= set(mine) and all(mine[k] == ref[k] for k in keys),
                         reference={k: ref[k] for k in keys}, this_run={k: mine.get(k) for k in keys}))
    return dict(reference=str(REFERENCE[seed]), epochs=rows, identical=len(rows) == 2 and all(r["identical"] for r in rows))


def train_long_both(tables, banks, ext, cfg, seed, out, check: bool, constant_initial: bool = False):
    """run_long.train_long with the 'both' parity model and its external rows.

    constant_initial: ablation arm A (run_ablation.py B11) has no HSA anchor, so its untrained prediction must equal the
    learnable constant b instead of the HSA reference; the default keeps the HSA check."""
    device = torch.device(cfg["device"]); t = cfg["training"]
    model = rp.build_parity_model(banks, ARM, seed, cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if t["learning_rate"] != SCHEDULE["lr_start"]:
        raise RuntimeError("Gate learning rate differs from the registered schedule start")
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULE["max_epochs"], eta_min=SCHEDULE["lr_end"])
    tr, va = tables["train"], tables["validation"]
    start = time.monotonic()
    initial = rp.predict_ext(model, va, banks, ext, cfg)
    if constant_initial:
        np.testing.assert_array_equal(initial, np.full_like(initial, model.output_bias.item()))
    else:
        np.testing.assert_allclose(initial, np.minimum(va.mono_a.numpy(), va.mono_b.numpy()), atol=3e-6, rtol=3e-6)
    best = gate.surface_macro(va, initial); state, best_epoch = pilot.frozen_state(model), -1
    history = [{"epoch": -1, "surface_macro_rmse": best, "pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), initial)}]
    stopped_by, last, first_check = "max_epochs", 0.0, None
    for epoch in range(t["max_epochs"]):
        if time.monotonic() - start + last > SCHEDULE["wall_cap_seconds"]:
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
        pilot.write_history(out, history, best_epoch=best_epoch, arm=ARM, device=str(device))
        print(f"LONG_BOTH seed={seed} epoch={epoch} lr={lr:.2e} val_surface_macro={score:.7f} "
              f"val_pooled={history[-1]['pooled_rmse_percent']:.4f} best={best:.7f} ({last:.0f}s)", flush=True)
        if check and epoch == 0:
            first_check = check_first_epochs(history, seed)
            if not first_check["identical"]:
                learned._write_json_atomic(out / "first_epochs_check.json", first_check)
                raise RuntimeError("Epochs -1/0 differ from the phase-1 'both' history (see first_epochs_check.json)")
            print("CHECK epochs -1 and 0 identical to phase-1 'both'", flush=True)
    model.load_state_dict(state)
    prediction = rp.predict_ext(model, va, banks, ext, cfg)
    torch.save(state, out / "model.pt")
    return prediction, {"best_epoch": best_epoch, "stopped_by": stopped_by, "epochs_run": len(history) - 1,
                        "first_epochs_identical_to_phase1": None if first_check is None else first_check["identical"],
                        "wall_seconds": time.monotonic() - start,
                        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


def main():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B` (torch/pickle feature file)")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=rp.SEEDS, required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny subsets, 2 epochs, no first-epoch check")
    a = ap.parse_args()
    out = OUT / f"{'smoke_' if a.smoke else ''}long_both_seed_{a.seed}"
    if out.exists():
        raise SystemExit(f"Output directory already exists: {out}")
    cfg = rp.effective_config()                              # gate config, checked against the baseline launches
    cfg["training"]["max_epochs"] = 2 if a.smoke else SCHEDULE["max_epochs"]
    cfg["schedule"] = dict(SCHEDULE, registered="runs/model_improvement/bisurfgnn_v2_dev/DEVIATIONS.md D6")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    out.mkdir(parents=True)
    device = torch.device(cfg["device"])
    with threadpool_limits(limits=cfg["threads"]):
        torch.set_num_threads(cfg["threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frames, banks, tables = run_gate.prepare(cfg)
        features = rp.load_deepsynba_features(banks.drug_ids, banks.cell_ids, frames,
                                              feature_dir=rp.guard(rp.FEATURE_DIR), mapping_dir=rp.guard(gate.S1))
        if a.smoke:
            frames = run_gate.tiny_frames(frames, cfg["timing"]["subset_seed"] + 2)
            tables = gate.build_tables(frames, banks)
        ext = rp.DeviceExternal(features, ARM, device)
        va = tables["validation"]
        print(f"START long_both seed={a.seed} smoke={a.smoke} train_points={len(tables['train'])} "
              f"validation_points={len(va)}", flush=True)
        pred, detail = train_long_both(tables, banks, ext, cfg, a.seed, out, check=not a.smoke)
        metrics = {"model": "bisurf_additive_parity_both_long_schedule", "arm": ARM, "seed": a.seed, "smoke": a.smoke,
                   "validation_surface_macro_rmse": gate.surface_macro(va, pred),
                   "validation_pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), pred),
                   **detail, "config": cfg, "torch": torch.__version__, "test_response_rows_read": 0}
        pd.DataFrame(dict(source_row=va.source_rows, surface_id=va.surface_ids, target=va.target.numpy(),
                          prediction=pred)).to_csv(out / "validation_predictions.csv", index=False)
        learned._write_json_atomic(out / "metrics.json", metrics)
        print(f"DONE long_both seed={a.seed} val_pooled={metrics['validation_pooled_rmse_percent']:.4f} "
              f"best_epoch={detail['best_epoch']} stopped_by={detail['stopped_by']}", flush=True)


if __name__ == "__main__":
    main()

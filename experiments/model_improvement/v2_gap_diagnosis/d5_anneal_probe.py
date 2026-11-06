"""V2 gap diagnosis, step D5: training-budget probe (VALIDATION only; diagnostic, not a candidate result).

Question: is the gate's BiSurfGNN limited by its schedule (constant LR 3e-4, stopped at epochs 13-19)?
Probe: start from a saved gate checkpoint (best validation state), continue training for EPOCHS epochs with the
gate's loss, batching, AdamW settings and clipping, but with the learning rate decayed linearly from 3e-4 to 0
per step. Validation (pooled and surface-macro RMSE) after every epoch.
Contrast already in the gate logs: after their best epoch these runs trained 4 more epochs at constant LR without
improving (that is why early stopping fired). A fresh AdamW state is used (the gate saved model weights only).

Each run has a wall cap of 28 minutes (task rule: no training run over 30 minutes). Nothing in the gate dir is
written; outputs go to runs/model_improvement/v2_gap_diagnosis/anneal_probe/.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/v2_gap_diagnosis/d5_anneal_probe.py --seeds 42 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public_split_gate"))
import run_gate  # noqa: E402  (frozen gate runner, imported read only; its main() is not called)
import gate_common as gate  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from gate_common import graph, pilot  # noqa: E402

OUT_DEFAULT = gate.ROOT / "runs/model_improvement/v2_gap_diagnosis/anneal_probe"
SOURCES = {41: gate.RUN / "s3_additive_r2/bisurf_additive_seed_41", 42: gate.RUN / "s4_bisurf_additive_a/bisurf_additive_seed_42",
           43: gate.RUN / "s4_bisurf_additive_b/bisurf_additive_seed_43", 44: gate.RUN / "s4_bisurf_additive_a/bisurf_additive_seed_44",
           45: gate.RUN / "s4_bisurf_additive_b/bisurf_additive_seed_45"}
EPOCHS = 4
WALL_CAP = 28 * 60
LR0 = 3e-4


def main():
    if not sys.flags.dont_write_bytecode:
        raise SystemExit("run with -B (no bytecode next to the frozen gate code)")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = gate.load_config(run_gate.CONFIG)
    t = cfg["training"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(cfg["threads"])
    device = torch.device(cfg["device"])
    t_setup = time.time()
    frames, banks, tables = run_gate.prepare(cfg)          # split 1/2 point tables only
    del frames
    tr, va = tables["train"], tables["validation"]
    target = va.target.numpy()
    setup_s = time.time() - t_setup
    report = {"step": "D5", "test_rows_read": 0, "epochs": EPOCHS, "lr0": LR0, "schedule": "linear to 0 per step",
              "optimizer": "fresh AdamW(lr0, weight_decay 1e-4), clip 1.0, batch 128 surfaces, equal-surface MSE",
              "setup_seconds": round(setup_s, 1), "runs": {}}
    for seed in args.seeds:
        src = SOURCES[seed]
        out = args.out / f"bisurf_additive_seed_{seed}"
        out.mkdir()
        start = time.monotonic()
        model = gate.build_model(banks, "additive", seed, cfg["device"])
        model.load_state_dict(torch.load(src / "model.pt", weights_only=True, map_location="cpu"))
        saved = pd.read_csv(src / "validation_predictions.csv")
        base = run_gate.predict(model, va, banks, cfg)
        diff = float(np.max(np.abs(base - saved.prediction.to_numpy(float))))
        if diff > 1e-5:
            raise RuntimeError(f"seed {seed}: saved checkpoint does not reproduce its validation predictions ({diff})")
        history = [{"epoch": 0, "pooled_rmse_percent": gate.pooled_rmse_percent(target, base),
                    "surface_macro_rmse": gate.surface_macro(va, base), "lr_end": LR0}]
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR0, weight_decay=t["weight_decay"])
        steps_per_epoch = int(np.ceil(len(np.unique(tr.surface_ids)) / t["batch_surfaces"]))
        total = EPOCHS * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.0, 1.0 - s / total))
        best = (history[0]["surface_macro_rmse"], 0, base.copy())
        stopped = "completed"
        for epoch in range(1, EPOCHS + 1):
            if time.monotonic() - start > WALL_CAP - 330:     # one epoch ~ 270-300 s
                stopped = "wall_cap"
                break
            t_ep = time.monotonic()
            model.train(); losses = []
            for idx in graph._surface_batch_indices(tr, t["batch_surfaces"], shuffle_seed=seed * 1000003 + 1000 + epoch):
                optimizer.zero_grad(set_to_none=True)
                output = graph._model_forward(model, tr, banks, idx, device)
                loss = pilot.loss_equal_surface(output.prediction, tr.target[idx].to(device), tr.surface_ids[idx.numpy()])
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"])
                optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
            pred = run_gate.predict(model, va, banks, cfg)
            row = {"epoch": epoch, "pooled_rmse_percent": gate.pooled_rmse_percent(target, pred),
                   "surface_macro_rmse": gate.surface_macro(va, pred), "train_batch_mean_loss": float(np.mean(losses)),
                   "lr_end": optimizer.param_groups[0]["lr"], "wall_seconds": round(time.monotonic() - t_ep, 1)}
            history.append(row)
            np.save(out / f"validation_prediction_epoch{epoch}.npy", pred.astype(np.float32))
            if row["surface_macro_rmse"] < best[0]:
                best = (row["surface_macro_rmse"], epoch, pred.copy())
            print(f"seed {seed} anneal epoch {epoch}: val pooled {row['pooled_rmse_percent']:.4f} macro "
                  f"{row['surface_macro_rmse']:.6f} ({row['wall_seconds']:.0f}s)", flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=1) + "\n", encoding="utf-8")
        report["runs"][f"BS_add_s{seed}"] = dict(source=str(src), reproduce_max_abs_diff=diff, stopped_by=stopped,
                                                 history=history, best_epoch_by_macro=best[1],
                                                 final_pooled=history[-1]["pooled_rmse_percent"],
                                                 start_pooled=history[0]["pooled_rmse_percent"],
                                                 wall_seconds=round(time.monotonic() - start, 1))
        del model, optimizer
        torch.cuda.empty_cache()
    with (args.out / "d5_report.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()

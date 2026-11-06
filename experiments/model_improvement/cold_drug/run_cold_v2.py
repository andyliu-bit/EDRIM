"""D11 (card D): BiSurfGNN v2 (final D7 configuration: 'both' input parity + 60-epoch cosine schedule) on the new-drug
split. Same model, inputs and schedule as run_long_both.py; only the data roles change (cold_split.py): train on the
'train' role, select the epoch on 'cold_val', then predict 'cold_test' with the selected state. No metric is computed
on cold_test here; score_cold.py scores all methods once after every run has finished.

  --arm both    real MoLFormer + 976-gene rows (seeds 41-45)
  --arm random  the random-vector control of run_random.py (same RNG seed 20261014; seeds 41-43)

  .venv/Scripts/python.exe -I -B experiments/model_improvement/cold_drug/run_cold_v2.py --arm both --seed 41 [--smoke]

D16 (card H): --partition YYYYMMDD runs the same code on an eval-paper partition: split
runs/eval_paper/cold_drug_partitions/p<partition>/split.json, output p<partition>/ours_both_seed_<s> there (arm both
only). --out-root DIR (smoke only) writes DIR/p<partition>/smoke_ours_both_seed_<s> instead. Without --partition
everything is D11 as before.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/cold_drug/run_cold_v2.py --arm both --seed 41 --partition 20261018

D17: --ablation B11 trains ablation arm A (run_ablation.patch_builder('B11') + constant_initial, exactly as
run_ablation.main) on the D11 split, output cold_drug/ablation_B11_seed_<s>. cold_test is neither predicted nor scored
(no cold_test file). Smoke needs --out-root DIR (writes DIR/smoke_ablation_B11_seed_<s>, never runs/).

  .venv/Scripts/python.exe -I -B experiments/model_improvement/cold_drug/run_cold_v2.py --arm both --seed 41 --ablation B11
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "input_parity"))
sys.path.insert(0, str(HERE.parent / "long_schedule"))
sys.path.insert(0, str(HERE.parent / "random_control"))
import run_parity as rp  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch is imported)
import run_long_both as rlb  # noqa: E402
import cold_split  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

gate, run_gate, learned = rp.gate, rlb.run_gate, rp.learned
OUT = cold_split.OUT
PARTITIONS = HERE.parents[2] / "runs/eval_paper/cold_drug_partitions"   # D16


def main():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B`")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("both", "random"), required=True)
    ap.add_argument("--seed", type=int, choices=rp.SEEDS, required=True)
    ap.add_argument("--smoke", action="store_true", help="0.5%% subsets, 2 epochs")
    ap.add_argument("--partition", type=int, choices=(20261018, 20261019, 20261020, 20261021), help="D16 partition")
    ap.add_argument("--out-root", type=Path, help="D16/D17 smoke only: output root instead of the runs/ folder")
    ap.add_argument("--ablation", choices=("B11",), help="D17: run_ablation arm B11 (A) on the D11 split, cold_val only")
    a = ap.parse_args()
    if a.partition and a.arm != "both" or a.out_root and not ((a.partition or a.ablation) and a.smoke):
        ap.error("--partition needs --arm both; --out-root needs --partition or --ablation, and --smoke")
    if a.ablation and (a.partition or a.arm != "both" or a.smoke and not a.out_root):
        ap.error("--ablation needs the D11 split (no --partition), --arm both, and --out-root with --smoke")
    split = PARTITIONS / f"p{a.partition}/split.json" if a.partition else cold_split.SPLIT
    root = (a.out_root or PARTITIONS) / f"p{a.partition}" if a.partition else a.out_root or OUT
    name = f"ablation_{a.ablation}" if a.ablation else f"{'ours' if a.partition else 'v2'}_{a.arm}"
    out = root / f"{'smoke_' if a.smoke else ''}{name}_seed_{a.seed}"
    if out.exists():
        raise SystemExit(f"Output directory already exists: {out}")
    if a.partition:
        print(f"D16 partition={a.partition} split={split} out={out}", flush=True)
    cfg = rp.effective_config()
    cfg["training"]["max_epochs"] = 2 if a.smoke else rlb.SCHEDULE["max_epochs"]
    cfg["schedule"] = dict(rlb.SCHEDULE, registered="DEVIATIONS.md D6 schedule; data roles " + (f"D16 p{a.partition}" if a.partition else "D11"))
    if a.arm == "random":
        import run_random
        rp.load_deepsynba_features = run_random.random_features
    out.mkdir(parents=True)
    device = torch.device(cfg["device"])
    with threadpool_limits(limits=cfg["threads"]):
        torch.set_num_threads(cfg["threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frames = cold_split.load_frames(split)
        if a.partition:   # same check as the eval-paper XGBoost runner: roles match the frozen split counts
            counts = json.loads(split.read_text(encoding="utf-8"))["counts"]
            if {r: len(f) for r, f in frames.items()} != {r: v["points"] for r, v in counts.items()}:
                raise RuntimeError("Frames do not match the partition's split.json counts")
        if a.smoke:
            frames = run_gate.tiny_frames(frames, cfg["timing"]["subset_seed"] + 2)
        banks = gate.build_banks(frames)
        features = rp.load_deepsynba_features(banks.drug_ids, banks.cell_ids,
                                              {"train": frames["train"], "validation": frames["cold_val"]},
                                              feature_dir=rp.guard(rp.FEATURE_DIR), mapping_dir=rp.guard(gate.S1))
        tables = {"train": learned.build_compact_point_table(frames["train"], banks, role="train", fold=0, monotherapy_upper_bound=3.0),
                  "validation": learned.build_compact_point_table(frames["cold_val"], banks, role="validation", fold=0, monotherapy_upper_bound=3.0)}
        if a.ablation:   # D17: no cold_test table; patch as run_ablation.main does (rp.build_parity_model -> B11 builder)
            sys.path.insert(0, str(HERE.parent / "ablation"))
            import run_ablation
            run_ablation.patch_builder(a.ablation, banks, tables)
            test_table = None
        else:
            test_table = learned.build_compact_point_table(frames["cold_test"], banks, role="test", fold=0, monotherapy_upper_bound=3.0)
        ext = rp.DeviceExternal(features, "both", device)
        print(f"START cold v2 arm={a.arm} seed={a.seed} train={len(tables['train'])} cold_val={len(tables['validation'])} "
              f"cold_test={'not built (D17)' if test_table is None else len(test_table)}", flush=True)
        val_pred, detail = rlb.train_long_both(tables, banks, ext, cfg, a.seed, out, check=False,
                                               constant_initial=bool(a.ablation))
        model = rp.build_parity_model(banks, "both", a.seed, cfg["device"])   # patched builder too under --ablation
        model.load_state_dict(torch.load(out / "model.pt", map_location=device, weights_only=True))
        again = rp.predict_ext(model, tables["validation"], banks, ext, cfg)
        if np.max(np.abs(np.asarray(again) - np.asarray(val_pred))) > 1e-6:
            raise RuntimeError("Reloaded model does not reproduce the cold_val predictions")
        test_pred = None if test_table is None else rp.predict_ext(model, test_table, banks, ext, cfg)
        for role, frame, pred in (("cold_val", frames["cold_val"], val_pred), ("cold_test", frames["cold_test"], test_pred)):
            if pred is not None:
                np.savez_compressed(out / f"{role}_predictions.npz", source=frame.source_combodrugseq.to_numpy(),
                                    prediction_percent=100.0 * np.asarray(pred, float))
        va = tables["validation"]
        metrics = {"model": "bisurf_v2_cold_drug", "arm": a.arm, "seed": a.seed, "smoke": a.smoke,
                   "cold_val_surface_macro_rmse": gate.surface_macro(va, val_pred),
                   "cold_val_pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), val_pred),
                   **detail, "cold_test_scored_here": False, "split3_rows_read": 0,
                   **({"partition": a.partition, "registered": "DEVIATIONS.md D16"} if a.partition else {}),
                   **({"ablation": a.ablation, "registered": "DEVIATIONS.md D17"} if a.ablation else {}),
                   "finished_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
        print(f"DONE cold v2 arm={a.arm} seed={a.seed} cold_val_pooled={metrics['cold_val_pooled_rmse_percent']:.4f} "
              f"best_epoch={detail['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()

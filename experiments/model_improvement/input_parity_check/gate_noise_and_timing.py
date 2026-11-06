"""Input-parity feasibility, step 3 (read-only, validation split only): timing and noise of the frozen gate runs,
used to size the parity experiment and to set what counts as a meaningful validation gain.

Reads the gate's S3/S4 BiSurfGNN and XGBoost run folders (training_progress.json, metrics.json,
validation_predictions.csv) and two ID columns of the S1 validation point table. Nothing under s5/ and no test
file is opened; nothing is written (stdout only). The gate is not re-run or re-evaluated.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -I experiments/model_improvement/input_parity_check/gate_noise_and_timing.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if not sys.flags.isolated:
    raise SystemExit("run with `python -I`")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/model_improvement/public_split_gate_v1"
BISURF = {"anchored_41": "s3_anchored_r2/bisurf_anchored_seed_41", "additive_41": "s3_additive_r2/bisurf_additive_seed_41",
          "additive_42": "s4_bisurf_additive_a/bisurf_additive_seed_42", "additive_44": "s4_bisurf_additive_a/bisurf_additive_seed_44",
          "additive_43": "s4_bisurf_additive_b/bisurf_additive_seed_43", "additive_45": "s4_bisurf_additive_b/bisurf_additive_seed_45"}
ADDITIVE = ["additive_41", "additive_42", "additive_43", "additive_44", "additive_45"]
WINDOW = range(10, 18)                 # epochs present in every gate run (shortest run: epochs 0-17)
BOOT, BOOT_SEED = 2000, 20261009       # same resampling scheme as the gate (pairs, with replacement)


def guard(path: Path) -> Path:
    text = path.resolve().as_posix()
    if "/s5" in text or "test" in path.name.lower() or "TEST_EVALUATED" in text:
        raise SystemExit(f"refusing to open {text}")
    return path


def run_summary(folder: Path) -> dict:
    progress = json.loads(guard(folder / "training_progress.json").read_text(encoding="utf-8"))
    metrics = json.loads(guard(folder / "metrics.json").read_text(encoding="utf-8"))
    hist = {h["epoch"]: h for h in progress["history"]}
    epochs = [e for e in sorted(hist) if e >= 0]
    wall = [hist[e]["wall_seconds"] for e in epochs]
    pooled = {e: hist[e]["pooled_rmse_percent"] for e in epochs}
    return dict(best_epoch=metrics["best_epoch"], stopped_by=metrics["stopped_by"], epochs_run=len(epochs),
                selected_pooled_rmse=metrics["validation_pooled_rmse_percent"],
                window_mean_pooled_rmse=float(np.mean([pooled[e] for e in WINDOW])),
                last_epoch_pooled_rmse=pooled[epochs[-1]], min_epoch_pooled_rmse=min(pooled.values()),
                epoch_wall_seconds=dict(median=float(np.median(wall)), min=float(min(wall)), max=float(max(wall))),
                run_wall_seconds=metrics["wall_seconds"], parameters=metrics["parameters"],
                peak_gpu_gb=round(metrics["peak_gpu_allocated_bytes"] / 1e9, 2))


def detectable(sd: float, base_mean: float, n_base: int = 5) -> list[dict]:
    rows = []
    for n_new in (1, 2, 3, 5):
        se = sd * np.sqrt(1 / n_new + 1 / n_base)
        pct = 100 * se / base_mean
        rows.append(dict(n_new_seeds=n_new, n_base_seeds=n_base, sd_of_difference_rmse=round(se, 4),
                         sd_of_difference_pct=round(pct, 3), one_sided_5pct_threshold_pct=round(1.645 * pct, 2),
                         false_positive_rate_at_minus_2pct=round(float(stats.norm.cdf(-2.0 / pct)), 3),
                         power_at_minus_2pct_threshold={f"true_{e}pct": round(float(stats.norm.cdf((e - 2.0) / pct)), 3)
                                                        for e in (2, 3, 4, 5)}))
    return rows


def pair_bootstrap(runs: list[str]) -> dict:
    ids = pd.read_parquet(guard(RUN / "s1/points_validation.parquet"), columns=["matrix_key", "pair_id"])
    sse, frames = {}, {}
    for name in runs:
        f = pd.read_csv(guard(RUN / BISURF[name] / "validation_predictions.csv"))
        if not (f.source_row.to_numpy() - 10_000_000 == np.arange(len(ids))).all() or not (f.surface_id.to_numpy() == ids.matrix_key.to_numpy()).all():
            raise SystemExit(f"row order differs from S1 for {name}")
        frames[name] = f
        sse[name] = ((f.prediction - f.target) ** 2).groupby(ids.pair_id.to_numpy()).sum()
    count = ids.groupby("pair_id").size().reindex(sse[runs[0]].index).to_numpy(float)
    matrix = np.stack([sse[n].to_numpy() for n in runs])                          # runs x pairs
    rng = np.random.default_rng(BOOT_SEED)
    weights = np.stack([np.bincount(rng.integers(0, len(count), len(count)), minlength=len(count)) for _ in range(BOOT)])
    rmse = 100 * np.sqrt((weights @ matrix.T) / (weights @ count)[:, None])          # BOOT x runs
    full = 100 * np.sqrt(matrix.sum(1) / count.sum())
    diffs = {f"{a}-{b}": rmse[:, i] - rmse[:, j] for (i, a), (j, b) in combinations(enumerate(runs), 2)}
    err = np.stack([(frames[n].prediction - frames[n].target).to_numpy() for n in runs])
    corr = np.corrcoef(err)
    mean_two = rmse[:, :2].mean(1) - rmse[:, 2:].mean(1)
    return dict(pairs=int(len(count)), resamples=BOOT, seed=BOOT_SEED,
                full_data_rmse=dict(zip(runs, np.round(full, 4).tolist())),
                single_run_bootstrap_sd=round(float(rmse.std(0).mean()), 4),
                paired_seed_difference_full_data=dict((k, round(float(full[i] - full[j]), 4))
                                                      for k, (i, j) in zip(diffs, combinations(range(len(runs)), 2))),
                paired_seed_difference_bootstrap_sd=dict(median=round(float(np.median([v.std() for v in diffs.values()])), 4),
                                                         max=round(float(max(v.std() for v in diffs.values())), 4)),
                mean_of_41_42_minus_mean_of_43_45=dict(full=round(float(full[:2].mean() - full[2:].mean()), 4),
                                                       bootstrap_sd=round(float(mean_two.std()), 4)),
                point_error_correlation_between_seeds=dict(min=round(float(corr[np.triu_indices(len(runs), 1)].min()), 3),
                                                           median=round(float(np.median(corr[np.triu_indices(len(runs), 1)])), 3)))


def main():
    runs = {k: run_summary(RUN / v) for k, v in BISURF.items()}
    sel = np.array([runs[k]["selected_pooled_rmse"] for k in ADDITIVE])
    win = np.array([runs[k]["window_mean_pooled_rmse"] for k in ADDITIVE])
    xgb = sorted(json.loads(guard(p).read_text(encoding="utf-8"))["validation_pooled_rmse_percent"]
                 for p in (RUN / "s4_xgb_r2").glob("xgb_*/metrics.json"))
    report = dict(
        runs=runs,
        additive_5_seeds=dict(selected=dict(mean=round(float(sel.mean()), 4), sd=round(float(sel.std(ddof=1)), 4)),
                              window_10_17_mean=dict(mean=round(float(win.mean()), 4), sd=round(float(win.std(ddof=1)), 4)),
                              selection_optimism_pct=round(float(100 * (win.mean() - sel.mean()) / sel.mean()), 3),
                              epochs_capped=[k for k in ADDITIVE if runs[k]["stopped_by"] == "max_epochs"]),
        xgb_validation_rmse=[round(v, 4) for v in xgb],
        detectable_effect_selected=detectable(float(sel.std(ddof=1)), float(sel.mean())),
        detectable_effect_window=detectable(float(win.std(ddof=1)), float(win.mean())),
        pair_bootstrap_validation=pair_bootstrap(ADDITIVE),
        wall_hours_per_run=dict(min=round(min(r["run_wall_seconds"] for r in runs.values()) / 3600, 2),
                                max=round(max(r["run_wall_seconds"] for r in runs.values()) / 3600, 2)),
        guards=dict(test_rows_read=0, files_written=0),
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

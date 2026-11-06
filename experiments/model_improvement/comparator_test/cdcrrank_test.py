"""CDCR-Rank (adapted), seeds 41-45 at the hybrid weight chosen on split 2, scored once on the NCI test split
(docs/CDCRRANK_ADAPTATION_CARD_20261103.md; test-use ledger row 21; marker CDCRRANK_TEST_EVALUATED.marker). The same
read completes DeepDRP (adapted) from its saved test predictions, without retraining: results without pralatrexate and
the synergy metrics row 20 did not compute. Values already reported for EDRIM, DeepSynBa (seeds 10-12), XGBoost + ID,
HSA and DeepDRP are recomputed and compared with the paper; outputs are written even if a comparison fails, and the
script then exits non-zero.

Definitions as in score_t_batch (T3) and synergy_check: pooled combination-cell RMSE; Delta of a vs b = a / b - 1 with
a drug-pair bootstrap per subset (2,000 draws, seed 20261009); Bliss and HSA excess with the measured edges of each
test matrix as the reference, synergy_check's pair bootstrap (seed 20261013, pairs in order of first appearance).
16-cell error of CDCR-Rank: control from the split-1 lookup, monotherapy and combination cells predicted.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/comparator_test/cdcrrank_test.py --lam 0.5 --dry-run
  .venv/Scripts/python.exe -I -B experiments/model_improvement/comparator_test/cdcrrank_test.py --lam 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "synergy_check"), str(HERE.parent / "cdcrrank_adapted")]
import common  # noqa: E402  (imports run_parity first: CUBLAS_WORKSPACE_CONFIG is set before torch)
import cdcrrank as cr  # noqa: E402
import synergy_check as sc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

dd, eg = cr.dd, common.eg
V2, SEEDS = common.V2, common.SEEDS
OUT, MARKER = V2 / "cdcrrank_test", V2 / "CDCRRANK_TEST_EVALUATED.marker"
DSB_ROOT = common.gate.DATA / "runs/audit_revision/deepsynba_repro_v2_20261012"
DSB_RUN = {10: DSB_ROOT / "full", 11: DSB_ROOT / "full_seed11", 12: DSB_ROOT / "full_seed12"}
DSB_VAL = dd.ROOT / "runs/model_improvement/v2_gap_diagnosis/deepsynba_val"
DSB_VAL_STAND_IN = {10: "full_epoch484_selected", 11: "pilot_epoch047", 12: "full_last"}    # dry run only
D7RUN = {s: V2 / f"phase2_long_both/long_both_seed_{s}" for s in SEEDS}
DDRP = {s: dd.RUNS / f"lr0.0003_seed{s}" for s in SEEDS}
OURS, CDCR, DDP, DSB, XID, HSA = "bisurf_v2", "cdcrrank", "deepdrp", "deepsynba_v2", "xgb_identity", "lookup_hsa"
COMPARE = ((OURS, CDCR), (CDCR, DSB), (CDCR, HSA), (CDCR, XID), (OURS, DDP), (OURS, DSB), (OURS, XID), (OURS, HSA))
# paper values recomputed here: Table 3 (RMSE, Delta in % with interval) and Table S-synergy (Bliss)
REPORTED_RMSE = {"all": {OURS: 8.241, DSB: 9.655, XID: 10.060, DDP: 10.597, HSA: 12.009},
                 "excluding_pralatrexate": {OURS: 8.198, DSB: 9.157, XID: 9.113, HSA: 10.909}}
REPORTED_DELTA = {"all": {(OURS, DSB): (-14.64, -18.61, -11.02), (OURS, XID): (-18.08, -22.57, -13.60),
                          (OURS, DDP): (-22.23, -25.78, -18.80), (OURS, HSA): (-31.38, -35.51, -27.41)},
                  "excluding_pralatrexate": {(OURS, DSB): (-10.47, -11.79, -9.13), (OURS, XID): (-10.04, -11.70, -8.37),
                                             (OURS, HSA): (-24.85, -26.90, -22.74)}}
REPORTED_SYN = {"all": {OURS: dict(pearson=0.808, spearman=0.735, precision_top5=0.595, precision_top1=0.828,
                                   precision_top10=0.611, slope_bliss=0.676, slope_hsa=0.664, rmse=4.888),
                        DDP: dict(pearson=0.580, precision_top5=0.386)},
                "excluding_pralatrexate": {OURS: dict(pearson=0.741, spearman=0.727, precision_top5=0.530,
                                                      precision_top1=0.444, precision_top10=0.581, slope_bliss=0.591,
                                                      slope_hsa=0.568, rmse=4.838)}}


# ---------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------
def check_order(frame) -> int:
    n = len(frame) // 9
    if not (len(frame) == 9 * n and (frame.row_i.to_numpy() == np.tile(np.repeat([1, 2, 3], 3), n)).all()
            and (frame.col_j.to_numpy() == np.tile([1, 2, 3], 3 * n)).all()
            and (frame.matrix_key.to_numpy() == np.repeat(frame.matrix_key.to_numpy()[::9], 9)).all()):
        raise RuntimeError("Expected 9 consecutive rows per matrix in (row, column) order")
    return n


def cdcr_predictions(frame, runs: dict, device) -> dict:
    """{seed: (n, 4, 4) percent}: control from the split-1 lookup, monotherapy and combination cells predicted."""
    n, out = check_order(frame), {}
    codes = np.repeat(np.arange(n), 9)
    for s, run in runs.items():
        model, state = cr.load_model(run / "model_selected.pt", device)
        cellx, genes = cr.cell_features(state["cells"])
        if genes != state["genes"]:
            raise RuntimeError("Cell features differ from training")
        M = dd.Matrices(frame, {d: i for i, d in enumerate(state["drugs"])}, {c: i for i, c in enumerate(state["cells"])},
                        tuple(state["dose_range"]), state["control_percent"], device, codes=codes)
        p = cr.predict(model, dd.drug_tokens(state["drugs"]).to(device), cellx.to(device), M,
                       nodes=torch.arange(1, 16, device=device))
        g = np.empty((n, 4, 4))
        g[:, 0, 0] = state["control_percent"]
        g.reshape(n, 16)[:, 1:] = p.cpu().numpy()
        out[s] = g
    return out


def pretest_alignment() -> np.ndarray:
    """Point IDs of every saved test prediction file agree (IDs only; no prediction or response is read)."""
    ids = np.load(V2 / "test_eval/test_combination_predictions_bisurf_v2.npz", allow_pickle=False)["point_ids"]
    for path in [V2 / f"xgb_identity_final/seed_{s}/test_predictions.npz" for s in SEEDS] + \
                [DDRP[s] / "test_predictions.npz" for s in SEEDS]:
        if not np.array_equal(np.load(path, allow_pickle=False)["point_ids"], ids):
            raise RuntimeError(f"{path}: point IDs differ from EDRIM's test predictions")
    for run in DSB_RUN.values():
        if not (run / "test_predictions.npz").exists():
            raise RuntimeError(f"{run}: no DeepSynBa test predictions")
    return ids


def load_test_inputs(cdcr_runs, device):
    frame, matrices = common.load_test()                         # the one read of the split-3 responses
    ids = frame.source_combodrugseq.to_numpy()
    if not np.array_equal(frame.matrix_key.to_numpy()[::9], np.array([k for k, _, _ in matrices])):
        raise RuntimeError("frame and matrices disagree")

    def npz(path, key="prediction"):
        z = np.load(path, allow_pickle=False)
        if not np.array_equal(z["point_ids"], ids):
            raise RuntimeError(f"{path}: not aligned with the test frame")
        return np.asarray(z[key], float)
    edrim = V2 / "test_eval/test_combination_predictions_bisurf_v2.npz"
    comb = {OURS: {s: npz(edrim, f"seed_{s}") for s in SEEDS},
            XID: {s: npz(V2 / f"xgb_identity_final/seed_{s}/test_predictions.npz") for s in SEEDS},
            DDP: {s: npz(DDRP[s] / "test_predictions.npz") for s in SEEDS}}
    dsb16, default = {}, eg.DSB
    try:
        for s, run in DSB_RUN.items():
            eg.DSB = run
            dsb16[s] = eg.deepsynba_combination(matrices)[0]
    finally:
        eg.DSB = default
    truth = np.stack([m for _, m, _ in matrices])
    lookup16 = np.stack([p for _, _, p in matrices])
    return frame, truth, lookup16, comb, dsb16, cdcr_predictions(frame, cdcr_runs, device)


def load_validation_inputs(cdcr_runs, device):
    """Dry run on split 2: same code path; DeepSynBa stand-ins as in score_t_batch, no comparison with the paper."""
    frame = pd.read_parquet(dd.S1 / "points_validation.parquet")
    n, keys = check_order(frame), frame.matrix_key.to_numpy()[::9]
    dsb16, truth = {}, None
    for s, name in DSB_VAL_STAND_IN.items():
        z = np.load(DSB_VAL / f"{name}_val_predictions.npz", allow_pickle=False)
        pos = pd.Index(z["matrix_key"]).get_indexer(keys)
        if (pos < 0).any():
            raise RuntimeError(f"{name}: validation matrices missing")
        dsb16[s], truth = z["predictions"].astype(float)[pos], z["truth"].astype(float)[pos]
    if np.abs(truth[:, 1:, 1:].reshape(-1) - frame.response_percent.to_numpy(float)).max() > 1e-3:
        raise RuntimeError("DeepSynBa validation truth differs from the point table")
    lookup16 = truth.copy()
    lookup16[:, 0, 0] = dd.control_percent()
    lookup16[:, 1:, 0] = frame.mono_a_percent.to_numpy(float).reshape(n, 3, 3)[:, :, 0]
    lookup16[:, 0, 1:] = frame.mono_b_percent.to_numpy(float).reshape(n, 3, 3)[:, 0, :]
    comb = {OURS: {}, XID: {}, DDP: {}}
    for s in SEEDS:
        f = pd.read_csv(D7RUN[s] / "validation_predictions.csv")
        if not ((f.surface_id.to_numpy() == frame.matrix_key.to_numpy()).all()
                and np.abs(f.target.to_numpy(float) * 100.0 - frame.response_percent.to_numpy(float)).max() < 1e-4):
            raise RuntimeError("EDRIM validation predictions not aligned")
        comb[OURS][s] = f.prediction.to_numpy(float) * 100.0
        comb[XID][s] = np.load(V2 / f"xgb_identity_final/seed_{s}/validation_predictions.npz")["prediction"] * 100.0
        z = np.load(DDRP[s] / "validation_predictions.npz")
        if not (z["matrix_key"] == frame.matrix_key.to_numpy(str)).all():
            raise RuntimeError("DeepDRP validation predictions not aligned")
        comb[DDP][s] = z["prediction"].astype(float)
    return frame, truth, lookup16, comb, dsb16, cdcr_predictions(frame, cdcr_runs, device)


# ---------------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------------
def rmse_part(frame, comb, keep) -> dict:
    """score_t_batch.subset_stats + compare: per-seed RMSE, seed means and Delta with the subset's pair bootstrap."""
    sub, y = frame[keep].reset_index(drop=True), frame.response_percent.to_numpy(float)[keep]
    sse, per_seed, n = {}, {}, None
    for m, seeds in comb.items():
        for s, p in seeds.items():
            p = np.asarray(p, float)[keep]
            sse[(m, s)], n, _ = eg.pair_sse(sub, p)
            per_seed[(m, s)] = float(np.sqrt(np.mean((p - y) ** 2)))
    point = {m: float(np.mean([per_seed[(m, s)] for s in seeds])) for m, seeds in comb.items()}
    boots = eg.bootstrap(sse, n, list(comb))
    delta = {}
    for a, b in COMPARE:
        d = 100.0 * (boots[a] - boots[b]) / boots[b]
        delta[f"{a}_vs_{b}"] = dict(percent=100.0 * (point[a] - point[b]) / point[b],
                                    ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    return dict(cells=int(keep.sum()), pairs=int(len(n)), rmse=point, delta=delta,
                per_seed={m: {s: per_seed[(m, s)] for s in seeds} for m, seeds in comb.items()},
                seed_sd={m: float(np.std([per_seed[(m, s)] for s in seeds], ddof=1)) for m, seeds in comb.items()
                         if len(seeds) > 1})


def synergy_part(frame, truth, comb, mask) -> dict:
    """Bliss / HSA excess against each matrix's measured edges; EDRIM minus each method with synergy_check's bootstrap."""
    n = len(truth)
    ref = sc.references(truth[:, 1:, 0], truth[:, 0, 1:])
    observed = sc.matrix_scores(ref, frame.response_percent.to_numpy(float).reshape(n, 3, 3))
    obs = {x: observed[x][mask] for x in sc.SCORES}
    derived = {f"{m}|{s}": sc.matrix_scores(ref, np.asarray(p, float).reshape(n, 3, 3))
               for m in (OURS, CDCR, DDP) for s, p in comb[m].items()}
    series = {k: {x: v[x][mask] for x in sc.SCORES} for k, v in derived.items()}
    pts = {k: {x: sc.point_metrics(v[x], obs[x], False) for x in sc.SCORES} for k, v in series.items()}
    codes, pairs = pd.factorize(frame.pair_id.to_numpy()[::9][mask])
    res = sc.run_bootstrap(series, obs, codes, len(pairs))
    out = dict(matrices=int(mask.sum()), pairs=int(len(pairs)), k_top5=sc.k_frac(int(mask.sum()), 0.05), methods={},
               edrim_minus=dict())
    for m in (OURS, CDCR, DDP):
        keys = [f"{m}|{s}" for s in comb[m]]
        bl = {k: [pts[key]["Bliss"][k] for key in keys] for k in ("pearson", "spearman", "precision_top1", "precision_top5",
                                                                  "precision_top10", "rmse")}
        bl["slope_bliss"] = [pts[key]["Bliss"]["slope_pred_on_obs"] for key in keys]
        bl["slope_hsa"] = [pts[key]["HSA"]["slope_pred_on_obs"] for key in keys]
        out["methods"][m] = {k: dict(mean=float(np.mean(v)), seeds=v) for k, v in bl.items()}
        if m == OURS:
            continue
        diff = {}
        for metric in ("pearson", "spearman", "precision_top5", "rmse"):
            mean = {mm: np.mean([res[(f"{mm}|{s}", "Bliss", metric)] for s in comb[mm]], axis=0) for mm in (OURS, m)}
            d = mean[OURS] - mean[m]
            diff[metric] = dict(difference=out["methods"][OURS][metric]["mean"] - out["methods"][m][metric]["mean"],
                                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        out["edrim_minus"][m] = diff
    return out


def all16(truth, cdcr16) -> dict:
    rm = {s: float(np.sqrt(np.mean((p - truth) ** 2))) for s, p in cdcr16.items()}
    r = {s: float(np.corrcoef(p.ravel(), truth.ravel())[0, 1]) for s, p in cdcr16.items()}
    mono = np.zeros((4, 4), bool)
    mono[0, 1:] = mono[1:, 0] = True
    mr = {s: float(np.sqrt(np.mean((p[:, mono] - truth[:, mono]) ** 2))) for s, p in cdcr16.items()}
    return dict(rmse_mean=float(np.mean(list(rm.values()))), rmse_seeds=rm, pearson_range=[min(r.values()), max(r.values())],
                pearson_seeds=r, monotherapy_cells_rmse_mean=float(np.mean(list(mr.values()))),
                note="control from the split-1 lookup; monotherapy and combination cells predicted")


def check_reported(result) -> list[str]:
    bad = []
    for sub, vals in REPORTED_RMSE.items():
        for m, v in vals.items():
            if abs(result["rmse"][sub]["rmse"][m] - v) > 6e-4:
                bad.append(f"RMSE {sub} {m}: {result['rmse'][sub]['rmse'][m]:.4f} vs {v}")
    for sub, vals in REPORTED_DELTA.items():
        for (a, b), (p, lo, hi) in vals.items():
            got = result["rmse"][sub]["delta"][f"{a}_vs_{b}"]
            if max(abs(got["percent"] - p), abs(got["ci95"][0] - lo), abs(got["ci95"][1] - hi)) > 6e-3:
                bad.append(f"Delta {sub} {a} vs {b}: {got} vs {(p, lo, hi)}")
    for sub, vals in REPORTED_SYN.items():
        for m, metrics in vals.items():
            for k, v in metrics.items():
                got = result["synergy"][sub]["methods"][m][k]["mean"]
                if abs(got - v) > 6e-4:
                    bad.append(f"synergy {sub} {m} {k}: {got:.4f} vs {v}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--suffix", default="_lrpat0_pat10", help="run-folder suffix of the reported configuration (C7)")
    ap.add_argument("--dry-run", action="store_true", help="split 2 instead of the test split; no marker")
    args = ap.parse_args()
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with python -I -B")
    t0 = time.perf_counter()
    runs = {s: cr.RUNS / f"lam{args.lam:g}_seed{s}{args.suffix}" for s in SEEDS}
    runs = {s: r for s, r in runs.items() if (r / "model_selected.pt").exists()} if args.dry_run else runs
    missing = [s for s, r in runs.items() if not (r / "model_selected.pt").exists()]
    if missing or not runs:
        raise SystemExit(f"seeds without a selected model: {missing or 'all'}")
    device = torch.device("cuda")
    if args.dry_run:
        out = V2 / "cdcrrank_test_dryrun"
        frame, truth, lookup16, comb, dsb16, cdcr16 = load_validation_inputs(runs, device)
    else:
        if MARKER.exists() or OUT.exists():
            raise SystemExit("CDCR-Rank test set already scored")
        pretest_alignment()
        with MARKER.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ledger_row": 21,
                                     "launch_card": "docs/CDCRRANK_ADAPTATION_CARD_20261103.md"}) + "\n")
        out = OUT
        frame, truth, lookup16, comb, dsb16, cdcr16 = load_test_inputs(runs, device)
    n = check_order(frame)
    comb[CDCR] = {s: g[:, 1:, 1:].reshape(-1) for s, g in cdcr16.items()}
    comb[DSB] = {s: g[:, 1:, 1:].reshape(-1) for s, g in dsb16.items()}
    comb[HSA] = {0: np.minimum(frame.mono_a_percent, frame.mono_b_percent).to_numpy(float)}
    has = ((frame.deepsynba_drug_a.to_numpy()[::9] == sc.EXCLUDED_DRUG)
           | (frame.deepsynba_drug_b.to_numpy()[::9] == sc.EXCLUDED_DRUG))
    masks = {"all": np.ones(n, bool), "excluding_pralatrexate": ~has}
    result = dict(role="validation dry run" if args.dry_run else "test (ledger row 21)", lam=args.lam,
                  run_folders=[str(r.relative_to(cr.RUNS)) for r in runs.values()],
                  seeds={CDCR: list(runs), DSB: list(dsb16)}, matrices=n,
                  rmse={k: rmse_part(frame, comb, np.repeat(m, 9)) for k, m in masks.items()},
                  all16_cdcrrank=all16(truth, cdcr16),
                  synergy={k: synergy_part(frame, truth, comb, m) for k, m in masks.items()},
                  bootstrap=dict(rmse=dict(draws=eg.DRAWS, seed=eg.BOOT_SEED, unit="drug pair, per subset"),
                                 synergy=dict(draws=sc.BOOT_N, seed=sc.BOOT_SEED, unit="drug pair, first-appearance order")),
                  pair_order_sorted_equals_first_appearance=bool(np.array_equal(
                      pd.factorize(frame.pair_id.to_numpy()[::9])[1], np.sort(pd.unique(frame.pair_id.to_numpy()[::9])))))
    result["checks_failed"] = [] if args.dry_run else check_reported(result)
    result["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    out.mkdir(parents=True, exist_ok=args.dry_run)
    (out / "test_evaluation.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    if not args.dry_run:
        for s, g in cdcr16.items():
            np.savez_compressed(runs[s] / "test_predictions16.npz", prediction16=g,
                                matrix_key=frame.matrix_key.to_numpy(str)[::9])
    show = {k: {m: round(v, 3) for m, v in result["rmse"][k]["rmse"].items()} for k in masks}
    print(json.dumps(dict(rmse=show, delta={k: {c: [round(d["percent"], 2)] + [round(x, 2) for x in d["ci95"]]
                                                for c, d in result["rmse"][k]["delta"].items()} for k in masks},
                          checks_failed=result["checks_failed"], runtime_seconds=result["runtime_seconds"]), indent=1))
    if result["checks_failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

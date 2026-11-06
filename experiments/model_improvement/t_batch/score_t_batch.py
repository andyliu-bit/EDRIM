"""D19 (card J, docs/T_BATCH_CARD_20261025.md): the T batch, the last read of the NCI public test (split 3, S54 set).

One script, one test contact. Stored predictions are scored; the only inference is the five Ours-lite models (D18).
  T1  main comparison: Ours (D7) 5-seed mean vs DeepSynBa v2 3-seed mean (seeds 10, 11, 12; 13 and 14 failed and are
      not used). Rule unchanged from D7: pair bootstrap (2000 draws, seed 20261009) 95% upper bound < 0 and >= 4 Ours
      seeds below the comparator mean. The D7 1-seed result is recomputed alongside (evaluate_v2.score).
  T2  sensitivity only, never changes a verdict: drug (vertex) bootstrap and leave-one-drug-out jackknife for Ours vs
      DeepSynBa (3 seeds and seed 10), Ours vs XGBoost+identity, and the Bliss-score Pearson (vs DeepSynBa, vs XGB+id).
      Vertex bootstrap: draw the D test drugs with replacement (2000 draws, seed 20261009); a drug pair (a, b) enters
      with weight m_a * m_b, m_d = times drug d was drawn, so pairs with an undrawn drug drop out. Both members of a
      pair come from one drug population, which makes this the standard node bootstrap for dyadic data (Snijders and
      Borgatti 1999; consistent for jointly exchangeable arrays, Davezies, D'Haultfoeuille and Guyonvarch 2021).
      Cells are not resampled: they stay aggregated inside each pair, the D7 unit.
  T3  every method with 'all' and 'excluding pralatrexate' columns; the subset gets its own pair bootstrap, exactly
      as evaluate_v2.score does for its sensitivity analysis.
  T4  gate XGBoost synergy rows (synergy_check.analyse: the D9 pre-declared metrics, pair bootstrap seed 20261013).
  T5  Ours-lite seeds 41-45 (framework variant): RMSE, vs D7 and vs XGBoost+identity.

Reused, not re-implemented: evaluate_gate (evaluate, bootstrap, pair_sse, deepsynba_combination), evaluate_v2 (score,
interval, rmse, check_validation, v2_runs), comparator_test/common.load_test, synergy_check (analyse and the score
functions), run_ablation.load_lite. Every number recomputed for an already-scored method is checked against its
stored result (exact to 1e-12) and the headline numbers against the documents' rounding; a mismatch exits non-zero
after all outputs are written.

  --dry-run --out DIR   rehearsal on validation (split 2): LITE inference, all bootstraps, synergy, tables and the
                        validation-document checks. DeepSynBa seeds 11/12 have no validation predictions, so two other
                        seed-10 checkpoints stand in for them (pipeline check only). Never reads split 3, no marker.
                        It also rehearses the test-side preconditions: run states, file hashes and the ID alignment
                        of every stored test prediction file (ID members only, no prediction or truth array).
  (no flag)             test: refuses if T_BATCH_TEST_SCORED.marker exists; checks every input and re-predicts the
                        stored LITE validation predictions first; then creates the marker exclusively, reads split 3
                        once and writes runs/model_improvement/bisurfgnn_v2_dev/t_batch/.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/t_batch/score_t_batch.py --dry-run --out <dir>
  .venv/Scripts/python.exe -I -B experiments/model_improvement/t_batch/score_t_batch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MI = HERE.parent
for _sub in ("comparator_test", "synergy_check", "ablation"):
    sys.path.insert(0, str(MI / _sub))
import common  # noqa: E402  (imports run_parity first: CUBLAS_WORKSPACE_CONFIG is set before torch)
import run_ablation as ra  # noqa: E402
import synergy_check as sc  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

rp, gate, eg = common.rp, common.gate, common.eg
learned = rp.learned
ev = gate.load_module("evaluate_v2_for_t_batch", MI / "final_eval/evaluate_v2.py")
V2, ROOT = rp.V2, gate.ROOT
OUT_TEST, MARKER = V2 / "t_batch", V2 / "T_BATCH_TEST_SCORED.marker"
SEEDS, DSB_SEEDS = (41, 42, 43, 44, 45), (10, 11, 12)
DSB_ROOT = gate.DATA / "runs/audit_revision/deepsynba_repro_v2_20261012"
DSB_RUN = {10: DSB_ROOT / "full", 11: DSB_ROOT / "full_seed11", 12: DSB_ROOT / "full_seed12"}
DSB_VAL = ROOT / "runs/model_improvement/v2_gap_diagnosis/deepsynba_val"
# dry run only: 484 = seed 10's selected checkpoint (the real seed 10); 047 = pilot; 'last' is byte-identical to 484
DSB_VAL_STAND_IN = {10: "full_epoch484_selected", 11: "pilot_epoch047", 12: "full_last"}
D7RUN = {s: V2 / f"phase2_long_both/long_both_seed_{s}" for s in SEEDS}
LITE = {s: V2 / f"ablation/LITE_seed_{s}" for s in SEEDS}
XID, DDP = V2 / "xgb_identity_final", V2 / "ddprism_measured_mono"
CKR = ROOT / "runs/model_improvement/combokr2_original_v1/public_split"
D9 = V2 / "synergy_check_v2"
OURS = "bisurf_v2"
METHODS = ("lookup_hsa", "lookup_bliss", "combokr2", "ddprism_measured_mono", "xgb", "xgb_identity", "bisurf",
           "deepsynba_v2_s10", "deepsynba_v2", "lite", "bisurf_v2", "bisurf_v2_ensemble")
LABEL = {"lookup_hsa": "查表 HSA", "lookup_bliss": "查表 Bliss", "combokr2": "comboKR 2.0",
         "ddprism_measured_mono": "DD-PRiSM 实测单药版", "xgb": "XGBoost 残差（gate）", "xgb_identity": "XGBoost＋身份码",
         "bisurf": "原 BiSurfGNN（gate 加性）", "deepsynba_v2_s10": "DeepSynBa v2 种子 10（原主比较）",
         "deepsynba_v2": "DeepSynBa v2 种子 10–12 均值", "lite": "Ours-lite（框架精简变体）", "bisurf_v2": "Ours（D7）",
         "bisurf_v2_ensemble": "Ours 5 种子集成"}
T2_RMSE = (("bisurf_v2", "deepsynba_v2"), ("bisurf_v2", "deepsynba_v2_s10"), ("bisurf_v2", "xgb_identity"))
SUBSETS = ("all", "excluding_pralatrexate")
# synergy series groups ("single model" = mean of the per-seed metrics, the D9 convention)
SYN_GROUPS = {"BS": [f"BS_s{s}" for s in SEEDS], "XI": [f"XI_s{s}" for s in SEEDS], "GX": [f"GX_s{s}" for s in SEEDS],
              "DSB3": ["DSB", "DSB_s11", "DSB_s12"]}
SYN_COMPARISONS = {
    "C1 BS - DSB(observed ref)": ("BS", "DSB"), "C2 BS_ens5 - DSB(observed ref)": ("BS_ens5", "DSB"),
    "C3 BS - DSB(own mono)": ("BS", "DSB_own"), "C4 BS_ens5 - DSB(own mono)": ("BS_ens5", "DSB_own"),
    "C5 BS - XGB_identity": ("BS", "XI"), "C6 XGB_identity - DSB(observed ref)": ("XI", "DSB"),
    "C7 BS - XGB_gate": ("BS", "GX"), "C8 XGB_gate - DSB(observed ref)": ("GX", "DSB"),
    "C9 BS - DSB3(observed ref)": ("BS", "DSB3")}
# D9 stored names -> this script's names (the D9 'XGB' series is the gate XGBoost on validation, XGB+identity on test)
D9_RENAME = {"validation": {"series": {**{f"XGB_s{s}": f"GX_s{s}" for s in SEEDS}, "XGB_ens5": "GX_ens5"},
                            "comparison": {"C5 BS - XGB": "C7 BS - XGB_gate",
                                           "C6 XGB - DSB(observed ref)": "C8 XGB_gate - DSB(observed ref)"}},
             "test": {"series": {**{f"XGB_s{s}": f"XI_s{s}" for s in SEEDS}, "XGB_ens5": "XI_ens5"}, "comparison": {}}}


def rj(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def forbidden(*_a, **_k):
    raise PermissionError("the dry run never reads split 3")


class Checks:
    def __init__(self):
        self.rows = []

    def near(self, name, value, ref, tol=1e-12):
        v, r = (np.nan if x is None else float(x) for x in (value, ref))
        diff = 0.0 if (np.isnan(v) and np.isnan(r)) else abs(v - r)
        self.rows.append(dict(check=name, value=v, reference=r, abs_diff=diff, tol=tol, ok=bool(diff <= tol)))

    def doc(self, name, value, ref, digits):
        self.near(f"{name} [doc, {digits} dp]", value, ref, 0.5 * 10 ** -digits + 1e-9)

    def same(self, name, ok):
        self.rows.append(dict(check=name, value=None, reference=None, abs_diff=0.0 if ok else np.nan, tol=0.0, ok=bool(ok)))

    def deep(self, name, mine, ref, tol=1e-12):
        if isinstance(ref, dict):
            for k, r in ref.items():
                self.deep(f"{name}.{k}", (mine or {}).get(k), r, tol)
        elif isinstance(ref, list):
            for i, r in enumerate(ref):
                self.deep(f"{name}[{i}]", mine[i] if mine is not None and i < len(mine) else None, r, tol)
        elif ref is None or isinstance(ref, (bool, str)):
            self.same(f"{name} == {ref!r}", mine == ref)
        else:
            self.near(name, mine, ref, tol)

    def table(self, name, ref_path, mine, keys, rename=None, tol=1e-12):
        ref = pd.read_csv(ref_path, float_precision="round_trip")
        for col, mapping in (rename or {}).items():
            ref[col] = ref[col].replace(mapping)
        cols = [c for c in ref.columns if c not in keys and pd.api.types.is_numeric_dtype(ref[c])]
        m = ref.merge(mine, on=keys, how="left", suffixes=("", "_mine"))
        worst = 0.0
        for c in cols:
            a, b = m[c].to_numpy(float), m[f"{c}_mine"].to_numpy(float)
            d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
            worst = max(worst, float(np.max(np.where(np.isnan(d), np.inf, d))))
        self.near(f"{name}: {len(ref)} rows x {len(cols)} numeric columns, max abs diff", worst, 0.0, tol)

    @property
    def ok(self):
        return all(r["ok"] for r in self.rows)


# ---------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------
def check_order(frame, n):
    if not (len(frame) == 9 * n and (frame.row_i.to_numpy() == np.tile(np.repeat([1, 2, 3], 3), n)).all()
            and (frame.col_j.to_numpy() == np.tile([1, 2, 3], 3 * n)).all()
            and (frame.matrix_key.to_numpy() == np.repeat(frame.matrix_key.to_numpy()[::9], 9)).all()):
        raise RuntimeError("Expected 9 consecutive rows per matrix in (row, column) order")


def csv_pred(path, frame):
    """Saved validation predictions (ratio) -> percent, with synergy_check's alignment checks."""
    f = pd.read_csv(path)
    if not ((f.source_row.to_numpy() == frame.source_combodrugseq.to_numpy()).all()
            and (f.surface_id.to_numpy() == frame.matrix_key.to_numpy()).all()
            and np.max(np.abs(f.target.to_numpy(float) * 100.0 - frame.response_percent.to_numpy(float))) < 1e-4):
        raise RuntimeError(f"{path}: not aligned with the validation table")
    return f.prediction.to_numpy(float) * 100.0


def npz_pred(path, key, ids):
    z = np.load(path, allow_pickle=False)
    if not np.array_equal(z["point_ids"], ids):
        raise RuntimeError(f"{path}: not aligned with the test frame")
    return np.asarray(z[key], float)


def combokr(role, keys):
    k = rj(CKR / "validation/selection.json")["selected_iters"]          # chosen on split 2 (D10 supplement)
    z = np.load(CKR / role / f"pred_iters{k}.npz", allow_pickle=False)
    if not np.array_equal(z["matrix_key"], keys.astype(str)):
        raise RuntimeError(f"comboKR {role} predictions are not aligned")
    pred = z["pred"].astype(float).reshape(-1)                           # (matrices, 9) row-major = frame order
    if not np.isfinite(pred).all():
        raise RuntimeError("non-finite comboKR predictions")
    return pred


def lookup_comb(frame):
    a, b = frame.mono_a_percent.to_numpy(float), frame.mono_b_percent.to_numpy(float)
    return {"lookup_hsa": {0: np.minimum(a, b)}, "lookup_bliss": {0: a * b / 100.0}}


def input_files(dry: bool, ckr_k: int) -> list:
    runs = eg.locate_runs("additive")
    lite = [p for s in SEEDS for p in (LITE[s] / "model.pt", LITE[s] / "validation_predictions.csv")]
    if dry:
        return lite + [D7RUN[s] / "validation_predictions.csv" for s in SEEDS] \
            + [runs[g][s] / "validation_predictions.csv" for g in ("xgb", "bisurf") for s in SEEDS] \
            + [XID / f"seed_{s}/validation_predictions.npz" for s in SEEDS] \
            + [DDP / f"seed_{s}/validation_predictions.csv" for s in SEEDS] + [CKR / f"validation/pred_iters{ckr_k}.npz"] \
            + [DSB_VAL / f"{n}_val_predictions.npz" for n in DSB_VAL_STAND_IN.values()]
    return lite + [V2 / "test_eval/test_combination_predictions_bisurf_v2.npz",
                   gate.RUN / "s5/test_combination_predictions_xgb.npz", gate.RUN / "s5/test_combination_predictions_bisurf.npz"] \
        + [XID / f"seed_{s}/test_predictions.npz" for s in SEEDS] + [DDP / f"seed_{s}/test_predictions.npz" for s in SEEDS] \
        + [CKR / f"test/pred_iters{ckr_k}.npz"] + [DSB_RUN[s] / "test_predictions.npz" for s in DSB_SEEDS]


def preconditions(dry: bool) -> dict:
    """Run states and file hashes; reads no response value."""
    for s in SEEDS:
        m = rj(LITE[s] / "metrics.json")
        if m.get("arm") != "LITE" or m.get("seed") != s or m.get("smoke") or m.get("stopped_by") != "max_epochs":
            raise RuntimeError(f"LITE seed {s}: not a finished formal run")
    ev.v2_runs()                                                        # D7 runs finished and checked
    if not dry:
        for s in (11, 12):
            fin = rj(DSB_RUN[s] / "train_finished.json")
            if fin.get("stopped_by") != "max_epochs" or not (DSB_RUN[s] / "TEST_EVALUATED.marker").exists():
                raise RuntimeError(f"DeepSynBa seed {s}: not a finished, once-evaluated run")
    files = input_files(dry, rj(CKR / "validation/selection.json")["selected_iters"])
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise RuntimeError(f"missing inputs: {missing}")
    return {p.resolve().relative_to(ROOT.parent).as_posix(): gate.sha256(p) for p in files}


def pretest_alignment() -> dict:
    """Before the marker: every stored test prediction file carries the D7 point / matrix IDs (ID members only)."""
    d7 = np.load(V2 / "test_eval/test_combination_predictions_bisurf_v2.npz", allow_pickle=False)
    ids, keys = d7["point_ids"], d7["matrix_keys"][::9]
    bad = [] if np.array_equal(np.repeat(keys, 9), d7["matrix_keys"]) else ["D7 matrix_keys not 9 per matrix"]
    for p in [gate.RUN / "s5/test_combination_predictions_xgb.npz", gate.RUN / "s5/test_combination_predictions_bisurf.npz"] \
            + [XID / f"seed_{s}/test_predictions.npz" for s in SEEDS] + [DDP / f"seed_{s}/test_predictions.npz" for s in SEEDS]:
        if not np.array_equal(np.load(p, allow_pickle=False)["point_ids"], ids):
            bad.append(str(p))
    k = rj(CKR / "validation/selection.json")["selected_iters"]
    if not np.array_equal(np.load(CKR / f"test/pred_iters{k}.npz", allow_pickle=False)["matrix_key"], keys):
        bad.append("comboKR test matrix_key")
    for s in DSB_SEEDS:
        have = {"_".join(map(str, r)) for r in np.load(DSB_RUN[s] / "test_predictions.npz", allow_pickle=False)["keys"]}
        if not set(keys) <= have:
            bad.append(f"DeepSynBa seed {s}: {len(set(keys) - have)} S54 matrices missing")
    if bad:
        raise RuntimeError(f"stored test predictions not aligned: {bad}")
    return {"points": int(len(ids)), "matrices": int(len(keys)), "files_aligned": "D7, S5 xgb/bisurf, XGB+id x5, DD-PRiSM x5, comboKR, DeepSynBa 10/11/12"}


def load_validation(frame):
    n = len(frame) // 9
    check_order(frame, n)
    keys = frame.matrix_key.to_numpy()[::9]
    dsb16, truth = {}, None
    for seed, name in DSB_VAL_STAND_IN.items():
        z = np.load(DSB_VAL / f"{name}_val_predictions.npz", allow_pickle=False)
        pos = pd.Index(z["matrix_key"]).get_indexer(keys)
        if (pos < 0).any():
            raise RuntimeError(f"{name}: {int((pos < 0).sum())} validation matrices missing")
        dsb16[seed] = z["predictions"].astype(float)[pos]
        t = z["truth"].astype(float)[pos]
        truth = t if truth is None else truth
        if not np.array_equal(t, truth):
            raise RuntimeError("DeepSynBa validation files disagree on the truth")
    grid = lambda c: frame[c].to_numpy(float).reshape(n, 3, 3)  # noqa: E731
    if np.max(np.abs(truth[:, 1:, 1:] - grid("response_percent"))) > 1e-3:
        raise RuntimeError("DeepSynBa validation truth differs from the gate table")
    a, b = grid("mono_a_percent"), grid("mono_b_percent")
    if np.ptp(a, axis=2).max() > 0 or np.ptp(b, axis=1).max() > 0:
        raise RuntimeError("Lookup anchors are not constant along the other drug's dose")
    lk = np.zeros((n, 4, 4))
    lk[:, 0, 0] = rj(gate.S1 / "s1_report.json")["lookup"]["control_mean_percent"]
    lk[:, 1:, 0], lk[:, 0, 1:] = a[:, :, 0], b[:, 0, :]
    matrices = [(k, truth[i], lk[i]) for i, k in enumerate(keys)]
    runs = eg.locate_runs("additive")
    comb = lookup_comb(frame)
    comb["combokr2"] = {0: combokr("validation", keys)}
    comb["ddprism_measured_mono"] = {s: csv_pred(DDP / f"seed_{s}/validation_predictions.csv", frame) for s in SEEDS}
    comb["xgb"] = {s: csv_pred(runs["xgb"][s] / "validation_predictions.csv", frame) for s in SEEDS}
    comb["xgb_identity"] = {}
    for s in SEEDS:
        p = np.load(XID / f"seed_{s}/validation_predictions.npz", allow_pickle=False)["prediction"]
        if len(p) != len(frame):
            raise RuntimeError("XGB+identity validation predictions have the wrong length")
        comb["xgb_identity"][s] = p.astype(float) * 100.0
    comb["bisurf"] = {s: csv_pred(runs["bisurf"][s] / "validation_predictions.csv", frame) for s in SEEDS}
    comb[OURS] = {s: csv_pred(D7RUN[s] / "validation_predictions.csv", frame) for s in SEEDS}
    return frame, matrices, comb, dsb16, {"deepsynba_stand_ins": DSB_VAL_STAND_IN}


def s55_combination_rmse(seed):
    """run_seed.evaluate's combination score on the whole released test file (55,120 matrices)."""
    z = np.load(DSB_RUN[seed] / "test_predictions.npz", allow_pickle=False)
    single = np.zeros((4, 4), bool)
    single[0, :] = single[:, 0] = True
    p, t = z["predictions"][:, ~single].astype(np.float64).ravel(), z["truth"][:, ~single].astype(np.float64).ravel()
    return float(np.sqrt(np.mean((p - t) ** 2)))


def load_test():
    frame, matrices = common.load_test()                 # the one read of the split-3 responses
    n = len(matrices)
    check_order(frame, n)
    ids, keys = frame.source_combodrugseq.to_numpy(), frame.matrix_key.to_numpy()[::9]
    if not np.array_equal(keys, np.array([k for k, _, _ in matrices])):
        raise RuntimeError("frame and matrices disagree")
    s5 = gate.RUN / "s5"
    comb = lookup_comb(frame)
    comb["combokr2"] = {0: combokr("test", keys)}
    comb["ddprism_measured_mono"] = {s: npz_pred(DDP / f"seed_{s}/test_predictions.npz", "prediction", ids) for s in SEEDS}
    comb["xgb"] = {s: npz_pred(s5 / "test_combination_predictions_xgb.npz", f"seed_{s}", ids) for s in SEEDS}
    comb["xgb_identity"] = {s: npz_pred(XID / f"seed_{s}/test_predictions.npz", "prediction", ids) for s in SEEDS}
    comb["bisurf"] = {s: npz_pred(s5 / "test_combination_predictions_bisurf.npz", f"seed_{s}", ids) for s in SEEDS}
    saved = V2 / "test_eval/test_combination_predictions_bisurf_v2.npz"
    comb[OURS] = {s: npz_pred(saved, f"seed_{s}", ids) for s in SEEDS}
    dsb16, layout, default = {}, {}, eg.DSB
    try:
        for seed in DSB_SEEDS:
            eg.DSB = DSB_RUN[seed]
            dsb16[seed], layout[seed] = eg.deepsynba_combination(matrices)
    finally:
        eg.DSB = default
    info = {"deepsynba_layout": layout, "deepsynba_s55_combination_rmse": {s: s55_combination_rmse(s) for s in DSB_SEEDS},
            "saved_ensemble": np.load(saved, allow_pickle=False)["ensemble"]}
    return frame, matrices, comb, dsb16, info


def finish_comb(comb, dsb16):
    comb["deepsynba_v2"] = {s: dsb16[s][:, 1:, 1:].reshape(-1) for s in DSB_SEEDS}
    comb["deepsynba_v2_s10"] = {10: comb["deepsynba_v2"][10]}
    comb["bisurf_v2_ensemble"] = {0: np.mean([comb[OURS][s] for s in SEEDS], axis=0)}
    return {m: comb[m] for m in METHODS}


# ---------------------------------------------------------------------------------------------------------------
# LITE (D18 models)
# ---------------------------------------------------------------------------------------------------------------
def lite_setup(cfg, frames, banks):
    models, lbanks = {}, None
    for s in SEEDS:
        models[s], lbanks = ra.load_lite(LITE[s] / "model.pt", s, banks, cfg["device"])
    features = rp.load_deepsynba_features(banks.drug_ids, banks.cell_ids, frames,
                                          feature_dir=rp.guard(rp.FEATURE_DIR), mapping_dir=rp.guard(gate.S1))
    return models, lbanks, rp.DeviceExternal(features, "both", torch.device(cfg["device"]))


def lite_predict(models, lbanks, ext, cfg, frame, role):
    table = learned.build_compact_point_table(frame, lbanks, role=role, fold=0, monotherapy_upper_bound=3.0)
    return {s: 100.0 * np.asarray(rp.predict_ext(m, table, lbanks, ext, cfg), float) for s, m in models.items()}


# ---------------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------------
def preds16_of(matrices, comb, dsb16):
    lookup16 = np.stack([p for _, _, p in matrices])
    out = {}
    for m, seeds in comb.items():
        out[m] = {}
        for s, p in seeds.items():
            if m.startswith("deepsynba"):
                out[m][s] = dsb16[s]
            else:
                full = lookup16.copy()
                full[:, 1:, 1:] = np.asarray(p, float).reshape(-1, 3, 3)
                out[m][s] = full
    return out


def subset_stats(frame, comb, keep):
    """Per-seed RMSE, pair SSE and the pair bootstrap of one subset, as evaluate_v2.score computes its sensitivity."""
    sub, y = frame[keep].reset_index(drop=True), frame.response_percent.to_numpy(float)[keep]
    sse, per_seed, n, pairs = {}, {}, None, None
    for m, seeds in comb.items():
        for s, p in seeds.items():
            p = np.asarray(p, float)[keep]
            sse[(m, s)], n, pairs = eg.pair_sse(sub, p)
            per_seed[(m, s)] = ev.rmse(p, y)
    point = {m: float(np.mean([per_seed[(m, s)] for s in seeds])) for m, seeds in comb.items()}
    return dict(sse=sse, n=n, pairs=pairs, per_seed=per_seed, point=point, boots=eg.bootstrap(sse, n, list(comb)),
                points=int(keep.sum()))


def compare(st, a, b):
    return {"relative_change": (st["point"][a] - st["point"][b]) / st["point"][b], "ci95": ev.interval(st["boots"][a], st["boots"][b])}


def verdict(st, comb, a, b):
    c = compare(st, a, b)
    below = int(sum(st["per_seed"][(a, s)] < st["point"][b] for s in comb[a]))
    return {**c, "ours_seeds_below_comparator_mean": below, "passed": bool(c["ci95"][1] < 0 and below >= 4)}


def seeds_of(st, comb, m):
    return {s: st["per_seed"][(m, s)] for s in comb[m]}


def paired_t(a: dict, b: dict) -> dict:
    """Seed-paired relative change with a t interval (the card I statistic)."""
    rel = np.array([100.0 * (a[s] - b[s]) / b[s] for s in SEEDS])
    h = stats.t.ppf(0.975, len(rel) - 1) * rel.std(ddof=1) / np.sqrt(len(rel))
    return {"per_seed_percent": rel.tolist(), "mean_percent": float(rel.mean()),
            "ci95_percent": [float(rel.mean() - h), float(rel.mean() + h)], "worse_seeds": int((rel > 0).sum())}


def drug_weights(pairs):
    """Vertex bootstrap over drugs: (draws, pairs) weights m_a * m_b."""
    ab = np.array([p.split("|") for p in pairs])
    drugs = np.unique(ab)
    ia, ib = np.searchsorted(drugs, ab[:, 0]), np.searchsorted(drugs, ab[:, 1])
    rng = np.random.default_rng(eg.BOOT_SEED)
    m = np.stack([np.bincount(rng.integers(0, len(drugs), len(drugs)), minlength=len(drugs)) for _ in range(eg.DRAWS)])
    return m[:, ia].astype(float) * m[:, ib], list(drugs), ia, ib


def boot_rmse(sse, n, methods, w):
    """evaluate_gate.bootstrap with the weights passed in (rows = draws)."""
    denom = w @ n
    return {m: np.stack([np.sqrt((w @ sse[(m, s)]) / denom) for s in sorted(s for k, s in sse if k == m)]).mean(axis=0)
            for m in methods}


def jackknife(theta, vals, drugs, prala_id):
    d = len(vals)
    se = float(np.sqrt((d - 1) / d * np.sum((vals - vals.mean()) ** 2)))
    i, j = int(np.argmin(vals)), int(np.argmax(vals))
    return {"estimate": float(theta), "jackknife_se": se, "normal_ci95": [float(theta - 1.96 * se), float(theta + 1.96 * se)],
            "min": {"drug": drugs[i], "value": float(vals[i])}, "max": {"drug": drugs[j], "value": float(vals[j])},
            "without_pralatrexate": float(vals[drugs.index(prala_id)]), "drugs": d}


def t2_rmse(st, comb, prala_id, chk):
    sse, n = st["sse"], st["n"]
    rng = np.random.default_rng(eg.BOOT_SEED)                    # self-check: with pair weights == evaluate_gate.bootstrap
    w_pair = np.stack([np.bincount(rng.integers(0, len(n), len(n)), minlength=len(n)) for _ in range(eg.DRAWS)]).astype(float)
    mine = boot_rmse(sse, n, [OURS, "deepsynba_v2"], w_pair)
    chk.same("T2 boot_rmse with pair weights == evaluate_gate.bootstrap (bitwise)",
             all(np.array_equal(mine[m], st["boots"][m]) for m in mine))
    w, drugs, ia, ib = drug_weights(st["pairs"])
    methods = sorted({m for pair in T2_RMSE for m in pair})
    boots = boot_rmse(sse, n, methods, w)
    kept = (w > 0).sum(axis=1)
    out, lodo_rows = {}, []
    for a, b in T2_RMSE:
        vals = []
        for j in range(len(drugs)):
            keep = (ia != j) & (ib != j)
            r = {m: np.mean([np.sqrt(sse[(m, s)][keep].sum() / n[keep].sum()) for s in comb[m]]) for m in (a, b)}
            vals.append((r[a] - r[b]) / r[b])
        vals = np.array(vals)
        c = compare(st, a, b)
        out[f"{a}_vs_{b}"] = {"estimate": c["relative_change"], "pair_bootstrap_ci95": c["ci95"],
                              "drug_bootstrap_ci95": ev.interval(boots[a], boots[b]),
                              "leave_one_drug_out": jackknife(c["relative_change"], vals, drugs, prala_id)}
        lodo_rows += [dict(comparison=f"{a}_vs_{b}", quantity="relative_rmse_change", left_out_drug=d, value=v)
                      for d, v in zip(drugs, vals)]
    info = {"drugs": len(drugs), "pairs": int(len(n)), "pairs_kept_per_draw": {"min": int(kept.min()),
            "median": float(np.median(kept)), "max": int(kept.max())}, "draws": eg.DRAWS, "seed": eg.BOOT_SEED,
            "scheme": "vertex bootstrap over drugs, pair weight m_a*m_b; cells stay aggregated within pairs"}
    return out, info, (w, drugs, ia, ib), lodo_rows


def t2_bliss(syn, st, weights, prala_id):
    w, drugs, ia, ib = weights
    observed, derived, meta, comps = syn["observed"], syn["derived"], syn["meta"], syn["comparisons"]
    codes = pd.Index(st["pairs"]).get_indexer(meta.pair_id)
    if (codes < 0).any():
        raise RuntimeError("synergy matrices with an unknown pair")
    names = SYN_GROUPS["BS"] + SYN_GROUPS["XI"] + SYN_GROUPS["DSB3"]
    obs, v = observed["Bliss"], {k: derived[k]["Bliss"] for k in names}
    draws = {k: np.empty(eg.DRAWS) for k in names}
    for lo in range(0, eg.DRAWS, 50):
        W = w[lo:lo + 50][:, codes]
        for k in names:
            draws[k][lo:lo + 50] = sc.w_pearson_fixed(v[k], obs, W)
    ma, mb = ia[codes], ib[codes]
    lodo = {k: np.empty(len(drugs)) for k in names}
    for j in range(len(drugs)):
        keep = (ma != j) & (mb != j)
        for k in names:
            lodo[k][j] = np.corrcoef(v[k][keep], obs[keep])[0, 1]
    group = lambda d, g: np.mean([d[k] for k in SYN_GROUPS.get(g, [g])], axis=0)  # noqa: E731
    point = {k: float(np.corrcoef(v[k], obs)[0, 1]) for k in names}
    out, lodo_rows = {}, []
    for (a, b), cname in ((("BS", "DSB"), "C1 BS - DSB(observed ref)"), (("BS", "DSB3"), "C9 BS - DSB3(observed ref)"),
                          (("BS", "XI"), "C5 BS - XGB_identity")):
        row = comps[(comps.subset == "all") & (comps.comparison == cname) & (comps.score == "Bliss") & (comps.metric == "pearson")].iloc[0]
        est = float(group(point, a) - group(point, b))
        if abs(est - row.difference) > 1e-12:
            raise RuntimeError(f"{cname}: Bliss Pearson difference disagrees with synergy_check.analyse")
        vals = group(lodo, a) - group(lodo, b)
        diff = group(draws, a) - group(draws, b)
        out[cname] = {"estimate": est, "pair_bootstrap_ci95": [float(row.ci_low), float(row.ci_high)],
                      "drug_bootstrap_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                      "leave_one_drug_out": jackknife(est, vals, drugs, prala_id)}
        lodo_rows += [dict(comparison=cname, quantity="bliss_pearson_difference", left_out_drug=d, value=x)
                      for d, x in zip(drugs, vals)]
    return out, lodo_rows


# ---------------------------------------------------------------------------------------------------------------
# synergy (synergy_check.main's body on the inputs built here; analyse() and the score functions unchanged)
# ---------------------------------------------------------------------------------------------------------------
def _aggregate(res, name, x, m):
    return np.mean([res[(s, x, m)] for s in SYN_GROUPS[name]], axis=0) if name in SYN_GROUPS else res[(name, x, m)]


def _point_value(points, name, x, m):
    return float(np.mean([points[(s, x)][m] for s in SYN_GROUPS[name]])) if name in SYN_GROUPS else points[(name, x)][m]


def synergy(frame, matrices, comb, dsb16):
    n = len(matrices)
    truth, lk = np.stack([m for _, m, _ in matrices]), np.stack([p for _, _, p in matrices])
    y = frame.response_percent.to_numpy(float).reshape(n, 3, 3)
    obs_a, obs_b, lk_a, lk_b = truth[:, 1:, 0], truth[:, 0, 1:], lk[:, 1:, 0], lk[:, 0, 1:]
    preds = {}
    for tag, method in (("BS", OURS), ("XI", "xgb_identity"), ("GX", "xgb")):
        for s in SEEDS:
            preds[f"{tag}_s{s}"] = np.asarray(comb[method][s], float).reshape(n, 3, 3)
    preds["DSB"], preds["DSB_s11"], preds["DSB_s12"] = (dsb16[s][:, 1:, 1:] for s in DSB_SEEDS)
    preds["LK_HSA"] = np.minimum(lk_a[:, :, None], lk_b[:, None, :])
    preds["LK_Bliss"] = lk_a[:, :, None] * lk_b[:, None, :] / 100.0
    for tag in ("BS", "XI", "GX"):
        preds[f"{tag}_ens5"] = np.mean([preds[k] for k in SYN_GROUPS[tag]], axis=0)
    ref_obs = sc.references(obs_a, obs_b)
    observed = sc.matrix_scores(ref_obs, y)
    derived = {s: sc.matrix_scores(ref_obs, p) for s, p in preds.items()}
    ref_own = sc.references(dsb16[10][:, 1:, 0], dsb16[10][:, 0, 1:])
    derived["DSB_own"] = sc.matrix_scores(ref_own, preds["DSB"])
    cell_rmse = {s: {x: (lambda mask, p=p: float(np.sqrt(np.mean((p[mask] - y[mask]) ** 2)))) for x in sc.SCORES}
                 for s, p in preds.items()}
    cell_rmse["DSB_own"] = {x: (lambda mask, x=x: float(np.sqrt(np.mean(
        (((ref_own[x] - preds["DSB"]) - (ref_obs[x] - y))[mask]) ** 2)))) for x in sc.SCORES}
    meta = frame.iloc[::9][["matrix_key", "pair_id", "deepsynba_cell", "deepsynba_drug_a", "deepsynba_drug_b"]].reset_index(drop=True)
    has = ((meta.deepsynba_drug_a == sc.EXCLUDED_DRUG) | (meta.deepsynba_drug_b == sc.EXCLUDED_DRUG)).to_numpy()
    sc.COMPARISONS, sc.aggregate, sc.point_value = SYN_COMPARISONS, _aggregate, _point_value
    parts = {label: sc.analyse(mask, label, meta, observed, derived, cell_rmse)
             for label, mask in (("all", np.ones(n, bool)), (f"excluding_{sc.EXCLUDED_DRUG}", ~has))}
    return {"points": pd.concat([p[0] for p in parts.values()], ignore_index=True),
            "comparisons": pd.concat([p[1] for p in parts.values()], ignore_index=True),
            "seeds": pd.concat([p[2] for p in parts.values()], ignore_index=True),
            "info": {label: p[3] for label, p in parts.items()}, "observed": observed, "derived": derived, "meta": meta}


def syn_value(points, series, subset, score, metric):
    names = SYN_GROUPS.get(series, [series])
    sel = points[(points.subset == subset) & (points.score == score) & points.series.isin(names)]
    return float(sel[metric].astype(float).mean())


def syn_comp(comps, name, subset, metric="pearson", score="Bliss"):
    return comps[(comps.subset == subset) & (comps.comparison == name) & (comps.score == score) & (comps.metric == metric)].iloc[0]


# ---------------------------------------------------------------------------------------------------------------
# reproduction checks
# ---------------------------------------------------------------------------------------------------------------
def check_saved(chk, label, ref_point, ref_cmp, st):
    rename = {"deepsynba_v2": "deepsynba_v2_s10"}
    for k, v in ref_point.items():
        chk.near(f"{label} point {k}", st["point"][rename.get(k, k)], v)
    for key, c in ref_cmp.items():
        a, b = (rename.get(x, x) for x in key.split("_vs_"))
        mine = compare(st, a, b)
        chk.deep(f"{label} {key}", mine, c)


def reproduce_test(chk, frame, matrices, comb, dsb16, prala, st, syn, info):
    d7 = rj(V2 / "test_eval/evaluation.json")
    comb_d7 = {k: comb[k] for k in ("lookup_hsa", "lookup_bliss", "xgb", "bisurf", OURS, "bisurf_v2_ensemble")}
    comb_d7["deepsynba_v2"], comb_d7["_dsb16"] = comb["deepsynba_v2_s10"], dsb16[10]
    rows7, point7, cmp7, dec7, sens7, pairs7 = ev.score(frame, matrices, comb_d7, prala)
    mine = json.loads(json.dumps({"point": point7, "comparisons": cmp7, "decision": dec7, "sensitivity": sens7}, default=float))
    chk.deep("D7 point", mine["point"], d7["point_estimates_combination_rmse_percent"])
    chk.deep("D7 comparisons", mine["comparisons"], d7["comparisons"])
    chk.deep("D7 decision", mine["decision"], d7["decision"])
    chk.deep("D7 sensitivity", mine["sensitivity"], d7["sensitivity_without_pralatrexate"])
    chk.table("D7 per_seed_metrics.csv", V2 / "test_eval/per_seed_metrics.csv", rows7, ["method", "seed"])
    chk.same("D7 saved ensemble == mean of the saved seeds", np.array_equal(info["saved_ensemble"], comb["bisurf_v2_ensemble"][0]))
    s5 = rj(gate.RUN / "s5/evaluation.json")
    check_saved(chk, "S5", s5["point_estimates_combination_rmse_percent"], s5["comparisons"], st["all"])
    d8 = rj(XID / "test_evaluation.json")
    check_saved(chk, "D8", d8["point_estimates_combination_rmse_percent"], d8["comparisons"], st["all"])
    for name in ("combokr_test", "ddprism_test"):
        d10 = rj(V2 / f"{name}/test_evaluation.json")
        check_saved(chk, f"D10 {name}", d10["point"], d10["comparisons"], st["all"])
    for s in DSB_SEEDS:
        chk.near(f"DeepSynBa seed {s} S55 combination RMSE vs its evaluation.json",
                 info["deepsynba_s55_combination_rmse"][s], rj(DSB_RUN[s] / "evaluation.json")["combination"]["rmse"])
    reproduce_d9(chk, syn, "test")


def reproduce_d9(chk, syn, role):
    rn = D9_RENAME[role]
    chk.table(f"D9 {role} point_estimates.csv", D9 / f"{role}/point_estimates.csv", syn["points"],
              ["subset", "series", "score"], {"series": rn["series"]})
    chk.table(f"D9 {role} paired_comparisons.csv", D9 / f"{role}/paired_comparisons.csv", syn["comparisons"],
              ["subset", "comparison", "score", "metric"], {"comparison": rn["comparison"]})


def reproduce_validation(chk, st, comb, syn, t5):
    ps = st["all"]["per_seed"]
    runs = eg.locate_runs("additive")
    for s in SEEDS:
        chk.near(f"val Ours s{s} vs metrics.json", ps[(OURS, s)], rj(D7RUN[s] / "metrics.json")["validation_pooled_rmse_percent"], 1e-6)
        chk.near(f"val LITE s{s} vs metrics.json", ps[("lite", s)], rj(LITE[s] / "metrics.json")["validation_pooled_rmse_percent"], 1e-6)
        for g in ("xgb", "bisurf"):
            chk.near(f"val {g} s{s} vs metrics.json", ps[(g, s)], rj(runs[g][s] / "metrics.json")["validation_pooled_rmse_percent"], 1e-6)
        chk.near(f"val DD-PRiSM s{s} vs metrics.json", ps[("ddprism_measured_mono", s)],
                 rj(DDP / f"seed_{s}/metrics.json")["validation"]["combination_rmse_percent"], 1e-5)
    for row in rj(XID / "selection_summary.json")["seeds"]:   # predictions stored as float32
        chk.near(f"val XGB+identity s{row['seed']} vs selection_summary.json", ps[("xgb_identity", row["seed"])],
                 row["validation_pooled_rmse_percent"], 1e-5)
    sel = rj(CKR / "validation/selection.json")
    chk.near("val comboKR vs selection.json", ps[("combokr2", 0)],
             sel["per_iteration"][str(sel["selected_iters"])]["combination_rmse_percent"], 1e-9)
    chk.near("val lookup HSA vs comboKR selection.json", ps[("lookup_hsa", 0)], sel["lookup_hsa_validation"]["combination_rmse_percent"], 1e-9)
    p = st["all"]["point"]
    ours = [ps[(OURS, s)] for s in SEEDS]
    for s, ref in zip(SEEDS, (8.667, 8.844, 8.755, 8.853, 8.848)):
        chk.doc(f"ABLATION_RESULTS §0 Ours val s{s}", ps[(OURS, s)], ref, 3)
    chk.doc("D7 val mean (DEVIATIONS D7)", p[OURS], 8.793, 3)
    chk.doc("D7 val SD", float(np.std(ours, ddof=1)), 0.081, 3)
    for s, ref in zip(SEEDS, (9.498, 9.686, 9.518, 9.533, 9.687)):
        chk.doc(f"D18 LITE val s{s}", ps[("lite", s)], ref, 3)
    chk.doc("D18 LITE val mean", p["lite"], 9.584, 3)
    chk.doc("D18 LITE vs D7 paired mean %", t5["all"]["seed_paired_vs_ours"]["mean_percent"], 9.00, 2)
    chk.doc("D18 LITE vs D7 paired CI low %", t5["all"]["seed_paired_vs_ours"]["ci95_percent"][0], 7.98, 2)
    chk.doc("D18 LITE vs D7 paired CI high %", t5["all"]["seed_paired_vs_ours"]["ci95_percent"][1], 10.01, 2)
    xi = [ps[("xgb_identity", s)] for s in SEEDS]
    chk.doc("TEST_RESULTS §7 XGB+identity val mean", float(np.mean(xi)), 10.075, 3)
    # the document's "± 0.012" is a rounding slip: the stored values give 0.012503 (-> 0.013); checked against the source
    src = [r["validation_pooled_rmse_percent"] for r in rj(XID / "selection_summary.json")["seeds"]]
    chk.near("XGB+identity val SD vs selection_summary.json", float(np.std(xi, ddof=1)), float(np.std(src, ddof=1)), 1e-8)
    for s, ref in zip(SEEDS, (11.62, 11.50, 11.45, 11.48, 11.66)):
        chk.doc(f"COMPARATORS_TEST §2 DD-PRiSM val s{s}", ps[("ddprism_measured_mono", s)], ref, 2)
    chk.doc("COMPARATORS_TEST §1 comboKR val (10 iters)", p["combokr2"], 13.316, 3)
    chk.doc("COMPARATORS_TEST §2 lookup HSA val", p["lookup_hsa"], 11.84, 2)
    chk.doc("V2_GAP DeepSynBa epoch 484 val (stand-in seed 10)", ps[("deepsynba_v2", 10)], 9.966, 3)
    chk.doc("V2_GAP DeepSynBa epoch 47 val (stand-in seed 11)", ps[("deepsynba_v2", 11)], 10.600, 3)
    pts, cps = syn["points"], syn["comparisons"]
    for label, value, ref in (
            ("BS Bliss Pearson", syn_value(pts, "BS", "all", "Bliss", "pearson"), 0.752),
            ("DSB Bliss Pearson", syn_value(pts, "DSB", "all", "Bliss", "pearson"), 0.628),
            ("gate XGB Bliss Pearson", syn_value(pts, "GX", "all", "Bliss", "pearson"), 0.460),
            ("C1 diff", syn_comp(cps, "C1 BS - DSB(observed ref)", "all").difference, 0.124),
            ("C1 CI low", syn_comp(cps, "C1 BS - DSB(observed ref)", "all").ci_low, 0.091),
            ("C1 CI high", syn_comp(cps, "C1 BS - DSB(observed ref)", "all").ci_high, 0.158),
            ("BS - gate XGB diff", syn_comp(cps, "C7 BS - XGB_gate", "all").difference, 0.292),
            ("BS - gate XGB CI low", syn_comp(cps, "C7 BS - XGB_gate", "all").ci_low, 0.234),
            ("BS - gate XGB CI high", syn_comp(cps, "C7 BS - XGB_gate", "all").ci_high, 0.356),
            ("C2 ens diff", syn_comp(cps, "C2 BS_ens5 - DSB(observed ref)", "all").difference, 0.157),
            ("excl BS Bliss Pearson", syn_value(pts, "BS", "excluding_pralatrexate", "Bliss", "pearson"), 0.727),
            ("excl DSB Bliss Pearson", syn_value(pts, "DSB", "excluding_pralatrexate", "Bliss", "pearson"), 0.596),
            ("excl C1 diff", syn_comp(cps, "C1 BS - DSB(observed ref)", "excluding_pralatrexate").difference, 0.131)):
        chk.doc(f"SYNERGY_RECHECK §1 {label}", value, ref, 3)
    reproduce_d9(chk, syn, "validation")


def check_test_docs(chk):
    """Document constants vs the stored test results (no split-3 read; recomputed == stored is checked separately)."""
    d7 = rj(V2 / "test_eval/evaluation.json")
    P, C, S = d7["point_estimates_combination_rmse_percent"], d7["comparisons"], d7["sensitivity_without_pralatrexate"]
    d8 = rj(XID / "test_evaluation.json")
    ck, dd = rj(V2 / "combokr_test/test_evaluation.json"), rj(V2 / "ddprism_test/test_evaluation.json")
    pts = pd.read_csv(D9 / "test/point_estimates.csv", float_precision="round_trip")
    cps = pd.read_csv(D9 / "test/paired_comparisons.csv", float_precision="round_trip")
    bliss = lambda names: float(pts[(pts.subset == "all") & (pts.score == "Bliss") & pts.series.isin(names)].pearson.mean())  # noqa: E731
    c1, c5 = syn_comp(cps, "C1 BS - DSB(observed ref)", "all"), syn_comp(cps, "C5 BS - XGB_identity", "all")
    for label, value, ref, dp in (
            ("TEST_RESULTS §1 Ours", P[OURS], 8.241, 3), ("TEST_RESULTS §1 DeepSynBa", P["deepsynba_v2"], 9.725, 3),
            ("TEST_RESULTS §1 ensemble", P["bisurf_v2_ensemble"], 7.766, 3), ("TEST_RESULTS §1 HSA", P["lookup_hsa"], 12.009, 3),
            ("TEST_RESULTS §1 Bliss", P["lookup_bliss"], 11.976, 3), ("TEST_RESULTS §1 gate XGB", P["xgb"], 10.408, 3),
            ("TEST_RESULTS §1 orig BiSurfGNN", P["bisurf"], 10.424, 3),
            ("TEST_RESULTS §2 rule %", 100 * C["bisurf_v2_vs_deepsynba_v2"]["relative_change"], -15.26, 2),
            ("TEST_RESULTS §2 CI low %", 100 * C["bisurf_v2_vs_deepsynba_v2"]["ci95"][0], -19.64, 2),
            ("TEST_RESULTS §2 CI high %", 100 * C["bisurf_v2_vs_deepsynba_v2"]["ci95"][1], -11.24, 2),
            ("TEST_RESULTS §4 excl Ours", S["point_estimates_combination_rmse_percent"][OURS], 8.198, 3),
            ("TEST_RESULTS §4 excl DeepSynBa", S["point_estimates_combination_rmse_percent"]["deepsynba_v2"], 9.171, 3),
            ("TEST_RESULTS §4 excl %", 100 * S["bisurf_v2_vs_deepsynba_v2"]["relative_change"], -10.61, 2),
            ("TEST_RESULTS §4 excl CI low %", 100 * S["bisurf_v2_vs_deepsynba_v2"]["ci95"][0], -11.92, 2),
            ("TEST_RESULTS §4 excl CI high %", 100 * S["bisurf_v2_vs_deepsynba_v2"]["ci95"][1], -9.30, 2),
            ("TEST_RESULTS §7 XGB+identity", d8["point_estimates_combination_rmse_percent"]["xgb_identity"], 10.060, 3),
            ("TEST_RESULTS §7 Ours vs XGB+id %", 100 * d8["comparisons"]["bisurf_v2_vs_xgb_identity"]["relative_change"], -18.08, 2),
            ("TEST_RESULTS §7 CI low %", 100 * d8["comparisons"]["bisurf_v2_vs_xgb_identity"]["ci95"][0], -22.57, 2),
            ("TEST_RESULTS §7 CI high %", 100 * d8["comparisons"]["bisurf_v2_vs_xgb_identity"]["ci95"][1], -13.60, 2),
            ("COMPARATORS_TEST comboKR", ck["point"]["combokr2"], 13.354, 3),
            ("COMPARATORS_TEST DD-PRiSM", dd["point"]["ddprism_measured_mono"], 11.661, 3),
            ("SYNERGY_RECHECK §2 Ours Bliss Pearson", bliss(SYN_GROUPS["BS"]), 0.808, 3),
            ("SYNERGY_RECHECK §2 DSB Bliss Pearson", bliss(["DSB"]), 0.658, 3),
            ("SYNERGY_RECHECK §2 XGB+id Bliss Pearson", bliss([f"XGB_s{s}" for s in SEEDS]), 0.599, 3),
            ("SYNERGY_RECHECK §2 C1 diff", c1.difference, 0.150, 3), ("SYNERGY_RECHECK §2 C1 CI low", c1.ci_low, 0.096, 3),
            ("SYNERGY_RECHECK §2 C1 CI high", c1.ci_high, 0.215, 3), ("SYNERGY_RECHECK §2 vs XGB+id diff", c5.difference, 0.209, 3),
            ("SYNERGY_RECHECK §2 vs XGB+id CI low", c5.ci_low, 0.161, 3), ("SYNERGY_RECHECK §2 vs XGB+id CI high", c5.ci_high, 0.256, 3)):
        chk.doc(f"stored {label}", value, ref, dp)


# ---------------------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------------------
def pc(c):
    return f"{100 * c['relative_change']:+.2f}%，[{100 * c['ci95'][0]:+.2f}, {100 * c['ci95'][1]:+.2f}]"


def tables_md(R, role):
    L = [f"# T 批次（D19）评分表：{role}", ""]
    if role != "test":
        L += ["> **验证集预演（split 2），不是结果。** DeepSynBa 种子 11、12 在验证集上没有预测，用种子 10 的其他检查点替身"
              "（11 = pilot 第 47 轮，12 = last，与第 484 轮逐字节相同）。", ""]
    t1 = R["T1"]
    L += ["## T1 主比较（药对 bootstrap 2000 次，种子 20261009；规则与 D7 相同）", "",
          "| 子集 | 对照 | 对照 RMSE | Ours RMSE | 相对变化，95% 区间 | Ours 低于对照均值的种子 | 判定 |", "|---|---|---:|---:|---|---:|---|"]
    for sub in SUBSETS:
        for key, name in (("main_3_seed", "DeepSynBa 种子 10–12 均值（新主比较）"), ("previous_1_seed", "DeepSynBa 种子 10（原 D7）")):
            v = t1[sub][key]
            L.append(f"| {sub} | {name} | {v['comparator_rmse']:.3f} | {v['ours_rmse']:.3f} | {pc(v)} | "
                     f"{v['ours_seeds_below_comparator_mean']}/5 | {'通过' if v['passed'] else '未通过'} |")
    L += ["", "DeepSynBa 各种子（全部）：" + "、".join(f"{s}: {x:.3f}" for s, x in t1["all"]["deepsynba_per_seed"].items())
          + "；Ours 各种子：" + "、".join(f"{s}: {x:.3f}" for s, x in t1["all"]["ours_per_seed"].items()), ""]
    L += ["## T3 全部方法（联用格 RMSE，%）", "",
          "| 方法 | 种子数 | 全部 | 种子 SD | 去掉 pralatrexate | Ours 相对它（全部） | Ours 相对它（去掉 pralatrexate） |",
          "|---|---:|---:|---:|---:|---|---|"]
    for m in METHODS:
        a, e = R["T3"]["all"][m], R["T3"]["excluding_pralatrexate"][m]
        sd = "—" if a["seed_sd"] is None else f"{a['seed_sd']:.3f}"
        L.append(f"| {LABEL[m]} | {len(a['per_seed'])} | {a['rmse']:.3f} | {sd} | {e['rmse']:.3f} | "
                 f"{pc(a['ours_vs']) if m != OURS else '—'} | {pc(e['ours_vs']) if m != OURS else '—'} |")
    ex = R["T3"]["subset_size"]
    L += ["", f"去掉 pralatrexate：{ex['excluding_pralatrexate']['pairs']} 个药对、{ex['excluding_pralatrexate']['points']} 个联用格"
          f"（全部 {ex['all']['pairs']} 个药对、{ex['all']['points']} 个联用格）。", ""]
    t2 = R["T2"]
    L += [f"## T2 敏感性（不改判定）：药物 bootstrap（{t2['info']['drugs']} 个药物，每次保留药对中位数 "
          f"{t2['info']['pairs_kept_per_draw']['median']:.0f}/{t2['info']['pairs']}）与留一药 jackknife", "",
          "| 比较 | 估计 | 药对 bootstrap 区间 | 药物 bootstrap 区间 | jackknife SE | jackknife 正态区间 | 留一药范围（药物） | 去掉 pralatrexate |",
          "|---|---:|---|---|---:|---|---|---:|"]
    for name, v in t2["rmse"].items():
        j = v["leave_one_drug_out"]
        f = lambda x: f"{100 * x:+.2f}%"  # noqa: E731
        L.append(f"| RMSE {name} | {f(v['estimate'])} | [{f(v['pair_bootstrap_ci95'][0])}, {f(v['pair_bootstrap_ci95'][1])}] | "
                 f"[{f(v['drug_bootstrap_ci95'][0])}, {f(v['drug_bootstrap_ci95'][1])}] | {100 * j['jackknife_se']:.2f} pp | "
                 f"[{f(j['normal_ci95'][0])}, {f(j['normal_ci95'][1])}] | {f(j['min']['value'])}（{j['min']['drug']}）到 "
                 f"{f(j['max']['value'])}（{j['max']['drug']}） | {f(j['without_pralatrexate'])} |")
    for name, v in t2["bliss_pearson"].items():
        j = v["leave_one_drug_out"]
        L.append(f"| Bliss Pearson {name} | {v['estimate']:+.3f} | [{v['pair_bootstrap_ci95'][0]:+.3f}, {v['pair_bootstrap_ci95'][1]:+.3f}] | "
                 f"[{v['drug_bootstrap_ci95'][0]:+.3f}, {v['drug_bootstrap_ci95'][1]:+.3f}] | {j['jackknife_se']:.3f} | "
                 f"[{j['normal_ci95'][0]:+.3f}, {j['normal_ci95'][1]:+.3f}] | {j['min']['value']:+.3f}（{j['min']['drug']}）到 "
                 f"{j['max']['value']:+.3f}（{j['max']['drug']}） | {j['without_pralatrexate']:+.3f} |")
    t4 = R["T4"]
    L += ["", "## T4 协同分数（实测单药参照；单模型口径 = 5 个种子指标的均值）", "",
          "| 系列 | 子集 | Bliss Pearson | Spearman | 分数 RMSE | top 5% 精度 |", "|---|---|---:|---:|---:|---:|"]
    for row in t4["rows"]:
        L.append(f"| {row['series']} | {row['subset']} | {row['pearson']:.3f} | {row['spearman']:.3f} | {row['rmse']:.2f} | "
                 f"{row['precision_top5']:.3f} |")
    L += ["", "| 比较（Bliss Pearson） | 子集 | 差 | 95% 区间（药对 bootstrap，种子 20261013） |", "|---|---|---:|---|"]
    for row in t4["comparisons"]:
        L.append(f"| {row['comparison']} | {row['subset']} | {row['difference']:+.3f} | [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] |")
    t5 = R["T5"]
    L += ["", "## T5 Ours-lite（框架的精简变体，不是主模型）", "",
          "| 子集 | lite 各种子 | lite 均值（SD） | 相对 Ours（药对 bootstrap） | 同种子配对 t | 相对 XGBoost＋身份码 | 相对 DeepSynBa 3 种子 |",
          "|---|---|---:|---|---|---|---|"]
    for sub in SUBSETS:
        v = t5[sub]
        p = v["seed_paired_vs_ours"]
        L.append(f"| {sub} | {' / '.join(f'{x:.3f}' for x in v['per_seed'].values())} | {v['mean']:.3f}（{v['sd']:.3f}） | "
                 f"{pc(v['vs_ours'])} | {p['mean_percent']:+.2f}%，[{p['ci95_percent'][0]:+.2f}, {p['ci95_percent'][1]:+.2f}]，"
                 f"{p['worse_seeds']}/5 更差 | {pc(v['vs_xgb_identity'])} | {pc(v['vs_deepsynba_3_seed'])} |")
    rc = R["reproduction"]
    bad = [r for r in rc["checks"] if not r["ok"]]
    L += ["", f"## 复现核对：{len(rc['checks'])} 项，{'全部通过' if not bad else f'{len(bad)} 项不符'}", ""]
    if bad:
        L += ["| 项目 | 本次 | 已有 | 差 |", "|---|---:|---:|---:|"] + [
            f"| {r['check']} | {r['value']} | {r['reference']} | {r['abs_diff']} |" for r in bad]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------
def analyse(frame, matrices, comb, dsb16, prala, role, info, chk):
    prala_id = next(iter(prala))
    rows, _, _, _ = eg.evaluate(frame, matrices, preds16_of(matrices, comb, dsb16), comb)
    keep = ~(frame.drug_a_id.isin(prala) | frame.drug_b_id.isin(prala)).to_numpy()
    st = {"all": subset_stats(frame, comb, np.ones(len(frame), bool)), "excluding_pralatrexate": subset_stats(frame, comb, keep)}
    for m in comb:                                       # eg.evaluate and subset_stats compute the same per-seed RMSE
        for s in comb[m]:
            chk.near(f"rows vs subset_stats {m} {s}", float(rows[(rows.method == m) & (rows.seed == s)].combination_rmse_percent.iloc[0]),
                     st["all"]["per_seed"][(m, s)])
    T1, T3, T5 = {}, {"subset_size": {}}, {}
    for sub in SUBSETS:
        x = st[sub]
        T1[sub] = {"main_3_seed": verdict(x, comb, OURS, "deepsynba_v2"), "previous_1_seed": verdict(x, comb, OURS, "deepsynba_v2_s10"),
                   "deepsynba_per_seed": seeds_of(x, comb, "deepsynba_v2"), "ours_per_seed": seeds_of(x, comb, OURS)}
        for key, b in (("main_3_seed", "deepsynba_v2"), ("previous_1_seed", "deepsynba_v2_s10")):
            T1[sub][key].update(comparator_rmse=x["point"][b], ours_rmse=x["point"][OURS])
        T3[sub] = {m: {"rmse": x["point"][m], "per_seed": seeds_of(x, comb, m),
                       "seed_sd": float(np.std(list(seeds_of(x, comb, m).values()), ddof=1)) if len(comb[m]) > 1 else None,
                       "ours_vs": compare(x, OURS, m) if m != OURS else None} for m in METHODS}
        T3["subset_size"][sub] = {"pairs": int(len(x["n"])), "points": x["points"]}
        lite = seeds_of(x, comb, "lite")
        T5[sub] = {"per_seed": lite, "mean": x["point"]["lite"], "sd": float(np.std(list(lite.values()), ddof=1)),
                   "vs_ours": compare(x, "lite", OURS), "vs_xgb_identity": compare(x, "lite", "xgb_identity"),
                   "vs_deepsynba_3_seed": compare(x, "lite", "deepsynba_v2"),
                   "seed_paired_vs_ours": paired_t(lite, seeds_of(x, comb, OURS))}
    t2_r, t2_info, weights, lodo_rows = t2_rmse(st["all"], comb, prala_id, chk)
    for name, v in t2_r.items():                         # LODO of pralatrexate == the excluded-subset estimate
        a, b = name.split("_vs_")
        ex = st["excluding_pralatrexate"]["point"]
        chk.near(f"T2 LODO(pralatrexate) == excluded subset, {name}", v["leave_one_drug_out"]["without_pralatrexate"],
                 (ex[a] - ex[b]) / ex[b], 1e-10)
    print(f"[{time.strftime('%H:%M:%S')}] RMSE part done; synergy bootstrap (several minutes)", flush=True)
    syn = synergy(frame, matrices, comb, dsb16)
    t2_b, lodo_b = t2_bliss(syn, st["all"], weights, prala_id)
    pts, cps = syn["points"], syn["comparisons"]
    T4 = {"rows": [dict(series=g, subset=sub, **{m: syn_value(pts, g, sub, "Bliss", m) for m in ("pearson", "spearman", "rmse", "precision_top5")})
                   for sub in ("all", "excluding_pralatrexate") for g in ("BS", "GX", "GX_ens5", "XI", "DSB", "DSB_s11", "DSB_s12", "DSB3")],
          "comparisons": [dict(comparison=c, subset=sub, **{k: float(syn_comp(cps, c, sub)[k]) for k in ("difference", "ci_low", "ci_high")})
                          for sub in ("all", "excluding_pralatrexate")
                          for c in ("C7 BS - XGB_gate", "C8 XGB_gate - DSB(observed ref)", "C1 BS - DSB(observed ref)",
                                    "C9 BS - DSB3(observed ref)", "C5 BS - XGB_identity")]}
    R = {"role": role, "registered": "DEVIATIONS.md D19 (card J, docs/T_BATCH_CARD_20261025.md)",
         "evaluable": {"matrices": len(matrices), "combination_points": int(len(frame)), "pairs": int(len(st["all"]["n"]))},
         "T1": T1, "T2": {"rmse": t2_r, "bliss_pearson": t2_b, "info": t2_info}, "T3": T3, "T4": T4, "T5": T5,
         "pearson_surface_all16": {m: {k: float(rows[rows.method == m][k].mean()) for k in
                                       ("combination_pearson", "combination_surface_macro_rmse_percent", "all16_rmse_percent")} for m in METHODS},
         "bootstrap": {"primary": {"unit": "drug pair", "draws": eg.DRAWS, "seed": eg.BOOT_SEED,
                                   "excluded_subset": "own pair bootstrap, same seed (as evaluate_v2.score)"},
                       "synergy": {"unit": "drug pair", "draws": sc.BOOT_N, "seed": sc.BOOT_SEED}}}
    return R, st, syn, rows, lodo_rows + lodo_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="validation rehearsal; never reads split 3")
    ap.add_argument("--out", type=Path, help="output directory of the dry run")
    a = ap.parse_args()
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B`")
    if a.dry_run:
        if a.out is None:
            raise SystemExit("--dry-run needs --out")
        out = a.out.resolve()
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty")
        common.load_test = eg.deepsynba_combination = forbidden          # tripwires
    else:
        if a.out is not None:
            raise SystemExit("the test run always writes runs/model_improvement/bisurfgnn_v2_dev/t_batch/")
        if MARKER.exists():
            raise SystemExit("The T batch has already been scored; the NCI public test is not read again")
        out = OUT_TEST
        if out.exists():
            raise SystemExit(f"{out} exists")
    role = "validation (dry run)" if a.dry_run else "test"
    start = time.monotonic()
    cfg = rp.effective_config()
    if a.dry_run:
        bindings = gate.check_bindings()
    else:
        bindings, dsb_eval, _, form, _, _ = eg.preconditions()   # input hashes incl. the DeepSynBa seed-10 file
        if form != "additive" or not dsb_eval.get("reproduced"):
            raise RuntimeError("Unexpected gate form selection or DeepSynBa acceptance state")
    hashes = preconditions(a.dry_run)
    test_ready = {"alignment": pretest_alignment(),       # ID members only; the dry run rehearses the test-side checks
                  "inputs_sha256": hashes if not a.dry_run else preconditions(False)}
    drugs_tab = pd.read_csv(gate.S1 / "drug_mapping_used.csv")
    prala = {r.canonical_drug_id for r in drugs_tab.itertuples() if str(r.deepsynba_drug).strip().lower() == "pralatrexate"}
    if len(prala) != 1:
        raise RuntimeError(f"Expected one pralatrexate ID, found {prala}")
    chk = Checks()
    check_test_docs(chk)
    with threadpool_limits(limits=cfg["xgboost"]["threads"]):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frames, banks = eg.dev_setup(cfg)
        models, lbanks, ext = lite_setup(cfg, frames, banks)
        lite_val = lite_predict(models, lbanks, ext, cfg, frames["validation"], "validation")
        lite_diff = {s: float(np.max(np.abs(lite_val[s] - csv_pred(LITE[s] / "validation_predictions.csv", frames["validation"]))))
                     for s in SEEDS}
        if max(lite_diff.values()) > 1e-4:                 # percent scale; evaluate_v2's 1e-6 on the ratio scale
            raise RuntimeError(f"Reloaded LITE models do not reproduce their validation predictions: {lite_diff}")
        print(f"[{time.strftime('%H:%M:%S')}] LITE validation re-prediction max abs diff (percent): {max(lite_diff.values()):.2e}", flush=True)
        out.mkdir(parents=True, exist_ok=a.dry_run)
        try:
            if a.dry_run:
                frame, matrices, comb, dsb16, info = load_validation(frames["validation"])
                comb["lite"] = lite_val
            else:
                with MARKER.open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps({"created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                             "registered": "DEVIATIONS.md D19"}) + "\n")
                frame, matrices, comb, dsb16, info = load_test()
                comb["lite"] = lite_predict(models, lbanks, ext, cfg, frame, "test")
                np.savez_compressed(out / "lite_test_predictions.npz", **{f"seed_{s}": comb["lite"][s] for s in SEEDS},
                                    point_ids=frame.source_combodrugseq.to_numpy())
            comb = finish_comb(comb, dsb16)
            R, st, syn, rows, lodo = analyse(frame, matrices, comb, dsb16, prala, role, info, chk)
        except BaseException:
            (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        try:   # a crash while checking must not lose the one-shot results: it is recorded as a failed check
            if a.dry_run:
                reproduce_validation(chk, st, comb, syn, R["T5"])
            else:
                reproduce_test(chk, frame, matrices, comb, dsb16, prala, st, syn, info)
        except Exception:
            chk.same(f"reproduction step crashed: {traceback.format_exc()}", False)
    R["test_side_readiness"] = test_ready
    R["reproduction"] = {"all_ok": chk.ok, "checks": chk.rows}
    R["inputs_sha256"], R["bindings"] = hashes, bindings
    R["lite_validation_max_abs_diff_percent"] = lite_diff
    R["notes"] = {k: v for k, v in info.items() if k != "saved_ensemble"}
    R["test_response_reads"] = 0 if a.dry_run else 1
    R["wall_seconds"] = time.monotonic() - start
    rows.to_csv(out / "per_seed_metrics.csv", index=False)
    syn["points"].to_csv(out / "synergy_point_estimates.csv", index=False)
    syn["comparisons"].to_csv(out / "synergy_paired_comparisons.csv", index=False)
    syn["seeds"].to_csv(out / "synergy_seed_level.csv", index=False)
    pd.DataFrame(lodo).to_csv(out / "leave_one_drug_out.csv", index=False)
    with (out / "t_batch_result.json").open("x", encoding="utf-8") as handle:
        json.dump(R, handle, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (out / "tables.md").write_text(tables_md(R, role), encoding="utf-8")
    bad = [r for r in chk.rows if not r["ok"]]
    print(json.dumps({"role": role, "out": str(out), "T1_main": {k: R["T1"]["all"]["main_3_seed"][k] for k in
                      ("relative_change", "ci95", "ours_seeds_below_comparator_mean", "passed")},
                      "reproduction_checks": len(chk.rows), "failed": bad, "wall_seconds": round(R["wall_seconds"])},
                     indent=1, default=float), flush=True)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

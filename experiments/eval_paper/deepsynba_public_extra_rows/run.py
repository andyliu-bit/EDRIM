"""Eval-paper supplement E: two no-training rows on DeepSynBa's public new-combination split (NCI).

Card and results: C:/LHZ/research/bisurfgnn/docs/EVAL_DEEPSYNBA_PUBLIC_EXTRA_ROWS_20261017.md

(1) Group-offset residual probe P4 (V2_GAP_DIAGNOSIS D3: position + drug x dose + drug x cell + cell x position offsets
    on y - lookup HSA, ridge lambda = 100), fitted on split 1 only, scored once on split 3:
      S54 (gate set, MDA-MB-468 excluded): the D3 fit itself (S1 point table, S1 lookup);
      S55 (all 55,120 test matrices): same design refit on all split-1 matrices with the all-cell lookup of
          external_deepsynba_lookup_v1 (MDA-MB-468 is its own cell identity).
(2) comboKR 2.0 frozen predictions converted to DeepSynBa's 16-cell metric (author Hill mono, control 100, saved 3x3);
    sensitivity: all 16 cells from comboKR's own predicted BRAID parameters.
Nothing is trained besides the closed-form ridge probe; split 3 is read once, after the split-2 self-check.

  .venv/Scripts/python.exe -I experiments/eval_paper/deepsynba_public_extra_rows/run.py
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/model_improvement/v2_gap_diagnosis"))
import d3_probes as d3  # noqa: E402  (design / fit_probe of the D3 probes, unchanged)

MAIN = ROOT.parent / "bisurfgnn"
spec = importlib.util.spec_from_file_location("ext_lookup", MAIN / "experiments/audit_revision/external_deepsynba_lookup_baseline.py")
ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)

SRC = MAIN / "data/raw/external_benchmarks/deepsynba/data"
GATE = ROOT / "runs/model_improvement/public_split_gate_v1"
S1 = GATE / "s1"
V2 = ROOT / "runs/model_improvement/bisurfgnn_v2_dev"
DSB = MAIN / "runs/audit_revision/deepsynba_repro_v2_20261012/full"
CKR = ROOT / "runs/model_improvement/combokr2_original_v1/public_split"
HILLS = MAIN / "data/raw/external_benchmarks/combokr2_zenodo_14509938/almanac_fitted_braids_and_hills/almanac_fitted_hills.pkl"
OUT = ROOT / "runs/eval_paper/deepsynba_public_extra_rows"
BLOCKS, LAM, EXCL = ["position", "drug_dose", "drug_cell", "cell_position"], 100.0, "MDA-MB-468"
DRAWS, BOOT_SEED = 2000, 20261009
EDGE = ~np.pad(np.ones((3, 3), bool), ((1, 0), (1, 0)))  # 7 single-agent + control cells of a 4x4 matrix


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def metrics(y16, p16):
    e = p16[:, 1:, 1:] - y16[:, 1:, 1:]
    return dict(combination_rmse=rmse(e), combination_pearson=corr(p16[:, 1:, 1:], y16[:, 1:, 1:]),
                combination_surface_macro=float(np.sqrt(np.mean(e.reshape(-1, 9) ** 2, axis=1)).mean()),
                all16_rmse=rmse(p16 - y16), all16_pearson=corr(p16, y16),
                mono_control_rmse=rmse((p16 - y16)[:, EDGE]))


def arrays(data, keys):
    return (np.stack([np.asarray(data[k]["matrix"], float) for k in keys]),
            np.stack([np.asarray(data[k]["x1_dose"], float) for k in keys]),
            np.stack([np.asarray(data[k]["x2_dose"], float) for k in keys]))


def points(part, m, x1, x2, table, drug_id):
    """Combination points (9 per matrix, row-major i, j) anchored on a split-1 lookup, plus the lookup 4x4 matrices."""
    tally = defaultdict(int)
    ma = np.array([[ext.mono(table, c, a, x[i], tally) for i in (1, 2, 3)] for c, a, x in zip(part.cell, part.drug1, x1)])
    mb = np.array([[ext.mono(table, c, b, x[j], tally) for j in (1, 2, 3)] for c, b, x in zip(part.cell, part.drug2, x2)])
    n = len(part)
    da, db = part.drug1.map(drug_id).to_numpy(), part.drug2.map(drug_id).to_numpy()
    frame = pd.DataFrame({
        "matrix_key": np.repeat(part.key.to_numpy(), 9), "deepsynba_cell": np.repeat(part.cell.to_numpy(), 9),
        "drug_a_id": np.repeat(da, 9), "drug_b_id": np.repeat(db, 9),
        "pair_id": np.repeat(np.array(["|".join(sorted(p)) for p in zip(da, db)]), 9),
        "row_i": np.tile(np.repeat([1, 2, 3], 3), n), "col_j": np.tile([1, 2, 3], 3 * n),
        "hsa": np.minimum(np.repeat(ma, 3, axis=1), np.tile(mb, (1, 3))).ravel(),
        "response_percent": m[:, 1:, 1:].reshape(-1)})
    look16 = np.empty((n, 4, 4)); look16[:, 0, 0] = table[2]; look16[:, 1:, 0] = ma; look16[:, 0, 1:] = mb
    return frame, look16, dict(tally)


def fit(frame, target_col="response_percent", hsa_col="hsa"):
    drugs = sorted(set(frame.drug_a_id) | set(frame.drug_b_id)); cells = sorted(set(frame.deepsynba_cell))
    x = d3.design(frame, cells, drugs, BLOCKS)
    coef, mean, iters = d3.fit_probe(x, (frame[target_col] - frame[hsa_col]).to_numpy(float), LAM)
    return dict(coef=coef, mean=mean, iters=iters, drugs=drugs, cells=cells, coefficients=int(x.shape[1]))


def predict(model, frame, hsa_col="hsa"):
    return frame[hsa_col].to_numpy(float) + model["mean"] + d3.design(frame, model["cells"], model["drugs"], BLOCKS) @ model["coef"]


def boot_pairs(pair_ids, preds, y):
    """Drug-pair bootstrap exactly as evaluate_gate.bootstrap (sorted pairs, same rng call sequence)."""
    uniq, inv = np.unique(pair_ids, return_inverse=True)
    n = np.bincount(inv).astype(float)
    rng = np.random.default_rng(BOOT_SEED)
    w = np.stack([np.bincount(rng.integers(0, len(n), len(n)), minlength=len(n)) for _ in range(DRAWS)]).astype(float)
    return len(uniq), {k: np.sqrt((w @ np.bincount(inv, weights=(p - y) ** 2)) / (w @ n)) for k, p in preds.items()}


def compare(point, boots, a, b):
    d = (boots[a] - boots[b]) / boots[b]
    return dict(relative_change=(point[a] - point[b]) / point[b], ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def braid(d1, d2, p):
    """comboKR2 braid_model_with_raw_c_input (delta = 1, author's -|dE| convention), vectorised over matrices."""
    E0, E1, E2, E3, h1, h2, C1, C2, kap = (p[:, k].reshape((-1,) + (1,) * (d1.ndim - 1)) for k in range(9))
    dE = -np.abs(np.stack([E1 - E0, E2 - E0, E3 - E0]))
    top = dE.min(axis=0)                       # all non-positive: the largest |dE|
    a, b, h = dE[0] / top, dE[1] / top, np.sqrt(h1 * h2)
    p1, p2 = np.power(d1 / C1, h1), np.power(d2 / C2, h2)
    q1, q2 = np.power(a * p1 / (1 + (1 - a) * p1), 1 / h), np.power(b * p2 / (1 + (1 - b) * p2), 1 / h)
    D = q1 + q2 + kap * np.sqrt(q1 * q2)
    with np.errstate(divide="ignore"):
        res = E0 + top / (1 + np.power(D, -h))
    return np.where(D == 0, 100.0, res)


def main():
    if not sys.flags.isolated:
        raise SystemExit("run with python -I")
    OUT.mkdir(parents=True, exist_ok=False)
    gate_eval = json.loads((GATE / "s5/evaluation.json").read_text(encoding="utf-8"))
    look_v1 = json.loads((MAIN / "runs/audit_revision/external_deepsynba_lookup_v1/result.json").read_text(encoding="utf-8"))
    dsb_eval = json.loads((DSB / "evaluation.json").read_text(encoding="utf-8"))
    ckr_eval = json.loads((V2 / "combokr_test/test_evaluation.json").read_text(encoding="utf-8"))
    per_seed = pd.read_csv(GATE / "s5/per_seed_metrics.csv").set_index("method")
    d3_ref = pd.read_csv(ROOT / "runs/model_improvement/v2_gap_diagnosis/probes/entity_probes.csv")
    d3_p4 = float(d3_ref[(d3_ref.probe == "P4") & (d3_ref.lam == LAM)].val_rmse.iloc[0])
    drug_id = dict(pd.read_csv(S1 / "drug_mapping_used.csv")[["deepsynba_drug", "canonical_drug_id"]].itertuples(index=False))
    res = {"card": "bisurfgnn/docs/EVAL_DEEPSYNBA_PUBLIC_EXTRA_ROWS_20261017.md", "lambda": LAM, "blocks": BLOCKS}

    # ---- development: the S54 fit is the D3 fit; split-2 self-check before split 3 is touched -------------------
    cols = ["matrix_key", "drug_a_id", "drug_b_id", "deepsynba_cell", "row_i", "col_j", "response_percent", "hsa_lookup_percent"]
    tr1 = pd.read_parquet(S1 / "points_train.parquet", columns=cols)
    m54 = fit(tr1, hsa_col="hsa_lookup_percent")
    s1_check = dict(keys=tr1.matrix_key.to_numpy()[::9].copy(), y=tr1.response_percent.to_numpy(float),
                    hsa=tr1.hsa_lookup_percent.to_numpy(float))
    del tr1
    va1 = pd.read_parquet(S1 / "points_validation.parquet", columns=cols)
    val54 = rmse(predict(m54, va1, "hsa_lookup_percent") - va1.response_percent.to_numpy(float))
    assert abs(val54 - d3_p4) < 1e-6, (val54, d3_p4)
    res["dev_split2_S54"] = dict(p4=val54, d3_saved=d3_p4, lookup_hsa=rmse(va1.hsa_lookup_percent - va1.response_percent),
                                 matrices=len(va1) // 9, lsqr_iterations=m54["iters"], coefficients=m54["coefficients"])
    del va1
    print("split-2 self-check passed", res["dev_split2_S54"], flush=True)

    meta = pd.read_csv(SRC / "nci_synergy_metadata.csv")
    meta["key"] = meta.drug1 + "_" + meta.drug2 + "_" + meta.cell
    data = pickle.load((SRC / "nci_synergy_dose_response.pickle").open("rb"))
    sp = {s: meta[meta.split == s].reset_index(drop=True) for s in (1, 2, 3)}
    keys1 = list(zip(sp[1].drug1, sp[1].drug2, sp[1].cell))
    look_full = ext.build_lookup(keys1, data)
    look_s1 = ext.build_lookup([k for k in keys1 if k[2] != EXCL], data)
    m1, x11, x21 = arrays(data, sp[1].key)
    tr55, _, _ = points(sp[1], m1, x11, x21, look_full, drug_id)
    del m1, x11, x21
    keep = tr55.deepsynba_cell.to_numpy() != EXCL
    assert np.array_equal(tr55.matrix_key.to_numpy()[keep][::9], s1_check["keys"])
    assert np.array_equal(tr55.response_percent.to_numpy()[keep], s1_check["y"])
    assert np.abs(tr55.hsa.to_numpy()[keep] - s1_check["hsa"]).max() <= 1e-9
    del s1_check, keep
    m55 = fit(tr55)
    res["fit_S55"] = dict(train_matrices=len(tr55) // 9, lsqr_iterations=m55["iters"], coefficients=m55["coefficients"],
                          cells=len(m55["cells"]))
    del tr55
    m2, x12, x22 = arrays(data, sp[2].key)
    va55, _, _ = points(sp[2], m2, x12, x22, look_full, drug_id)
    res["dev_split2_S55"] = dict(p4=rmse(predict(m55, va55) - va55.response_percent), matrices=len(va55) // 9,
                                 lookup_hsa=rmse(va55.hsa - va55.response_percent))
    del va55, m2, x12, x22
    print("dev done", res["fit_S55"], res["dev_split2_S55"], flush=True)

    # ---- test (split 3), read once -------------------------------------------------------------------------------
    te = sp[3]
    m3, x13, x23 = arrays(data, te.key)
    del data
    f55, look55, tally55 = points(te, m3, x13, x23, look_full, drug_id)
    sub = (te.cell != EXCL).to_numpy()
    te54 = te[sub].reset_index(drop=True)
    f54, look54, tally54 = points(te54, m3[sub], x13[sub], x23[sub], look_s1, drug_id)
    y55, y54 = m3, m3[sub]
    assert len(te) == 55120 and len(te54) == 54196 and EXCL not in set(f54.deepsynba_cell)
    saved = np.load(GATE / "s5/test_combination_predictions_lookup_hsa.npz")
    assert np.array_equal(saved["matrix_keys"], f54.matrix_key.to_numpy().astype(str))
    assert np.abs(saved["seed_0"] - f54.hsa.to_numpy()).max() <= 1e-9
    assert tally55 == look_v1["single_agent_lookup_coverage"]["train"], tally55

    z = np.load(DSB / "test_predictions.npz")
    idx = {"_".join(k): i for i, k in enumerate(z["keys"].tolist())}
    rows = np.array([idx[k] for k in te.key])
    dsb16, truth = z["predictions"].astype(float)[rows], z["truth"].astype(float)[rows]
    assert np.abs(truth - m3).max() <= 1e-3                      # same layout as the release (gate found "as-is")

    p4_55, p4_54 = predict(m55, f55), predict(m54, f54)
    def full16(look, comb):
        p = look.copy(); p[:, 1:, 1:] = np.asarray(comb).reshape(-1, 3, 3); return p
    rows_out = {
        "S55": {"lookup_hsa": metrics(y55, full16(look55, f55.hsa)), "deepsynba_v2_seed10": metrics(y55, dsb16),
                "p4_full_fit": metrics(y55, full16(look55, p4_55))},
        "S54": {"lookup_hsa": metrics(y54, full16(look54, f54.hsa)), "deepsynba_v2_seed10": metrics(y54, dsb16[sub]),
                "p4_s1_fit": metrics(y54, full16(look54, p4_54))},
        "S54_consistency_full_fit": {"lookup_hsa_full_lookup": metrics(y54, full16(look55[sub], f55.hsa.to_numpy().reshape(-1, 9)[sub])),
                                     "p4_full_fit": metrics(y54, full16(look55[sub], p4_55.reshape(-1, 9)[sub]))}}
    r55, r54 = rows_out["S55"], rows_out["S54"]
    for got, want in [(r55["lookup_hsa"]["combination_rmse"], look_v1["primary_entries"]["combination_only"]["rmse"]),
                      (r55["lookup_hsa"]["all16_rmse"], look_v1["primary_mono_lookup_hsa"]["rmse"]),
                      (r55["deepsynba_v2_seed10"]["combination_rmse"], dsb_eval["combination"]["rmse"]),
                      (r55["deepsynba_v2_seed10"]["all16_rmse"], dsb_eval["overall_primary"]["rmse"]),
                      (r54["lookup_hsa"]["combination_rmse"], gate_eval["point_estimates_combination_rmse_percent"]["lookup_hsa"]),
                      (r54["lookup_hsa"]["all16_rmse"], gate_eval["all16_rmse_percent"]["lookup_hsa"]),
                      (r54["lookup_hsa"]["combination_surface_macro"], per_seed.loc["lookup_hsa", "combination_surface_macro_rmse_percent"]),
                      (r54["deepsynba_v2_seed10"]["combination_rmse"], gate_eval["point_estimates_combination_rmse_percent"]["deepsynba_v2"]),
                      (r54["deepsynba_v2_seed10"]["all16_rmse"], gate_eval["all16_rmse_percent"]["deepsynba_v2"])]:
        assert abs(got - want) < 1e-6, (got, want)   # float32 DeepSynBa file: 1e-6 rather than 1e-9

    boots = {}
    for tag, frame, preds in [("S55", f55, {"lookup_hsa": f55.hsa.to_numpy(), "p4": p4_55, "dsb": dsb16[:, 1:, 1:].reshape(-1)}),
                              ("S54", f54, {"lookup_hsa": f54.hsa.to_numpy(), "p4": p4_54, "dsb": dsb16[sub][:, 1:, 1:].reshape(-1)})]:
        y = frame.response_percent.to_numpy(float)
        n_pairs, b = boot_pairs(frame.pair_id.to_numpy(), preds, y)
        point = {k: rmse(p - y) for k, p in preds.items()}
        boots[tag] = dict(pairs=n_pairs, p4_vs_lookup_hsa=compare(point, b, "p4", "lookup_hsa"),
                          dsb_vs_lookup_hsa=compare(point, b, "dsb", "lookup_hsa"),
                          p4_vs_dsb=compare(point, b, "p4", "dsb"),
                          gap_closed_share=(point["lookup_hsa"] - point["p4"]) / (point["lookup_hsa"] - point["dsb"]))
    ref = gate_eval["comparisons"]["deepsynba_v2_vs_lookup_hsa"]
    assert boots["S54"]["pairs"] == 979 and np.allclose(boots["S54"]["dsb_vs_lookup_hsa"]["ci95"], ref["ci95"], atol=1e-6)
    res.update(test=rows_out, bootstrap=boots, lookup_fallbacks={"S55": tally55, "S54": tally54},
               bootstrap_setup={"unit": "test drug pair", "draws": DRAWS, "seed": BOOT_SEED})
    np.savez_compressed(OUT / "p4_test_combination_predictions.npz", s55=p4_55, s55_matrix_keys=f55.matrix_key.to_numpy().astype(str),
                        s54=p4_54, s54_matrix_keys=f54.matrix_key.to_numpy().astype(str))
    del f55, look55

    # ---- (2) comboKR 2.0 on 16 cells (S54 only: comboKR predicted these 54,196 matrices) ---------------------------
    zk, q = np.load(CKR / "test/pred_iters10.npz"), np.load(CKR / "query_test.npz")
    assert np.array_equal(zk["matrix_key"], te54.key.to_numpy().astype(str)) and np.array_equal(q["matrix_key"], zk["matrix_key"])
    x1, x2 = x13[sub], x23[sub]
    assert np.allclose(q["c1"][:, [0, 3, 6]], x1[:, 1:], rtol=1e-12, atol=0) and np.allclose(q["c2"][:, :3], x2[:, 1:], rtol=1e-12, atol=0)
    comb = zk["pred"].astype(float)
    assert abs(rmse(comb.ravel() - f54.response_percent.to_numpy()) - ckr_eval["point"]["combokr2"]) < 1e-9
    assert np.abs(braid(q["c1"], q["c2"], zk["params"]) - comb).max() < 1e-6
    hills = pickle.load(HILLS.open("rb"))
    hk = {(c, str(d)): (e, h, k) for c, d, e, h, k in zip(hills.cell, hills.drug, hills.Emax, hills.h, hills.C)}
    par_a = np.array([hk[(c, a)] for c, a in zip(q["cdd"][:, 0], q["cdd"][:, 1])])
    par_b = np.array([hk[(c, b)] for c, b in zip(q["cdd"][:, 0], q["cdd"][:, 2])])
    hill = lambda c, p: 100.0 + (p[:, :1] - 100.0) * c ** p[:, 1:2] / (p[:, 2:3] ** p[:, 1:2] + c ** p[:, 1:2])  # helpers.hill_equation, E0 = 100
    ck16 = np.empty_like(y54); ck16[:, 0, 0] = 100.0
    ck16[:, 1:, 0], ck16[:, 0, 1:], ck16[:, 1:, 1:] = hill(x1[:, 1:], par_a), hill(x2[:, 1:], par_b), comb.reshape(-1, 3, 3)
    ck_braid16 = braid(x1[:, :, None], x2[:, None, :], zk["params"])
    assert np.abs(ck_braid16[:, 1:, 1:] - ck16[:, 1:, 1:]).max() < 1e-6 and (ck_braid16[:, 0, 0] == 100).all()
    res["combokr2_16cell"] = dict(
        set="S54", matrices=len(te54), entries=int(ck16.size),
        primary_hill_mono_control100=metrics(y54, ck16), sensitivity_own_braid_all16=metrics(y54, ck_braid16),
        hill_vs_braid_mono_max_abs=float(np.abs(ck16 - ck_braid16)[:, EDGE].max()),
        deepsynba_table1=dict(rmse=23.86, pearson=0.65))
    with (OUT / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(res, handle, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("test", "bootstrap", "combokr2_16cell")}, indent=1, default=float))


if __name__ == "__main__":
    main()

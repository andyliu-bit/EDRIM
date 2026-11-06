"""Comparator diagnostics D1-D3 (CPU, validation-only, no training).

D1  comboKR 2.0 representation vs learner: neutral BRAID (kappa=0, E3=min(E1,E2), the
    author's `predict_with_neutral_braid` parameterisation) built from the saved
    single-agent Hill fits, scored next to observed HSA and the saved comboKR predictions;
    Hill-fit pathology counts.
D2  stratification of validation surfaces by >1 responses; diagnostic re-score against
    the clipped target min(y, 1).
D3  comboKR learned part: (prediction - neutral) vs (truth - neutral).

Reads only: fold-0 validation/train shards of the unclipped OneNewScaffoldAnchored-v1
benchmark restricted to the frozen comparator sample IDs, the clipped benchmark's fold-0
validation shard (sanity check only), saved predictions/fit artefacts under
runs/model_improvement. test.parquet is never opened. fitted_training.npz is opened only
inside an isolated `python -I` subprocess with allow_pickle=False.

Writes only under reports/diagnostics_20261011/.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

T0 = time.perf_counter()
ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pandas as pd
# scipy.stats aborts in this Windows venv (OMP Error #15: duplicate libiomp5md), so NumPy is used.


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def spearman(a, b):
    return pearson(pd.Series(np.asarray(a)).rank().to_numpy(), pd.Series(np.asarray(b)).rank().to_numpy())


RUNS =ROOT / "runs/model_improvement"
OUT = ROOT / "reports/diagnostics_20261011"
DATA = ROOT.parent / "bisurfgnn/data/processed"
UNCLIPPED = DATA / "one_new_scaffold_benchmark_unclipped/learned_point_shards/fold_0"
CLIPPED = DATA / "one_new_scaffold_benchmark/learned_point_shards/fold_0"
COMBOKR_DIR = RUNS / "comparator_gpu_v2/combokr2_dense_minres_seed_None_primary"
SEEDS = (41, 42, 43, 44, 45)
MODEL_DIRS = {
    "DD-PRiSM adapted (dd_final)": [RUNS / f"algorithm_search_v1/dd_final/ddprism_hallmark_observed_mono_seed_{s}_primary" for s in SEEDS],
    "comboKR2 adapted": [COMBOKR_DIR],
    "BiSurfGNN v2 (bisurf_mono_v2)": [RUNS / f"factorial_exchange_v1/bisurf_mono_v2_seed_{s}_primary" for s in SEEDS],
    "XGB HSA-residual d10_m1 (T)": [RUNS / f"algorithm_search_v1/xgb_final/xgb_residual_d10_m1_seed_{s}_primary" for s in SEEDS],
}
for path in (UNCLIPPED, CLIPPED):
    assert "test" not in path.name


def log(*args):
    print(*args, flush=True)


# --------------------------------------------------------------------------- data
def load_role(shard: Path, ids_csv: Path) -> pd.DataFrame:
    assert shard.name in {"train.parquet", "validation.parquet"}, shard
    ids = pd.read_csv(ids_csv)
    frame = pd.read_parquet(shard).set_index("source_combodrugseq").loc[ids.source_combodrugseq].reset_index()
    assert (frame.surface_replicate_id.to_numpy() == ids.surface_replicate_id.to_numpy()).all()
    return frame


va = load_role(UNCLIPPED / "validation.parquet", RUNS / "comparator_gpu_v2/validation_ids.csv")
tr = load_role(UNCLIPPED / "train.parquet", RUNS / "comparator_gpu_v2/train_ids.csv")
y = va.response.to_numpy(float)
surf = va.surface_replicate_id.to_numpy()
log(f"validation points={len(va)} surfaces={va.surface_replicate_id.nunique()} train points={len(tr)}")


def per_surface_rmse(target, prediction, surfaces=surf):
    e = pd.DataFrame({"s": surfaces, "e": (np.asarray(prediction, float) - np.asarray(target, float)) ** 2})
    return np.sqrt(e.groupby("s", sort=True).e.mean())


def metrics(target, prediction, surfaces=surf):
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    ps = per_surface_rmse(target, prediction, surfaces)
    return dict(surface_macro_rmse=float(ps.mean()), point_rmse=float(np.sqrt(np.mean((prediction - target) ** 2))),
                mae=float(np.mean(np.abs(prediction - target))), bias=float(np.mean(prediction - target)),
                pearson=pearson(prediction, target) if np.std(prediction) > 0 else float("nan"),
                n_points=int(len(target)), n_surfaces=int(len(ps)))


def load_predictions(directory: Path) -> np.ndarray:
    p = pd.read_csv(directory / "predictions.csv")
    assert (p.source_row.to_numpy() == va.source_combodrugseq.to_numpy()).all(), directory
    assert (p.surface_id.to_numpy() == surf).all(), directory
    assert np.max(np.abs(p.target.to_numpy() - y)) < 1e-6, directory
    return p.prediction.to_numpy(float)


predictions = {name: [load_predictions(d) for d in dirs] for name, dirs in MODEL_DIRS.items()}
for s in (41, 42):  # v2 seeds 41/42 also exist under algorithm_search_v1/screen_b1; must be identical
    alt = load_predictions(RUNS / f"algorithm_search_v1/screen_b1/bisurf_mono_v2_seed_{s}_primary")
    diff = float(np.max(np.abs(alt - predictions["BiSurfGNN v2 (bisurf_mono_v2)"][SEEDS.index(s)])))
    log(f"v2 seed {s} factorial_exchange_v1 vs screen_b1 max|diff|={diff:.3g}")
    assert diff < 1e-6

hsa = np.minimum(va.mono_response_a.to_numpy(float), va.mono_response_b.to_numpy(float))
bliss = va.mono_response_a.to_numpy(float) * va.mono_response_b.to_numpy(float)

# ----------------------------------------------------------------- comboKR neutral BRAID
hills =pd.read_csv(COMBOKR_DIR / "hills.csv")
hill_lookup = {(r.cell, r.drug): (float(r.Emax), float(r.h), float(r.C)) for r in hills.itertuples()}


def molar(dose_model):
    return (10.0 ** np.asarray(dose_model, float) - 1.0) * 1e-12  # identical to run_comparator_pilot.py


def hill_curve(dose, emax, h, c):
    power = (dose / c) ** h
    return (100.0 + (emax - 100.0) * power / (1.0 + power)) / 100.0


def braid_local(d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta=1.0):
    """Same algebra as the author's braid_model_with_raw_c_input (verified below in a subprocess)."""
    delta_es = -np.abs(np.array([E1 - E0, E2 - E0, E3 - E0]))
    max_delta = delta_es[np.argmax(np.abs(delta_es))]
    da_ratio, db_ratio = delta_es[0] / max_delta, delta_es[1] / max_delta
    h = np.sqrt(h1 * h2); power = 1.0 / (delta * h)
    p1 = np.power(d1 / C1, h1); p2 = np.power(d2 / C2, h2)
    D1 = da_ratio * p1 / (1 + (1 - da_ratio) * p1); D2 = db_ratio * p2 / (1 + (1 - db_ratio) * p2)
    a, b = np.power(D1, power), np.power(D2, power)
    D = a + b + kappa * np.sqrt(a * b)
    res = E0 + max_delta / (1 + np.power(D, -delta * h))
    res[D == 0] = 100
    return res


neutral = np.empty(len(va)); hill_a = np.empty(len(va)); hill_b = np.empty(len(va))
author_jobs = []
for key, idx in va.groupby(["cell_id", "drug_a_id", "drug_b_id"], sort=True).indices.items():
    cell, a, b = key
    e1, h1, c1 = hill_lookup[(cell, a)]; e2, h2, c2 = hill_lookup[(cell, b)]
    da, db = molar(va.dose_a_model.to_numpy()[idx]), molar(va.dose_b_model.to_numpy()[idx])
    params = [100.0, e1, e2, min(e1, e2), h1, h2, c1, c2, 0.0]  # = predict_with_neutral_braid
    neutral[idx] = braid_local(da, db, *params) / 100.0
    hill_a[idx] = hill_curve(da, e1, h1, c1); hill_b[idx] = hill_curve(db, e2, h2, c2)
    author_jobs.append(dict(idx=idx.tolist(), da=da.tolist(), db=db.tolist(), params=params))
assert np.isfinite(neutral).all()

# Verify against the unmodified author function in a separate process. The author module imports
# `synergy` -> torch; mixing torch's OpenMP runtime with NumPy/MKL's in this process aborts on this
# Windows venv (OMP Error #15), so the check runs out of process (-B: no bytecode writes).
author_code = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("combokr2_braid_author", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np
jobs = json.load(sys.stdin); out = {}
for job in jobs:
    r = m.braid_model_with_raw_c_input([np.array(job["da"]), np.array(job["db"])], *job["params"])
    for i, v in zip(job["idx"], np.atleast_1d(r)): out[i] = float(v) / 100.0
print(json.dumps([out[i] for i in range(len(out))]))
"""
author_proc = subprocess.run(
    [sys.executable, "-B", "-c", author_code,
     str(ROOT / "data/raw/model_comparators/combokr2/comboKR2/braid_surface_model_for_fit.py")],
    input=json.dumps(author_jobs), capture_output=True, text=True, check=True)
neutral_author = np.array(json.loads(author_proc.stdout.strip().splitlines()[-1]))
author_max_diff = float(np.max(np.abs(neutral_author - neutral)))
log(f"neutral BRAID local vs author braid_model_with_raw_c_input max|diff|={author_max_diff:.3g}")
assert author_max_diff < 1e-12
combokr = predictions["comboKR2 adapted"][0]

# ---------------------------------------------------------------------------- D1
cov_parts = []  # reconstruct the Hill inputs exactly as mono_covariates() did (train+validation sample)
for frame in (tr, va):
    for arm in ("a", "b"):
        cov_parts.append(pd.DataFrame({"drug": frame[f"drug_{arm}_id"], "cell": frame.cell_id,
                                       "dose": molar(frame[f"dose_{arm}_model"]), "mono": frame[f"mono_response_{arm}"] * 100.0}))
cov = pd.concat(cov_parts).groupby(["cell", "drug", "dose"], as_index=False).mono.median()
cov_lookup = {(c, d, x): m / 100.0 for c, d, x, m in cov.itertuples(index=False)}
med_a = np.array([cov_lookup[(c, d, x)] for c, d, x in zip(va.cell_id, va.drug_a_id, molar(va.dose_a_model))])
med_b = np.array([cov_lookup[(c, d, x)] for c, d, x in zip(va.cell_id, va.drug_b_id, molar(va.dose_b_model))])

d1_rows = []
for name, pred in [("observed HSA min(mono_a,mono_b)", hsa), ("observed Bliss mono_a*mono_b", bliss),
                   ("median-mono HSA (Hill input: median per cell,drug,dose)", np.minimum(med_a, med_b)),
                   ("Hill-HSA min(Hill_a,Hill_b)", np.minimum(hill_a, hill_b)),
                   ("Hill-Bliss Hill_a*Hill_b", hill_a * hill_b),
                   ("neutral BRAID (kappa=0, E3=min(E1,E2), no learning)", neutral),
                   ("comboKR2 adapted (saved predictions)", combokr)]:
    d1_rows.append(dict(model=name, **metrics(y, pred), max_prediction=float(np.max(pred)),
                        frac_pred_gt1=float(np.mean(pred > 1.0))))
d1 = pd.DataFrame(d1_rows)
d1.to_csv(OUT / "d1_representation_vs_learner.csv", index=False)
log("\n[D1] representation vs learner\n" + d1.to_string(index=False))
assert abs(d1.surface_macro_rmse[0] - 0.069950) < 5e-7, "HSA mismatch vs competitive_position_v1"
log(f"[D1] recomputed comboKR surface-macro minus metrics.json value = {d1.surface_macro_rmse[6] - 0.0936125360442528:.3e}")
assert abs(d1.surface_macro_rmse[6] - 0.0936125360442528) < 1e-9, "comboKR mismatch vs metrics.json"

mono_obs = np.concatenate([va.mono_response_a.to_numpy(float), va.mono_response_b.to_numpy(float)])
mono_med = np.concatenate([med_a, med_b])
mono_hill = np.concatenate([hill_a, hill_b])
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
mono_fit = dict(arm_points=int(mono_obs.size),
                rmse_observed_vs_median=rms(mono_obs - mono_med),
                rmse_hill_vs_median_fit_target=rms(mono_hill - mono_med),
                rmse_hill_vs_observed=rms(mono_hill - mono_obs),
                rmse_hill_vs_median_where_median_gt1=rms((mono_hill - mono_med)[mono_med > 1]),
                rmse_hill_vs_median_where_median_le1=rms((mono_hill - mono_med)[mono_med <= 1]),
                frac_observed_mono_gt1=float(np.mean(mono_obs > 1)), frac_median_mono_gt1=float(np.mean(mono_med > 1)),
                max_hill=float(mono_hill.max()))
log("[D1] mono representation at validation doses:", json.dumps(mono_fit))

# Hill pathologies -------------------------------------------------------------------
diag = pd.read_csv(COMBOKR_DIR / "fit_diagnostics.csv")
hd = diag[diag.kind == "hill"].merge(hills, on=["cell", "drug"], how="left", validate="one_to_one")
assert len(hd) == len(hills) == 2935
dose_info = cov.groupby(["cell", "drug"]).agg(n_doses=("dose", "size"), min_dose=("dose", "min"),
                                             max_dose=("dose", "max"), max_mono=("mono", "max")).reset_index()
hd = hd.merge(dose_info, on=["cell", "drug"], how="left", validate="one_to_one")
assert (hd.n_doses == hd.unique_doses).all(), "reconstructed Hill inputs differ from fit_diagnostics"
eps = 1e-6
flags = pd.DataFrame({
    "three_point_fit": hd.unique_doses == 3,
    "slope_at_lower_bound_0.2": hd.h <= 0.2 + eps,
    "slope_at_upper_bound_3": hd.h >= 3 - eps,
    "Emax_lt_1pct (near total kill)": hd.Emax < 1.0,
    "Emax_ge_99.99 (flat curve, upper bound)": hd.Emax >= 99.99,
    "tiny_C_lt_min_dose/10": hd.C < hd.min_dose / 10.0,
    "C_at_lower_bound_1e-16": np.log10(hd.C) <= -16 + 1e-6,
    "huge_C_gt_max_dose*10": hd.C > hd.max_dose * 10.0,
    "non_converged (success=False, status 0)": ~hd.success.astype(bool),
    "observed_mono_gt_100_unrepresentable": hd.max_mono > 100.0,
})
identifiability = ["slope_at_lower_bound_0.2", "slope_at_upper_bound_3", "Emax_ge_99.99 (flat curve, upper bound)",
                   "tiny_C_lt_min_dose/10", "C_at_lower_bound_1e-16", "huge_C_gt_max_dose*10",
                   "non_converged (success=False, status 0)"]
flags["any_bound_or_unidentified (union of the 7 rows above minus 3pt/Emax<1)"] = flags[identifiability].any(axis=1)
patho = pd.DataFrame({"flag": flags.columns, "count": flags.sum().to_numpy(), "of": len(hd),
                      "fraction": flags.mean().to_numpy()})
# how many validation points touch a flagged Hill fit in at least one arm
arm_keys_a = list(zip(va.cell_id, va.drug_a_id)); arm_keys_b = list(zip(va.cell_id, va.drug_b_id))
flag_maps = {f: dict(zip(zip(hd.cell, hd.drug), flags[f].to_numpy())) for f in flags.columns}
point_flags = {f: np.array([m[ka] or m[kb] for ka, kb in zip(arm_keys_a, arm_keys_b)]) for f, m in flag_maps.items()}
patho["validation_surfaces_with_flag_in_either_arm"] = [int(len(np.unique(surf[point_flags[f]]))) for f in flags.columns]
patho.to_csv(OUT / "d1_hill_pathologies.csv", index=False)
log("[D1] Hill pathologies\n" + patho.to_string(index=False))
log("[D1] Hill unique_doses distribution:", hd.unique_doses.value_counts().sort_index().to_dict(),
    "status:", hd.status.value_counts().sort_index().to_dict())

# Pathology vs validation error (flags are constant within a surface: same cell, drug pair) --------
patho_err = []
for f in flags.columns:
    for state, mask in (("flagged", point_flags[f]), ("not flagged", ~point_flags[f])):
        if mask.sum() == 0:
            continue
        row = dict(flag=f, subset=state, n_surfaces=int(len(np.unique(surf[mask]))))
        for name, pred in (("observed HSA", hsa), ("Hill-HSA", np.minimum(hill_a, hill_b)),
                           ("neutral BRAID", neutral), ("comboKR2", combokr)):
            row[name] = metrics(y[mask], pred[mask], surf[mask])["surface_macro_rmse"]
        patho_err.append(row)
patho_err = pd.DataFrame(patho_err)
patho_err.to_csv(OUT / "d1_error_by_hill_pathology.csv", index=False)
log("[D1] surface-macro RMSE by Hill pathology (either arm)\n" + patho_err.round(6).to_string(index=False))

# Training BRAID fits (isolated subprocess, allow_pickle=False) -------------------------
npz_code = r"""
import json, sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
b = z["braids"].astype(float); n = b.shape[0] // 2; b = b[:n]
kappa, e3, e1, e2 = b[:, 8], b[:, 3], b[:, 1], b[:, 2]
print(json.dumps(dict(training_surfaces=int(n), kappa_mean=float(kappa.mean()), kappa_median=float(np.median(kappa)),
    kappa_at_lower_bound=int(np.sum(kappa <= -1.9 + 1e-6)), kappa_at_upper_bound=int(np.sum(kappa >= 20 - 1e-6)),
    kappa_abs_lt_0p05=int(np.sum(np.abs(kappa) < 0.05)), e3_eq_min_e1e2=int(np.sum(np.abs(e3 - np.minimum(e1, e2)) < 1e-6)),
    e3_at_zero=int(np.sum(e3 <= 1e-6)), keys=sorted(z.files),
    cdd=[[str(v) for v in row] for row in z["cdd"][:n]], params=b.tolist())))
"""
npz_proc = subprocess.run([sys.executable, "-I", "-c", npz_code, str(COMBOKR_DIR / "fitted_training.npz")],
                          capture_output=True, text=True, check=True)
braid_train = json.loads(npz_proc.stdout)
train_cdd, train_params = braid_train.pop("cdd"), braid_train.pop("params")
braid_rows = diag[diag.kind == "braid_train_only"]
braid_train.update(train_braid_fit_success=int(braid_rows.success.astype(bool).sum()),
                   train_braid_fit_rmse_mean=float(braid_rows.rmse.mean()),
                   train_braid_fit_rmse_median=float(braid_rows.rmse.median()))
log("[D1] training BRAID fits:", json.dumps(braid_train))

# In-sample representation ceiling on the 1,972 training (cell, drug_a, drug_b) groups: the per-group
# fitted BRAID (E3 and kappa fitted to that group's own training responses, i.e. the targets comboKR
# learns from) versus zero-parameter references on the same points. Mean of per-group point RMSE.
ceiling = []
groups = tr.groupby(["cell_id", "drug_a_id", "drug_b_id"], sort=True).indices
assert [tuple(k) for k in train_cdd] == list(groups.keys())
for (key, idx), params in zip(groups.items(), train_params):
    cell, a, b = key
    da, db = molar(tr.dose_a_model.to_numpy()[idx]), molar(tr.dose_b_model.to_numpy()[idx])
    yt = tr.response.to_numpy(float)[idx]
    e1, h1, c1 = hill_lookup[(cell, a)]; e2, h2, c2 = hill_lookup[(cell, b)]
    fitted = braid_local(da, db, *params) / 100.0
    neut = braid_local(da, db, 100.0, e1, e2, min(e1, e2), h1, h2, c1, c2, 0.0) / 100.0
    ma, mb = tr.mono_response_a.to_numpy(float)[idx], tr.mono_response_b.to_numpy(float)[idx]
    ceiling.append(dict(points=len(idx), fitted_braid=rms(fitted - yt), neutral_braid=rms(neut - yt),
                        observed_hsa=rms(np.minimum(ma, mb) - yt),
                        hill_hsa=rms(np.minimum(hill_curve(da, e1, h1, c1), hill_curve(db, e2, h2, c2)) - yt)))
ceiling = pd.DataFrame(ceiling)
fit_check = float(np.max(np.abs(ceiling.fitted_braid.to_numpy() - braid_rows.rmse.to_numpy())))
log(f"[D1] recomputed fitted-BRAID group RMSE vs fit_diagnostics max|diff|={fit_check:.3g}")
assert fit_check < 1e-8
ceiling_summary = {k: float(ceiling[k].mean()) for k in ("fitted_braid", "neutral_braid", "hill_hsa", "observed_hsa")}
ceiling_summary.update(groups=int(len(ceiling)), points=int(ceiling.points.sum()),
                       frac_groups_fitted_braid_beats_observed_hsa=float((ceiling.fitted_braid < ceiling.observed_hsa).mean()))
pd.DataFrame([ceiling_summary]).to_csv(OUT / "d1_training_in_sample_ceiling.csv", index=False)
log("[D1] training in-sample ceiling (mean per-group point RMSE):", json.dumps(ceiling_summary))

# ---------------------------------------------------------------------------- D3
dp, dt = combokr - neutral, y - neutral
ps_ck, ps_nb, ps_hsa = per_surface_rmse(y, combokr), per_surface_rmse(y, neutral), per_surface_rmse(y, hsa)
sm = pd.DataFrame({"s": surf, "dp": dp, "dt": dt}).groupby("s", sort=True).mean()
wins = int((ps_ck < ps_nb).sum()); n_s = len(ps_ck)
wilson = lambda k, n, z=1.959964: ((k / n + z * z / (2 * n) - z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n))) / (1 + z * z / n),
                                  (k / n + z * z / (2 * n) + z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n))) / (1 + z * z / n))
slope = float(np.polyfit(dp, dt, 1)[0])
d3 = dict(
    point_pearson_delta=pearson(dp, dt), point_spearman_delta=spearman(dp, dt),
    surface_mean_pearson_delta=pearson(sm.dp, sm.dt),
    ols_slope_truthdelta_on_preddelta=slope,
    mean_abs_pred_delta=float(np.mean(np.abs(dp))), mean_abs_truth_delta=float(np.mean(np.abs(dt))),
    mean_pred_delta=float(np.mean(dp)), mean_truth_delta=float(np.mean(dt)),
    sign_agreement_rate=float(np.mean(np.sign(dp) == np.sign(dt))),
    surfaces=n_s, combokr_beats_neutral_surfaces=wins, combokr_beats_neutral_fraction=wins / n_s,
    combokr_beats_neutral_wilson95=list(wilson(wins, n_s)),
    combokr_beats_hsa_fraction=float((ps_ck < ps_hsa).mean()),
    neutral_beats_hsa_fraction=float((ps_nb < ps_hsa).mean()),
    rmse_combokr=float(ps_ck.mean()), rmse_neutral=float(ps_nb.mean()))
(OUT / "d3_combokr_learned_part.json").write_text(json.dumps(d3, indent=2))
log("[D3]", json.dumps(d3, indent=1))

# ---------------------------------------------------------------------------- D2
all_models = {"observed HSA (zero-parameter)": [hsa], "neutral BRAID (no learning)": [neutral], **predictions}
mono_max = np.maximum(va.mono_response_a.to_numpy(float), va.mono_response_b.to_numpy(float))
sinfo = pd.DataFrame({"s": surf, "y_gt1": y > 1, "mono_gt1": mono_max > 1}).groupby("s", sort=True).agg(
    any_combo_gt1=("y_gt1", "any"), n_combo_gt1=("y_gt1", "sum"), any_mono_gt1=("mono_gt1", "any"))
pt_any_combo = sinfo.any_combo_gt1.reindex(surf).to_numpy()
pt_any_mono = sinfo.any_mono_gt1.reindex(surf).to_numpy()
pt_ncombo = sinfo.n_combo_gt1.reindex(surf).to_numpy()
bins = {"0/9 points >1": pt_ncombo == 0, "1-2/9": (pt_ncombo >= 1) & (pt_ncombo <= 2),
        "3-5/9": (pt_ncombo >= 3) & (pt_ncombo <= 5), "6-9/9": pt_ncombo >= 6}
strata = {"all surfaces": np.ones(len(y), bool),
          "no combo >1 & no mono >1": ~pt_any_combo & ~pt_any_mono,
          "no combo >1 & mono >1": ~pt_any_combo & pt_any_mono,
          "combo >1 & no mono >1": pt_any_combo & ~pt_any_mono,
          "combo >1 & mono >1": pt_any_combo & pt_any_mono,
          "any combo point >1": pt_any_combo, "no combo point >1": ~pt_any_combo,
          "any mono anchor >1 (input-defined)": pt_any_mono, "no mono anchor >1 (input-defined)": ~pt_any_mono,
          **{f"fraction bin {k}": v for k, v in bins.items()}}
d2_rows = []
for sname, mask in strata.items():
    for mname, preds in all_models.items():
        vals = [metrics(y[mask], p[mask], surf[mask]) for p in preds]
        macro = np.array([v["surface_macro_rmse"] for v in vals]); bias = np.array([v["bias"] for v in vals])
        d2_rows.append(dict(stratum=sname, model=mname, seeds=len(preds), n_surfaces=vals[0]["n_surfaces"],
                            surface_macro_rmse_mean=float(macro.mean()),
                            surface_macro_rmse_sd=float(macro.std(ddof=1)) if len(macro) > 1 else float("nan"),
                            bias_mean=float(bias.mean())))
d2 = pd.DataFrame(d2_rows)
d2.to_csv(OUT / "d2_strata_by_gt1.csv", index=False)
wide = d2.pivot(index="stratum", columns="model", values="surface_macro_rmse_mean").reindex(list(strata))
wide.insert(0, "n_surfaces", d2.drop_duplicates("stratum").set_index("stratum").n_surfaces.reindex(list(strata)))
wide.to_csv(OUT / "d2_strata_by_gt1_wide.csv")
log("[D2] strata (surface-macro RMSE, seed mean)\n" + wide.round(6).to_string())
rel = wide.drop(columns="n_surfaces").div(wide["observed HSA (zero-parameter)"], axis=0) - 1.0
rel.insert(0, "n_surfaces", wide.n_surfaces)
rel.to_csv(OUT / "d2_strata_rel_vs_hsa.csv")
log("[D2] strata, relative to observed HSA in the same stratum\n" + rel.round(4).to_string())

# Group bootstrap of rel = macro(model)/macro(HSA) - 1 within each stratum, using the project's
# convention (summarize_algorithm_search.group_bootstrap): resample prediction groups within held-out
# clusters; model per-surface RMSE is the seed mean. 2000 draws, fixed seed.
meta = va.drop_duplicates("surface_replicate_id").set_index("surface_replicate_id").sort_index()[
    ["prediction_group_id", "heldout_cluster_id"]]
seed_mean_ps = {m: pd.concat([per_surface_rmse(y, p) for p in preds], axis=1).mean(axis=1) for m, preds in all_models.items()}


def boot_rel(cand: pd.Series, ref: pd.Series, surfaces, draws=2000, seed=20261011):
    frame = pd.DataFrame({"k": meta.loc[surfaces, "heldout_cluster_id"], "g": meta.loc[surfaces, "prediction_group_id"],
                          "c": cand.loc[surfaces], "r": ref.loc[surfaces]})
    agg = frame.groupby(["k", "g"]).agg(c=("c", "sum"), r=("r", "sum")).reset_index()
    rng = np.random.default_rng(seed); c_tot = np.zeros(draws); r_tot = np.zeros(draws)
    for _, part in agg.groupby("k"):
        pick = rng.integers(0, len(part), size=(draws, len(part)))
        c_tot += part.c.to_numpy()[pick].sum(axis=1); r_tot += part.r.to_numpy()[pick].sum(axis=1)
    point = float(frame.c.sum() / frame.r.sum() - 1)
    low, high = np.percentile(c_tot / r_tot - 1, [2.5, 97.5])
    return point, float(low), float(high)


boot_rows = []
ref_name = "observed HSA (zero-parameter)"
for sname, mask in strata.items():
    surfaces = np.unique(surf[mask])
    for mname in all_models:
        if mname == ref_name:
            continue
        point, low, high = boot_rel(seed_mean_ps[mname], seed_mean_ps[ref_name], surfaces)
        boot_rows.append(dict(stratum=sname, n_surfaces=len(surfaces), model=mname, rel_vs_hsa=point, ci95_low=low, ci95_high=high))
boot = pd.DataFrame(boot_rows)
boot.to_csv(OUT / "d2_strata_rel_vs_hsa_bootstrap.csv", index=False)
log("[D2] rel vs HSA with group-bootstrap 95% CI\n" + boot.round(4).to_string(index=False))
d3_boot = boot_rel(seed_mean_ps["comboKR2 adapted"], seed_mean_ps["neutral BRAID (no learning)"], meta.index.to_numpy())
log("[D3] comboKR vs neutral BRAID rel diff (point, 95% CI):", d3_boot)

point_rows = []
for label, mask in [("points y>1", y > 1), ("points y<=1", y <= 1)]:
    for mname, preds in all_models.items():
        r = np.array([np.sqrt(np.mean((p[mask] - y[mask]) ** 2)) for p in preds])
        b = np.array([np.mean(p[mask] - y[mask]) for p in preds])
        point_rows.append(dict(subset=label, n_points=int(mask.sum()), model=mname, point_rmse_mean=float(r.mean()), bias_mean=float(b.mean())))
d2p = pd.DataFrame(point_rows)
d2p.to_csv(OUT / "d2_point_level_gt1.csv", index=False)
log("[D2] point-level\n" + d2p.to_string(index=False))

# Clipped-target diagnostic re-score ---------------------------------------------------
clip = pd.read_parquet(CLIPPED / "validation.parquet",
                       columns=["source_combodrugseq", "surface_replicate_id", "response", "mono_response_a", "mono_response_b"]
                       ).set_index("source_combodrugseq").loc[va.source_combodrugseq].reset_index()
y_clip = np.minimum(y, 1.0)
clip_check = dict(
    same_surface_ids=bool((clip.surface_replicate_id.to_numpy() == surf).all()),
    max_abs_clipped_shard_minus_min_y_1=float(np.max(np.abs(clip.response.to_numpy() - y_clip))),
    max_abs_clipped_mono_a_minus_min=float(np.max(np.abs(clip.mono_response_a.to_numpy() - np.minimum(va.mono_response_a, 1.0)))),
    max_abs_clipped_mono_b_minus_min=float(np.max(np.abs(clip.mono_response_b.to_numpy() - np.minimum(va.mono_response_b, 1.0)))),
    frac_target_exactly_1_after_clip=float(np.mean(y_clip == 1.0)),
    frac_target_and_clippedHSA_both_1=float(np.mean((y_clip == 1.0) & (np.minimum(hsa, 1.0) == 1.0))))
log("[D2] clipped shard check:", json.dumps(clip_check))
clip_rows = []
for mname, preds in all_models.items():
    raw = np.array([metrics(y_clip, p)["surface_macro_rmse"] for p in preds])
    cl = np.array([metrics(y_clip, np.minimum(p, 1.0))["surface_macro_rmse"] for p in preds])
    unc = np.array([metrics(y, p)["surface_macro_rmse"] for p in preds])
    clip_rows.append(dict(model=mname, seeds=len(preds), unclipped_target=float(unc.mean()),
                          clipped_target_raw_prediction=float(raw.mean()),
                          clipped_target_clipped_prediction=float(cl.mean()),
                          clipped_target_clipped_prediction_sd=float(cl.std(ddof=1)) if len(cl) > 1 else float("nan")))
d2c = pd.DataFrame(clip_rows)
hsa_unc = float(d2c.unclipped_target[0]); hsa_cl = float(d2c.clipped_target_clipped_prediction[0])
d2c["rel_vs_HSA_unclipped"] = d2c.unclipped_target / hsa_unc - 1
d2c["rel_vs_HSA_clipped"] = d2c.clipped_target_clipped_prediction / hsa_cl - 1
d2c.to_csv(OUT / "d2_clipped_target_rescore.csv", index=False)
log("[D2] clipped-target re-score (DIAGNOSTIC ONLY; models trained on unclipped targets)\n" + d2c.round(6).to_string(index=False))

summary = dict(d1=d1_rows, neutral_braid_author_max_abs_diff=author_max_diff, mono_fit=mono_fit,
               hill_pathologies=patho.to_dict("records"), braid_train=braid_train, training_ceiling=ceiling_summary,
               d3=d3, d3_combokr_vs_neutral_rel_ci=d3_boot, clip_check=clip_check, stratum_sizes={k: int(len(np.unique(surf[v]))) for k, v in strata.items()},
               runtime_seconds=round(time.perf_counter() - T0, 2), python=sys.version.split()[0],
               numpy=np.__version__, pandas=pd.__version__)
(OUT / "d1_d3_summary.json").write_text(json.dumps(summary, indent=2, default=float))
log(f"done in {summary['runtime_seconds']} s")

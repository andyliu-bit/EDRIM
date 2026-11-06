"""JointFeature-v1 audits (validation/training roles only, no test access).

1. Data/grouping consistency: what "1972" vs "2000" counts, ID files and hashes.
2. DoseBasis edge behaviour: runtime values inside/outside the knots, and the
   fraction of training/validation arm doses that fall on the flat edges.
3. Error complementarity C3-g2 vs full XGBoost: signed point residuals,
   per-surface RMSE differences, win/loss breakdown. Descriptive only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch

from bisurfgnn.models.numeric_surface import DoseBasis

RUNS = ROOT / "runs/model_improvement"
NS = RUNS / "numeric_surface_v1"
SHARD_DIR = ROOT.parent / "bisurfgnn/data/processed/one_new_scaffold_benchmark_unclipped/learned_point_shards/fold_0"
spec = importlib.util.spec_from_file_location("nss", Path(__file__).with_name("summarize_numeric_surface.py"))
NSS = importlib.util.module_from_spec(spec); spec.loader.exec_module(NSS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role_frame(role):
    ids = pd.read_csv(RUNS / f"comparator_gpu_v2/{role}_ids.csv")
    cols = ["source_combodrugseq", "prediction_group_id", "surface_replicate_id", "dose_a_model", "dose_b_model",
            "response", "mono_response_a", "mono_response_b", "drug_a_id", "drug_b_id", "cell_id"]
    f = pd.read_parquet(SHARD_DIR / f"{role}.parquet", columns=cols).set_index("source_combodrugseq").loc[ids.source_combodrugseq].reset_index()
    assert (f.surface_replicate_id.to_numpy() == ids.surface_replicate_id.to_numpy()).all()
    return ids, f


def audit_grouping():
    out = {}
    for role in ("train", "validation"):
        ids, f = role_frame(role)
        out[role] = dict(points=len(ids), prediction_groups=int(ids.prediction_group_id.nunique()),
                         surface_replicates=int(ids.surface_replicate_id.nunique()),
                         groups_with_multiple_replicates=int((ids.groupby("prediction_group_id").surface_replicate_id.nunique() > 1).sum()),
                         id_file_sha256=sha(RUNS / f"comparator_gpu_v2/{role}_ids.csv"))
    # every later run must hold byte-identical ID files
    runs = sorted(set(p.parent for p in RUNS.rglob("train_ids.csv")))
    out["runs_with_id_files"] = len(runs)
    out["runs_with_identical_ids"] = sum(sha(r / "train_ids.csv") == out["train"]["id_file_sha256"] and
                                         sha(r / "validation_ids.csv") == out["validation"]["id_file_sha256"] for r in runs)
    out["runs_with_different_ids"] = [str(r.relative_to(ROOT)) for r in runs
                                      if sha(r / "train_ids.csv") != out["train"]["id_file_sha256"]]
    return out


def audit_dose_basis():
    knots = json.loads(next((NS / "snap_ae92da88a034/runs/model_improvement/stab_full_g2").glob(
        "*_seed_41_primary/metrics.json")).read_text())["numeric_stats"]["dose_knots"]
    basis = DoseBasis(tuple(knots))
    probe = torch.tensor([knots[0] - 2, knots[0] - 0.5, knots[0], (knots[0] + knots[1]) / 2, knots[1],
                          (knots[1] + knots[2]) / 2, knots[2], knots[2] + 0.5, knots[2] + 2])
    phi = basis(probe)
    out = dict(knots=knots, probe=probe.tolist(), phi=phi.tolist(),
               behaviour_below_first_knot="constant [1,0,0] (clamped, no linear extrapolation)",
               behaviour_above_last_knot="constant [0,0,1] (clamped)")
    for role in ("train", "validation"):
        _, f = role_frame(role)
        d = np.concatenate([f.dose_a_model, f.dose_b_model])
        lo, hi = d < knots[0], d > knots[2]
        pt_flat = (f.dose_a_model < knots[0]) | (f.dose_a_model > knots[2]) | (f.dose_b_model < knots[0]) | (f.dose_b_model > knots[2])
        out[role] = dict(arm_fraction_below_first_knot=float(lo.mean()), arm_fraction_above_last_knot=float(hi.mean()),
                         point_fraction_any_arm_on_flat_edge=float(pt_flat.mean()))
    return out


def seed_avg_pred(sources, prefix):
    preds = {}
    for run, pre in sources:
        for job in sorted(run.glob(f"{pre}_seed_*_primary")):
            if job.name.split("_seed_")[0] != pre:
                continue
            seed = job.name.split("_seed_")[1].split("_")[0]
            preds.setdefault(seed, pd.read_csv(job / "predictions.csv").prediction.to_numpy())
    return np.mean([preds[s] for s in sorted(preds)], axis=0), sorted(preds)


def audit_complementarity():
    _, va = role_frame("validation")
    y = va.response.to_numpy()
    c3, s1 = seed_avg_pred(NSS.SOURCES["C3_numsurf_tabm_o1_g2"], None)
    xg, s2 = seed_avg_pred(NSS.SOURCES["xgb_full_T"], None)
    ea, eb = c3 - y, xg - y
    surf = va.surface_replicate_id.to_numpy()
    rs = pd.DataFrame({"s": surf, "a": ea ** 2, "b": eb ** 2}).groupby("s").mean() ** 0.5
    diff = rs.a - rs.b
    out = dict(seeds_c3=s1, seeds_xgb=s2,
               point_signed_residual_pearson=float(np.corrcoef(ea, eb)[0, 1]),
               point_signed_residual_spearman=float(pd.Series(ea).rank().corr(pd.Series(eb).rank())),
               point_fraction_same_sign=float((np.sign(ea) == np.sign(eb)).mean()),
               residual_difference_sd=float(np.std(ea - eb)),
               surface_rmse_pearson=float(np.corrcoef(rs.a, rs.b)[0, 1]),
               surface_diff_quantiles_c3_minus_xgb={q: float(diff.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
               surface_fraction_c3_better=float((diff < 0).mean()),
               surface_fraction_c3_better_by_0p005=float((diff < -0.005).mean()),
               surface_fraction_xgb_better_by_0p005=float((diff > 0.005).mean()),
               note="descriptive only; no blend weight fitted, no candidate added")
    return out


if __name__ == "__main__":
    out_dir = NS.parent / "joint_feature_v1"; out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(grouping=audit_grouping(), dose_basis=audit_dose_basis(), complementarity=audit_complementarity(),
                  test_response_rows_read=0)
    (out_dir / "audits.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))

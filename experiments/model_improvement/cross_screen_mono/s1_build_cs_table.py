"""CrossScreenMono-v1 steps 1-2 (docs/CROSS_SCREEN_MONO_V1_PROTOCOL.md section 2).

Step 1: cross-screen (CS) monotherapy value for every (drug, cell, dose) used by the frozen train/validation
        points (NCI-60 DOSERESP via nci60_points.parquet; log10-dose linear interpolation inside one
        (NSC, cell) curve, constant extrapolation outside, flagged). Coverage report; stop if fewer than 80%
        of validation points are covered.
Step 2: descriptive agreement between CS and same-plate monotherapy on covered points. The same-plate values
        are read ONLY here, are never written into the CS table, and never reach any model.

Usage (repository root):
  .venv/Scripts/python.exe experiments/model_improvement/cross_screen_mono/s1_build_cs_table.py
Outputs: runs/model_improvement/cross_screen_mono_v1/cs_table/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CS_DIR, HERE, MONO_MAX, NCI60_DIR, NCI60_INPUT_SHA256, ROLES, ROOT, RUN_ROOT,  # noqa: E402
                    append_log, cell_mapping, frozen_frames, load_config, sha256_file, write_ids)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from bisurfgnn.cross_screen.mono import (NODE_TOLERANCE_LOG10, CurveBank, candidate_compounds,  # noqa: E402
                                         choose_curves, cross_screen_values)
from bisurfgnn.pretrain.nci60 import forbid_test_responses  # noqa: E402


def agreement(cs, sp) -> dict:
    cs, sp = np.asarray(cs, float), np.asarray(sp, float)
    if len(cs) < 3:
        return dict(n=int(len(cs)))
    return dict(n=int(len(cs)), pearson=float(pearsonr(cs, sp)[0]), spearman=float(spearmanr(cs, sp)[0]),
                rmse=float(np.sqrt(np.mean((cs - sp) ** 2))), mae=float(np.mean(np.abs(cs - sp))),
                mean_diff_cs_minus_sp=float(np.mean(cs - sp)), cs_mean=float(cs.mean()), sp_mean=float(sp.mean()))


def main():
    t0 = time.monotonic()
    if (CS_DIR / "manifest.json").exists():
        raise FileExistsError("Step 1 already completed; outputs are frozen")
    CS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(HERE / "configs/base.yaml")
    data_root = (ROOT / cfg["data_root"]).resolve()
    for name, expected in NCI60_INPUT_SHA256.items():
        if sha256_file(NCI60_DIR / name) != expected:
            raise RuntimeError(f"NCI60-Pretrain-v1 output changed: {name}")
    with forbid_test_responses() as guard:
        frames, banks, _, _, binding = frozen_frames(cfg, CS_DIR)
        cmap = cell_mapping(cfg)
        compounds = pd.read_parquet(NCI60_DIR / "nci60_compounds.parquet", columns=["nsc", "identity_group"])
        drugs = pd.read_csv(data_root / "data/processed/nci_almanac_unclipped/drugs.csv", keep_default_na=False)
        needed = sorted({d for f in frames.values() for c in ("drug_a_id", "drug_b_id") for d in f[c]})
        cand = candidate_compounds(drugs, compounds, needed)
        points = pd.read_parquet(NCI60_DIR / "nci60_points.parquet", columns=["nsc", "cell", "log10_conc", "y"],
                                 filters=[("nsc", "in", sorted(int(n) for n in cand.nsc))])
        assignments = pd.read_parquet(data_root / "data/processed/one_new_scaffold_benchmark_unclipped/surface_assignments.parquet",
                                      columns=["surface_replicate_id", "heldout_drug_id"])
    curves = CurveBank(points)
    # ------------------------------------------------------------------ step 1: CS values per point and arm
    requests = []
    for role in ROLES:
        f = frames[role]
        bio = f.cell_id.map(cmap)
        if bio.isna().any():
            raise RuntimeError(f"Unmapped ALMANAC cells in {role}: {sorted(set(f.cell_id[bio.isna()]))}")
        for arm in ("a", "b"):
            requests.append(pd.DataFrame({"role": role, "arm": arm, "row": np.arange(len(f)),
                                          "source_combodrugseq": f.source_combodrugseq.to_numpy(),
                                          "surface_replicate_id": f.surface_replicate_id.to_numpy(),
                                          "drug_id": f[f"drug_{arm}_id"].to_numpy(), "cell_id": f.cell_id.to_numpy(),
                                          "cell_bio": bio.to_numpy(), "dose_model": f[f"dose_{arm}_model"].to_numpy()}))
    req = pd.concat(requests, ignore_index=True)
    choice = choose_curves(cand, curves, req[["drug_id", "cell_bio"]])
    values = cross_screen_values(req.drug_id, req.cell_bio, req.dose_model, choice, curves)
    req = pd.concat([req, values.drop(columns=["drug_id", "cell_bio", "dose_model"])], axis=1)
    flag_cols = ["cs_mono", "covered", "nsc", "method", "n_curve_doses", "log10_molar", "extrapolated",
                 "extrapolated_low", "extrapolated_high", "extrapolation_distance_log10", "at_measured_dose"]
    stats = dict(protocol=cfg["protocol"], node_tolerance_log10=NODE_TOLERANCE_LOG10, frozen_ids=binding["id_sha256"])
    for role in ROLES:
        f = frames[role]
        wide = pd.DataFrame({"source_combodrugseq": f.source_combodrugseq.to_numpy(),
                             "prediction_group_id": f.prediction_group_id.to_numpy(),
                             "surface_replicate_id": f.surface_replicate_id.to_numpy()})
        for arm in ("a", "b"):
            part = req[(req.role == role) & (req.arm == arm)].sort_values("row")
            for c in flag_cols:
                wide[f"{c}_{arm}" if c != "covered" else f"covered_{arm}"] = part[c].to_numpy()
        wide["covered"] = wide.covered_a.astype(bool) & wide.covered_b.astype(bool)
        wide.to_csv(CS_DIR / f"cs_points_{role}.csv", index=False)
        write_ids(f.loc[wide.covered.to_numpy()], CS_DIR / f"covered_{role}_ids.csv")
        cov = wide.covered.to_numpy()
        surf = wide.groupby("surface_replicate_id").covered.agg(["all", "any"])
        grp = wide.groupby("prediction_group_id").covered.agg(["all", "any"])
        if (surf["all"] != surf["any"]).any():
            raise RuntimeError("Partially covered surface: coverage must be per (drug, cell)")
        role_drugs = sorted(set(f.drug_a_id) | set(f.drug_b_id))
        cov_rows = f.loc[cov]
        cov_drugs = sorted(set(cov_rows.drug_a_id) | set(cov_rows.drug_b_id))
        arms = req[(req.role == role)]
        arms_cov = arms[arms.covered.astype(bool)]
        pts_cov = wide[cov]
        any_extrap = pts_cov.extrapolated_a.astype(bool) | pts_cov.extrapolated_b.astype(bool)
        raw_cs = np.concatenate([pts_cov.cs_mono_a.to_numpy(float), pts_cov.cs_mono_b.to_numpy(float)])
        stats[role] = dict(
            points=len(f), points_covered=int(cov.sum()), point_coverage=float(cov.mean()),
            surfaces=int(len(surf)), surfaces_covered=int(surf["all"].sum()),
            prediction_groups=int(len(grp)), prediction_groups_covered=int(grp["all"].sum()),
            drugs=len(role_drugs), drugs_with_any_covered_point=len(cov_drugs),
            drugs_without_any_covered_point=sorted(set(role_drugs) - set(cov_drugs)),
            cells=int(f.cell_id.nunique()), cells_covered=int(cov_rows.cell_id.nunique()),
            heldout_clusters=int(f.heldout_cluster_id.nunique()), heldout_clusters_covered=int(cov_rows.heldout_cluster_id.nunique()),
            drug_cell_pairs=int(arms[["drug_id", "cell_bio"]].drop_duplicates().shape[0]),
            drug_cell_pairs_covered=int(arms_cov[["drug_id", "cell_bio"]].drop_duplicates().shape[0]),
            arm_values_on_covered_points=int(2 * cov.sum()),
            arm_extrapolated_fraction=float(np.mean(np.concatenate([pts_cov.extrapolated_a, pts_cov.extrapolated_b]).astype(bool))),
            arm_extrapolated_low_fraction=float(np.mean(np.concatenate([pts_cov.extrapolated_low_a, pts_cov.extrapolated_low_b]).astype(bool))),
            arm_extrapolated_high_fraction=float(np.mean(np.concatenate([pts_cov.extrapolated_high_a, pts_cov.extrapolated_high_b]).astype(bool))),
            arm_at_measured_dose_fraction=float(np.mean(np.concatenate([pts_cov.at_measured_dose_a, pts_cov.at_measured_dose_b]).astype(bool))),
            point_any_arm_extrapolated_fraction=float(any_extrap.mean()),
            extrapolation_distance_log10_quantiles={q: float(np.quantile(np.concatenate(
                [pts_cov.extrapolation_distance_log10_a[pts_cov.extrapolated_a.astype(bool)],
                 pts_cov.extrapolation_distance_log10_b[pts_cov.extrapolated_b.astype(bool)]]), q))
                for q in (0.25, 0.5, 0.75, 0.95, 1.0)} if any_extrap.any() else None,
            identity_group_fallback_arm_values=int((arms_cov.method == "identity_group").sum()),
            identity_group_fallback_pairs=int(arms_cov.loc[arms_cov.method == "identity_group", ["drug_id", "cell_bio"]].drop_duplicates().shape[0]),
            cs_values_outside_0_3_before_clipping=int(((raw_cs < 0) | (raw_cs > MONO_MAX)).sum()),
            cs_value_range=[float(raw_cs.min()), float(raw_cs.max())])
    for role in ROLES:  # held-out (new-scaffold) drug of each surface: label-free assignment metadata
        wide = pd.read_csv(CS_DIR / f"cs_points_{role}.csv")
        meta = wide[["surface_replicate_id", "covered"]].drop_duplicates("surface_replicate_id").merge(
            assignments, on="surface_replicate_id", how="left", validate="one_to_one")
        held = meta.groupby("heldout_drug_id").covered.agg(["size", "sum"])
        stats[role]["heldout_drugs"] = int(len(held))
        stats[role]["heldout_drugs_with_covered_surfaces"] = int((held["sum"] > 0).sum())
        stats[role]["heldout_drug_surface_coverage"] = {d: [int(r["sum"]), int(r["size"])] for d, r in held.iterrows()}
    choice.merge(cand.groupby("drug_id").size().rename("candidate_nscs").reset_index(), on="drug_id", how="left") \
        .to_csv(CS_DIR / "curve_choice.csv", index=False)
    lookup = req[req.covered.astype(bool)].assign(dose_key=lambda x: np.round(x.dose_model, 6)) \
        .drop_duplicates(["drug_id", "cell_bio", "dose_key"])[["drug_id", "cell_bio", "dose_key", "log10_molar", "nsc", "method",
                                                             "n_curve_doses", "cs_mono", "extrapolated_low", "extrapolated_high",
                                                             "extrapolation_distance_log10", "at_measured_dose"]] \
        .sort_values(["drug_id", "cell_bio", "dose_key"])
    lookup.to_csv(CS_DIR / "cs_lookup.csv", index=False)
    per_drug = []
    for role in ROLES:
        arms = req[req.role == role]
        for drug, g in arms.groupby("drug_id"):
            pairs = g[["cell_bio", "covered", "uncovered_reason"]].drop_duplicates(["cell_bio"])
            per_drug.append(dict(role=role, drug_id=drug, cells=len(pairs), cells_covered=int(pairs.covered.astype(bool).sum()),
                                 uncovered_reasons="|".join(sorted(set(pairs.uncovered_reason[~pairs.covered.astype(bool)]))),
                                 arm_values=len(g), arm_values_extrapolated=int(g.extrapolated.astype(bool).sum())))
    pd.DataFrame(per_drug).to_csv(CS_DIR / "drug_coverage.csv", index=False)
    stats["curve_choice"] = dict(pairs=len(choice), covered=int(choice.covered.sum()),
                                 by_method=choice[choice.covered].method.value_counts().to_dict(),
                                 tied_candidates_gt1=int((choice.tied_candidates > 1).sum()),
                                 curve_doses_quantiles={q: float(np.quantile(choice[choice.covered].n_curve_doses, q))
                                                        for q in (0, 0.25, 0.5, 0.75, 1.0)})
    stats["candidates"] = dict(drugs=len(needed), drugs_with_structured_own_nsc=int(cand[(cand.tier == 0) & cand.structured_in_nci60].drug_id.nunique()),
                               candidate_nscs=int(cand.nsc.nunique()), nci60_points_loaded=len(points))
    val_cov = stats["validation"]["point_coverage"]
    stats["stop_rule"] = dict(minimum_validation_coverage=cfg["minimum_validation_coverage"], validation_coverage=val_cov,
                              stop=bool(val_cov < cfg["minimum_validation_coverage"]))
    # ------------------------------------------------------------------ step 2: CS vs same-plate (descriptive only)
    sanity = {}
    for role in ROLES:
        f, wide = frames[role], pd.read_csv(CS_DIR / f"cs_points_{role}.csv")
        cov = wide.covered.to_numpy(bool)
        parts = []
        for arm in ("a", "b"):
            parts.append(pd.DataFrame({"drug": f[f"drug_{arm}_id"].to_numpy()[cov], "cell": f.cell_id.to_numpy()[cov],
                                       "dose": np.round(f[f"dose_{arm}_model"].to_numpy()[cov], 6),
                                       "sp": f[f"mono_response_{arm}"].to_numpy(float)[cov],
                                       "cs": wide[f"cs_mono_{arm}"].to_numpy(float)[cov],
                                       "extrapolated": wide[f"extrapolated_{arm}"].to_numpy(bool)[cov],
                                       "source_row": f.source_combodrugseq.to_numpy()[cov]}))
        arms = pd.concat(parts, ignore_index=True)
        obs = arms.sort_values("source_row").drop_duplicates(["drug", "cell", "dose", "sp"])
        dcd = arms.groupby(["drug", "cell", "dose"]).agg(sp=("sp", "mean"), cs=("cs", "first"),
                                                         extrapolated=("extrapolated", "first")).reset_index()
        hsa_sp = np.minimum(f.mono_response_a.to_numpy(float), f.mono_response_b.to_numpy(float))[cov]
        hsa_cs = np.minimum(wide.cs_mono_a.to_numpy(float), wide.cs_mono_b.to_numpy(float))[cov]
        sanity[role] = dict(
            arm_level_as_model_input=agreement(arms.cs, arms.sp),
            arm_level_interpolated=agreement(arms.cs[~arms.extrapolated], arms.sp[~arms.extrapolated]),
            arm_level_extrapolated=agreement(arms.cs[arms.extrapolated], arms.sp[arms.extrapolated]),
            unique_observations=agreement(obs.cs, obs.sp),
            drug_cell_dose_mean=agreement(dcd.cs, dcd.sp),
            drug_cell_dose_mean_interpolated=agreement(dcd.cs[~dcd.extrapolated], dcd.sp[~dcd.extrapolated]),
            drug_cell_dose_mean_extrapolated=agreement(dcd.cs[dcd.extrapolated], dcd.sp[dcd.extrapolated]),
            hsa_point_level=agreement(hsa_cs, hsa_sp))
        rows = []
        for drug, g in obs.groupby("drug"):
            rows.append(dict(role=role, drug_id=drug, **{k: v for k, v in agreement(g.cs, g.sp).items()},
                             extrapolated_fraction=float(g.extrapolated.mean())))
        pd.DataFrame(rows).to_csv(CS_DIR / f"sanity_by_drug_{role}.csv", index=False)
    sanity["reference_nci60_pretrain_v1_matched_training_drugs"] = dict(
        pearson=0.6352724687703529, rmse=0.19545910827111448, note="0.5-log dose bins, 49 training drugs, s2_stats.json")
    (CS_DIR / "sanity.json").write_text(json.dumps(sanity, indent=1), encoding="utf-8")
    (CS_DIR / "coverage.json").write_text(json.dumps(stats, indent=1, default=float), encoding="utf-8")
    outputs = {p.name: sha256_file(p) for p in sorted(CS_DIR.iterdir()) if p.is_file() and p.name != "manifest.json"}
    code = {str(p.relative_to(ROOT)): sha256_file(p) for p in
            [Path(__file__), HERE / "common.py", HERE / "configs/base.yaml", ROOT / "src/bisurfgnn/cross_screen/mono.py"]}
    manifest = dict(protocol=cfg["protocol"], outputs=outputs, code=code,
                    inputs={**{n: sha256_file(NCI60_DIR / n) for n in NCI60_INPUT_SHA256},
                            "drugs.csv": sha256_file(data_root / "data/processed/nci_almanac_unclipped/drugs.csv")},
                    frozen_id_sha256=binding["id_sha256"], parquet_reads=guard.accesses, test_response_rows_read=0,
                    same_plate_monotherapy_use="step-2 descriptive agreement only; not stored in any output table",
                    seconds=time.monotonic() - t0)
    (CS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    v, t = stats["validation"], stats["train"]
    msg = (f"STEP1 CS table: validation coverage {v['points_covered']}/{v['points']} points ({v['point_coverage']:.4f}), "
           f"{v['surfaces_covered']}/{v['surfaces']} surfaces, drugs {v['drugs_with_any_covered_point']}/{v['drugs']}; "
           f"train {t['points_covered']}/{t['points']} ({t['point_coverage']:.4f}), {t['surfaces_covered']}/{t['surfaces']} surfaces; "
           f"arm extrapolation val {v['arm_extrapolated_fraction']:.4f} / train {t['arm_extrapolated_fraction']:.4f}; "
           f"stop={stats['stop_rule']['stop']}; STEP2 val arm-level Pearson "
           f"{sanity['validation']['arm_level_as_model_input'].get('pearson', float('nan')):.3f} RMSE "
           f"{sanity['validation']['arm_level_as_model_input'].get('rmse', float('nan')):.4f}")
    append_log(RUN_ROOT / "STEP_LOG.md", msg)
    print(msg)
    if stats["stop_rule"]["stop"]:
        (CS_DIR / "STOP.json").write_text(json.dumps(stats["stop_rule"], indent=1), encoding="utf-8")
        raise SystemExit("Validation coverage below the pre-registered 80%: stopping (protocol section 2)")


if __name__ == "__main__":
    main()

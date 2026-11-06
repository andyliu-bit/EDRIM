"""NCI60-Pretrain-v1 summary: D_gnn_pt vs the existing SparseMono-v1 results on the identical frozen sample.

Reuses collect()/compare()/per_surface() of experiments/model_improvement/analysis/summarize_sparse_mono.py
(paired seeds, drug bootstrap over the 16 held-out target drugs). Writes only to
runs/model_improvement/nci60_pretrain_v1/summary/.
  python summarize.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("sm_summary", ROOT / "experiments/model_improvement/analysis/summarize_sparse_mono.py")
smry = importlib.util.module_from_spec(spec); spec.loader.exec_module(smry)
SM = ROOT / "runs/model_improvement/sparse_mono_v1"
EXISTING = [SM / "snap_921bdc09d9a4/runs/model_improvement/screen1", SM / "snap_1e79403e10f6/runs/model_improvement/screen2",
            SM / "snap_1d1be2d92a79/runs/model_improvement/stab"]
RUN = ROOT / "runs/model_improvement/nci60_pretrain_v1"
OUT = RUN / "summary"
SCREEN = (41, 42)


def seed_mean(rows, pipeline, k, seeds=None):
    r = rows[(rows.pipeline == pipeline) & (rows.k == k)]
    if seeds is not None:
        r = r[r.seed.isin(seeds)]
    return float(r.primary_rmse.mean()) if len(r) else np.nan, sorted(r.seed.tolist())


def pretrain_summary():
    rows = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for g, color in (("g1", "#1f77b4"), ("g2", "#d62728")):
        d = RUN / f"pretrain_{g}"
        if not (d / "pretrain_metrics.json").exists():
            continue
        m = json.loads((d / "pretrain_metrics.json").read_text())
        c = pd.read_csv(d / "curve.csv")
        ax.plot(c.step, c.holdout_rmse, color=color, label=f"{g} hold-out RMSE (eval)")
        ax.plot(c.step[1:], np.sqrt(c.train_mse_window[1:]), color=color, ls=":", label=f"{g} train RMSE (window, dropout on)")
        refs = m["info"]["holdout_reference_rmse"]
        rows.append(dict(arch=g, best_step=m["best_step"], best_holdout_rmse=m["best_holdout_rmse"], last_step=m["last_step"],
                         stop_reason=m["stop_reason"], train_minutes=m["train_seconds"] / 60, total_minutes=m["total_seconds"] / 60,
                         ref_global_mean=refs["global_mean"], ref_cell_dose_bin=refs["cell_dose_bin_mean"],
                         train_points=m["info"]["train_points"], holdout_points=m["info"]["holdout_points"],
                         parameters=m["info"]["parameters"], checkpoint_sha256=m["checkpoint_sha256"]))
    if rows:
        ax.axhline(rows[0]["ref_global_mean"], color="grey", lw=0.8, ls="--", label="global mean (train)")
        ax.axhline(rows[0]["ref_cell_dose_bin"], color="black", lw=0.8, ls="--", label="cell x dose-bin mean (train)")
        ax.set_xlabel("step (batch 512 points)"); ax.set_ylabel("RMSE (y = PTC/100)"); ax.set_ylim(0.14, 0.32)
        ax.set_title("NCI60-Pretrain-v1: compound-level 5% hold-out"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(OUT / "pretrain_curves.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "pretrain_summary.csv", index=False)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(pd.DataFrame(pretrain_summary()).T.to_string())
    new = sorted(d for d in RUN.glob("transfer_*") if (d / "summary.json").exists()
                 and json.loads((d / "summary.json").read_text())["status"] == "passed")
    ref_views = json.loads((EXISTING[0] / "task_manifest.json").read_text())
    view_check = {}
    for d in new + EXISTING[1:]:
        m = json.loads((d / "task_manifest.json").read_text())
        view_check[d.name] = all(m[k]["view_sha256"] == ref_views[k]["view_sha256"] for k in ref_views)
        if not view_check[d.name]:
            raise RuntimeError(f"RoleView drift in {d}")
    controls = []
    for d in new:
        r = pd.read_csv(d / "results.csv")
        for _, x in r[r.run == "control"].iterrows():
            controls.append(dict(run=d.name, pipeline=x.pipeline, seed=int(x.seed), k=int(x.k), primary_rmse=x.primary_rmse,
                                 reference_primary_rmse=x.control_reference_primary_rmse, max_abs_prediction_diff=x.control_max_abs_diff))
        for _, x in r[r.run == "repeat"].iterrows():
            controls.append(dict(run=d.name, pipeline=x.pipeline, seed=int(x.seed), k=int(x.k), primary_rmse=x.primary_rmse,
                                 reference_primary_rmse=np.nan, max_abs_prediction_diff=x.repeat_max_abs_diff, kind="repeat"))
    pd.DataFrame(controls).to_csv(OUT / "controls_and_repeats.csv", index=False)
    rows, surf = smry.collect([str(p) for p in EXISTING + new])
    dup = rows[rows.duplicated(["pipeline", "k", "seed"], keep=False)]
    for (pl, k, s), g in dup.groupby(["pipeline", "k", "seed"]):
        if g.primary_rmse.max() - g.primary_rmse.min() > 1e-9:
            raise RuntimeError(f"Cross-run mismatch {pl} k{k} seed {s}")
    rows = rows.drop_duplicates(["pipeline", "k", "seed"]).sort_values(["pipeline", "k", "seed"])
    rows.to_csv(OUT / "per_seed.csv", index=False)
    agg = rows.groupby(["pipeline", "k"]).agg(seeds=("seed", "nunique"), primary_mean=("primary_rmse", "mean"),
                                              primary_sd=("primary_rmse", "std"), recon_mean=("recon_rmse_hidden_query", "mean"),
                                              all_points_mean=("all_points_rmse", "mean")).reset_index()
    s2 = rows[rows.seed.isin(SCREEN) | (rows.pipeline == "A_ref")].groupby(["pipeline", "k"]).agg(
        primary_mean_41_42=("primary_rmse", "mean"), recon_mean_41_42=("recon_rmse_hidden_query", "mean")).reset_index()
    agg = agg.merge(s2, on=["pipeline", "k"], how="left"); agg.to_csv(OUT / "models.csv", index=False)
    pts = [p for p in ("D_gnn_pt_g1", "D_gnn_pt_g2") if ((rows.pipeline == p) & (rows.k == 1)).any()]
    screen = {p: seed_mean(rows, p, 1, SCREEN)[0] for p in pts}
    sel = min(screen, key=screen.get) if screen else None
    g = sel.rsplit("_", 1)[1] if sel else None
    comps = []
    def add(c, r):
        if c in surf and r in surf:
            comps.append(smry.compare(surf, c, r))
    if sel:
        for ref in (("D_morgan_g2", 1), ("D_gnn_g1", 1), ("D_gnn_g2", 1), ("C_tabm", 1), ("B_xgb", 1)):
            add((sel, 1), ref)
        for p in pts:
            add((p, 1), (p.replace("_pt", ""), 1)); add((p, 0), (p.replace("_pt", ""), 0))
        for ref in (("D_morgan_g2", 0), ("C_tabm", 0), ("D_gnn_g2", 0)):
            add((sel, 0), ref)
        add((sel, 1), (sel, 0))
        for p in ("D_gnn_pt_frozen_g1", "D_gnn_pt_frozen_g2"):
            add((p, 1), (sel, 1)); add((p, 1), ("D_morgan_g2", 1)); add((p, 0), (sel, 0))
    comps = pd.DataFrame(comps).drop_duplicates(["candidate", "reference"]); comps.to_csv(OUT / "comparisons.csv", index=False)
    # per target drug (surface RMSE on primary query points averaged over seeds 41/42, then over the drug's surfaces)
    per_drug = {}
    other = [p for p in pts if p != sel]
    for key in [(sel, 1), (sel, 0)] + [(p, kk) for p in other for kk in (1, 0)] + [
            ("D_gnn_g2", 1), ("D_gnn_g1", 1), ("D_morgan_g2", 1), ("C_tabm", 1), ("D_gnn_g2", 0), ("D_gnn_g1", 0),
            ("D_morgan_g2", 0), ("C_tabm", 0)]:
        if key[0] and key in surf:
            seeds = sorted(s for s in surf[key] if s in SCREEN)
            m = pd.concat([surf[key][s].rmse for s in seeds], axis=1).mean(1)
            drug = surf[key][seeds[0]].drug.loc[m.index]
            per_drug[f"{key[0]}_k{key[1]}"] = m.groupby(drug).mean()
    pdg = pd.DataFrame(per_drug); pdg.index.name = "target_drug"; pdg.to_csv(OUT / "per_target_drug.csv")
    # protocol decision (frozen before transfer results; see FROZEN_RECORD.md)
    dec = dict(selected_config=sel, screen_means_k1_41_42=screen)
    if sel:
        ref2 = {p: seed_mean(rows, p, 1, SCREEN)[0] for p in ("D_morgan_g2", "D_gnn_g1", "D_gnn_g2")}
        dec["references_k1_41_42"] = ref2
        dec["extension_triggered"] = bool(all(screen[sel] < v for v in ref2.values()))
        cand5, seeds5 = seed_mean(rows, sel, 1)
        dec["candidate_seeds"] = seeds5; dec["candidate_mean_all_seeds"] = cand5
        if len(seeds5) >= 5:
            five = {p: seed_mean(rows, p, 1) for p in ("D_morgan_g2", "D_gnn_g2")}
            strongest = min(five, key=lambda p: five[p][0])
            c = smry.compare(surf, (sel, 1), (strongest, 1))
            dec["final_gate"] = dict(strongest_same_information=strongest, relative_improvement=c["relative_improvement"],
                                     improved_seeds=c["improved_seeds"], paired_seeds=c["paired_seeds"],
                                     drug_bootstrap_ci95=[c["drug_bootstrap_rel_ci95_low"], c["drug_bootstrap_rel_ci95_high"]],
                                     passed=bool(c["relative_improvement"] >= 0.01 and c["improved_seeds"] >= 4))
    dec["view_identity"] = view_check
    dec["transfer_timing_minutes"] = {d.name: dict(
        jobs_wall_sum=float(pd.read_csv(d / "results.csv").wall_seconds.sum() / 60),
        process_wall=float(((d / "summary.json").stat().st_mtime - (d / "config.yaml").stat().st_mtime) / 60)) for d in new}
    dec["transfer_process_minutes_total"] = float(sum(v["process_wall"] for v in dec["transfer_timing_minutes"].values()))
    (OUT / "decision.json").write_text(json.dumps(dec, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(agg.round(6).to_string()); print(comps.round(5).to_string() if len(comps) else "no comparisons")
    print(pdg.round(4).to_string()); print(json.dumps(dec, indent=1, default=float))
    print(pd.DataFrame(controls).to_string() if controls else "no controls")


if __name__ == "__main__":
    main()

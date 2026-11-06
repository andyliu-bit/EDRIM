"""Collect comboKR 2.0 original-setting results into tables (reads our own JSON outputs only).

Usage (host or container): python summarize.py <runs/model_improvement/combokr2_original_v1>
Writes summary.json and results_table.csv in that directory.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

R = sys.argv[1]
PAPER = {  # Supplementary Table 1 (Pearson, mean +- sd over 5 seeds) and Table 4 (win-rate vs Baseline)
    "newcell": {"combokr2": 0.897, "baseline": 0.897, "winrate": 0.502},
    "newcombo": {"combokr2": 0.902, "baseline": 0.902, "winrate": 0.501},
    "newdrug": {"combokr2": 0.904, "baseline": 0.904, "winrate": 0.492},
}
TRUTHS = ["raw4x4_notz_clip0_100", "raw3x3_notz_clip0_100", "author_braid_fit_4x4_clip0_100",
          "author_braid_fit_3x3_clip0_100", "raw4x4_notz"]
rows, summary = [], {"paper": PAPER, "runs": {}}
for d in sorted(glob.glob(f"{R}/test/*")) + sorted(glob.glob(f"{R}/ablation/*")):
    pm, mm = f"{d}/posthoc_metrics.json", f"{d}/metrics.json"
    if not (os.path.exists(pm) and os.path.exists(mm)):
        continue
    p, m = json.load(open(pm)), json.load(open(mm))
    name = os.path.relpath(d, R).replace("\\", "/")
    for k, v in p["per_iters"].items():
        rec = {"run": name, "scenario": m["args"]["scenario"], "seed": m["args"]["seed"], "n_iters": int(k),
               "n_test": p["n_test"], "n_train_triplets": m["n_train_triplets"],
               "preimage": m["args"].get("preimage", "gd"), "landmarks": m["nystrom_landmarks"],
               "failed_braid_fits": m["results"][f"iters{k}"]["failed_braid_fits"],
               "raw_predict_s": round(m["timings"][f"iters{k}"]["raw_predict_s"], 1),
               "preimage_wall_s": round(m["timings"][f"iters{k}"]["preimage_wall_s"], 1),
               "preimage_s_per_surface_1cpu": round(m["timings"][f"iters{k}"]["preimage_s_per_surface_worker"], 4),
               "total_s": round(m["timings"]["total_s"], 1)}
        for t in TRUTHS:
            mt = v[t]
            rec[f"{t}|pearson_ckr2"] = round(mt["pearson_combokr2"], 4)
            rec[f"{t}|pearson_base"] = round(mt["pearson_baseline"], 4)
            rec[f"{t}|win_ckr2"] = round(mt["winrate_combokr2_lt_baseline"], 4)
        for t in ["raw4x4_notz_clip0_100|excl_MDA-MB-468"]:
            if t in v:
                rec[f"{t}|pearson_ckr2"] = round(v[t]["pearson_combokr2"], 4)
                rec[f"{t}|pearson_base"] = round(v[t]["pearson_baseline"], 4)
                rec[f"{t}|win_ckr2"] = round(v[t]["winrate_combokr2_lt_baseline"], 4)
        b = v["bridge_unclipped_3x3"]
        rec.update({"bridge|frac_gt100": round(b["frac_points_gt100"], 4),
                    "bridge|smRMSE_ckr2": round(b["combokr2"]["surface_macro_rmse_frac"], 4),
                    "bridge|smRMSE_neutral": round(b["braid_neutral"]["surface_macro_rmse_frac"], 4),
                    "bridge|smRMSE_hsa": round(b["hsa_measured_mono"]["surface_macro_rmse_frac"], 4),
                    "bridge|pearson_ckr2": round(b["combokr2"]["pearson"], 4),
                    "bridge|pearson_hsa": round(b["hsa_measured_mono"]["pearson"], 4),
                    "bridge|win_ckr2_vs_hsa": round(b["winrate_combokr2_lt_hsa"], 4),
                    "bridge|rmse_gt100_ckr2": round(b["combokr2"]["point_rmse_gt100_frac"], 4),
                    "bridge|rmse_le100_ckr2": round(b["combokr2"]["point_rmse_le100_frac"], 4),
                    "bridge|rmse_gt100_hsa": round(b["hsa_measured_mono"]["point_rmse_gt100_frac"], 4),
                    "bridge|rmse_le100_hsa": round(b["hsa_measured_mono"]["point_rmse_le100_frac"], 4)})
        rows.append(rec)
summary["runs"] = rows
# seed aggregation for test runs
agg = {}
for scen in ["newcell", "newcombo", "newdrug"]:
    rs = [r for r in rows if r["run"].startswith("test/") and r["scenario"] == scen]
    if not rs:
        continue
    keys = [k for k in rs[0] if "|" in k]
    agg[scen] = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs],
                 **{k: [float(np.mean([r[k] for r in rs if k in r])),
                        float(np.std([r[k] for r in rs if k in r], ddof=1)) if len(rs) > 1 else None]
                    for k in keys}}
summary["test_seed_aggregate"] = agg
for sel in glob.glob(f"{R}/cv/*_selection.json"):
    summary.setdefault("cv_selection", {})[os.path.basename(sel)] = json.load(open(sel))
if os.path.exists(f"{R}/baselines_all_seeds.json"):
    summary["baselines_all_seeds"] = {k: v["mean_sd"] for k, v in json.load(open(f"{R}/baselines_all_seeds.json")).items()}
json.dump(summary, open(f"{R}/summary.json", "w"), indent=1)
if rows:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{R}/results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
print(json.dumps(agg, indent=1)[:4000])
for r in rows:
    print(r["run"], r["n_iters"], "4x4:", r["raw4x4_notz_clip0_100|pearson_ckr2"], r["raw4x4_notz_clip0_100|pearson_base"],
          r["raw4x4_notz_clip0_100|win_ckr2"], "| bridge smRMSE ckr2/neutral/hsa:", r["bridge|smRMSE_ckr2"],
          r["bridge|smRMSE_neutral"], r["bridge|smRMSE_hsa"])

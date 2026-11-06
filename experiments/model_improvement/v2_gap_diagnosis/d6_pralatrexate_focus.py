"""V2 gap diagnosis, step D6: the pralatrexate pairs (VALIDATION only; training points read for description).

Background (D2): the 18 validation pairs containing pralatrexate carry ~28% of the BiSurfGNN-DeepSynBa squared-error
gap. This step records, reproducibly:
  * training description: per training partner of pralatrexate, mean deviation from lookup HSA, pralatrexate's
    lookup monotherapy level, drug order and dose design;
  * validation tracking: per validation pralatrexate pair, true mean deviation vs each model's mean correction;
  * the overall validation comparison with and without the pralatrexate pairs.

  .venv/Scripts/python.exe -I experiments/model_improvement/v2_gap_diagnosis/d6_pralatrexate_focus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_gap_analysis as d2  # noqa: E402

OUT = d2.DIAG / "pralatrexate"
DRUG = "pralatrexate"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    v, _, y, preds, edges, _, _ = d2.load_inputs()
    hsa = d2.add_derived(v, y, preds, edges)
    tr = pd.read_parquet(d2.S1 / "points_train.parquet",
                         columns=["pair_id", "deepsynba_drug_a", "deepsynba_drug_b", "deepsynba_cell", "response_percent",
                                  "hsa_lookup_percent", "mono_a_percent", "mono_b_percent", "dose_a_M", "dose_b_M"])

    def partners(frame, dev):
        m = ((frame.deepsynba_drug_a == DRUG) | (frame.deepsynba_drug_b == DRUG)).to_numpy()
        first = (frame.deepsynba_drug_a == DRUG).to_numpy()
        t = pd.DataFrame({"pair_id": frame.pair_id[m], "partner": np.where(first, frame.deepsynba_drug_b, frame.deepsynba_drug_a)[m],
                          "prala_is_drug1": first[m], "dev": dev[m], "y": frame.response_percent.to_numpy(float)[m],
                          "prala_mono": np.where(first, frame.mono_a_percent, frame.mono_b_percent)[m],
                          "partner_mono": np.where(first, frame.mono_b_percent, frame.mono_a_percent)[m],
                          "prala_dose_max": np.where(first, frame.dose_a_M, frame.dose_b_M)[m]})
        return m, t

    _, ttr = partners(tr, (tr.response_percent - tr.hsa_lookup_percent).to_numpy(float))
    train_tab = ttr.groupby("partner").agg(points=("dev", "size"), mean_dev=("dev", "mean"), sd_dev=("dev", "std"),
                                           mean_y=("y", "mean"), prala_mono=("prala_mono", "mean"),
                                           partner_mono=("partner_mono", "mean"), prala_is_drug1=("prala_is_drug1", "mean"),
                                           prala_dose_max=("prala_dose_max", "max")).sort_values("mean_dev")
    train_tab.round(4).to_csv(OUT / "train_partners.csv")
    mask, tva = partners(v, y - hsa)
    for k in ("BS_ens5", "XGB_ens5", "DSB_e484", "DSB_e047"):
        tva[f"corr_{k}"] = (preds[k] - hsa)[mask]
    val_tab = tva.groupby("partner").agg(points=("dev", "size"), true_mean_dev=("dev", "mean"),
                                         **{f"mean_corr_{k}": (f"corr_{k}", "mean") for k in ("BS_ens5", "XGB_ens5", "DSB_e484", "DSB_e047")}
                                         ).sort_values("true_mean_dev")
    val_tab.round(4).to_csv(OUT / "val_partners.csv")
    track = {k: float(np.corrcoef(val_tab.true_mean_dev, val_tab[f"mean_corr_{k}"])[0, 1])
             for k in ("BS_ens5", "XGB_ens5", "DSB_e484", "DSB_e047")}

    def table(sel):
        r = {k: d2.rmse(preds[k][sel] - y[sel]) for k in ["HSA", "BS_ens5", "XGB_ens5", "DSB_e484", "DSB_e047", *d2.BS_SEEDS, *d2.XGB_SEEDS]}
        r["BS_mean_seed"] = float(np.mean([r[k] for k in d2.BS_SEEDS]))
        r["XGB_mean_seed"] = float(np.mean([r[k] for k in d2.XGB_SEEDS]))
        r["BS_vs_DSB"] = r["BS_mean_seed"] / r["DSB_e484"] - 1
        r["BSens_vs_DSB"] = r["BS_ens5"] / r["DSB_e484"] - 1
        r["XGB_vs_DSB"] = r["XGB_mean_seed"] / r["DSB_e484"] - 1
        r["points"] = int(sel.sum())
        return r

    bs_sse = np.mean([(preds[k] - y) ** 2 for k in d2.BS_SEEDS], axis=0)
    gap = bs_sse - (preds["DSB_e484"] - y) ** 2
    train_active = train_tab.mean_dev < -15
    result = {"step": "D6", "test_rows_read": 0, "drug": DRUG,
              "train_pairs": int(len(train_tab)), "train_pairs_mean_dev_below_-15": int(train_active.sum()),
              "train_prala_lookup_mono_mean": float(ttr.prala_mono.mean()),
              "train_prala_is_drug1_share": float(ttr.prala_is_drug1.mean()),
              "val_pairs": int(len(val_tab)), "val_points": int(mask.sum()), "val_point_share": float(mask.mean()),
              "val_share_of_BS_DSB_gap": float(gap[mask].sum() / gap.sum()),
              "val_pair_tracking_corr": track,
              "all_validation": table(np.ones(len(y), bool)), "without_pralatrexate_pairs": table(~mask),
              "pralatrexate_pairs_only": table(mask)}
    with (OUT / "d6_summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=1, ensure_ascii=False, default=float) + "\n")
    print(json.dumps(result, indent=1, default=float))
    print(val_tab.round(2).to_string())


if __name__ == "__main__":
    main()

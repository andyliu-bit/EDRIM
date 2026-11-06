"""Eval-paper supplement G, final scoring on the 5 new-drug partitions (card rules; nothing is selected here).

Per partition, new-drug test, combination-cell pooled RMSE: lookup HSA, P4 (prepare.py predictions), XGBoost cap 20000
(mean of seeds 41-43; D11 reuses its existing seeds and also reports its 5-seed value). Drug-pair bootstrap (score_cold.py
scheme, seed 20261009, 2000 draws); XGBoost vs lookup HSA and P4 vs lookup HSA; Holm over the 10 tests, separately for
"all" (primary) and "excluding_pralatrexate". Self-check: the D11 5-seed XGBoost comparison reproduces cold_test_evaluation.json.
If any of the 12 new XGBoost runs is missing, writes evaluation_partial.json (no Holm, no decision); else evaluation.json
and final_table.csv.

  .venv/Scripts/python.exe -I experiments/eval_paper/cold_drug_partitions/score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

COLS = ["pair_id", "drug_a_id", "drug_b_id", "mono_a_percent", "mono_b_percent", "response_percent"]


def xgb_dir(part: int, seed: int) -> Path:
    return c.D11_DIR / f"xgb_cap20000_seed_{seed}" if part == c.D11_SEED else c.OUT / f"p{part}/xgb_seed_{seed}"


def aligned(z, key: str, source: np.ndarray, value: str) -> np.ndarray:
    if not np.array_equal(z[key], source):
        raise RuntimeError(f"prediction file not aligned with the cold_test frame ({value})")
    return np.asarray(z[value], float)


def main():
    if not sys.flags.isolated:
        raise SystemExit("run with python -I")
    missing = [str(xgb_dir(p, s)) for p in c.NEW_SEEDS for s in c.XGB_SEEDS if not (xgb_dir(p, s) / "cold_test_predictions.npz").exists()]
    complete = not missing
    base = json.loads((c.OUT / "baselines.json").read_text(encoding="utf-8"))
    d11_eval = json.loads((c.D11_DIR / "cold_test_evaluation.json").read_text(encoding="utf-8"))
    both = c.read_points(COLS)
    sets = ("all", "excluding_pralatrexate")
    res = {"card": "bisurfgnn/docs/EVAL_COLD_DRUG_PARTITIONS_20261018.md", "complete": complete, "missing_runs": missing,
           "bootstrap": {"unit": "test drug pair", "draws": c.DRAWS, "seed": c.BOOT_SEED}, "partitions": {}}
    for part in c.PARTS:
        te = c.frames(part, both)["cold_test"]
        src = te.source_combodrugseq.to_numpy()
        p4 = aligned(np.load(c.OUT / f"p{part}/p4_predictions.npz"), "cold_test_source", src, "cold_test_p4_percent")
        preds = {"lookup_hsa": {0: c.lookup_hsa(te)}, "p4": {0: p4}}
        seeds = (41, 42, 43, 44, 45) if part == c.D11_SEED else c.XGB_SEEDS
        xgb = {s: aligned(np.load(xgb_dir(part, s) / "cold_test_predictions.npz"), "source", src, "prediction_percent")
               for s in seeds if (xgb_dir(part, s) / "cold_test_predictions.npz").exists()}
        if all(s in xgb for s in c.XGB_SEEDS):
            preds["xgb"] = {s: xgb[s] for s in c.XGB_SEEDS}
        if part == c.D11_SEED:
            preds["xgb_5seed"] = xgb
        boot = c.bootstrap(te, preds)
        row = {"val": base["partitions"][str(part)]["cold_val"], "xgb_val": {}, "test": {}}
        for s in seeds:
            m = xgb_dir(part, s) / "metrics.json"
            if m.exists():
                mj = json.loads(m.read_text(encoding="utf-8"))
                assert mj["max_trees"] == 20000, m
                row["xgb_val"][s] = {k: mj[k] for k in ("best_trees", "at_cap", "cold_val_pooled_rmse_percent")}
        for label in sets:
            b = boot[label]
            assert abs(b["point"]["lookup_hsa"] - base["partitions"][str(part)]["cold_test"]["lookup_hsa"][label]) < 1e-9
            assert abs(b["point"]["p4"] - base["partitions"][str(part)]["cold_test"]["p4"][label]) < 1e-9
            comps = {f"{m}_vs_lookup_hsa": c.compare(b, m, "lookup_hsa") for m in preds if m != "lookup_hsa"}
            row["test"][label] = dict(pairs=b["pairs"], points=b["points"], point=b["point"], per_seed=b["per_seed"], comparisons=comps)
            if part == c.D11_SEED:
                want = d11_eval["results"][label]["comparisons"]["xgb_vs_lookup_hsa"]
                got = comps["xgb_5seed_vs_lookup_hsa"]
                assert abs(got["relative_change"] - want["relative_change"]) < 1e-12 and np.allclose(got["ci95"], want["ci95"], atol=1e-12, rtol=0), (got, want)
                row["test"][label]["deepsynba_seed10_from_D11"] = dict(
                    rmse=d11_eval["results"][label]["point"]["deepsynba"],
                    vs_lookup_hsa=d11_eval["results"][label]["comparisons"]["deepsynba_vs_lookup_hsa"])
        res["partitions"][part] = row
        print(part, json.dumps({k: {m: round(v, 4) for m, v in row["test"][k]["point"].items()} for k in sets}), flush=True)
        del te, preds, boot
    rows = []
    for label in sets:
        tests = {(p, m): res["partitions"][p]["test"][label]["comparisons"][f"{m}_vs_lookup_hsa"]
                 for p in c.PARTS for m in ("xgb", "p4") if f"{m}_vs_lookup_hsa" in res["partitions"][p]["test"][label]["comparisons"]}
        adj = c.holm({k: v["p"] for k, v in tests.items()}) if complete else {}
        for (p, m), v in tests.items():
            v["p_holm"] = adj.get((p, m))
            v["verdict"] = None if not complete else ("better" if adj[(p, m)] < 0.05 and v["relative_change"] < 0 else
                                                      "worse" if adj[(p, m)] < 0.05 else "no_difference")
            rows.append(dict(set=label, partition=p, method=m, rmse=res["partitions"][p]["test"][label]["point"][m],
                             lookup_hsa=res["partitions"][p]["test"][label]["point"]["lookup_hsa"],
                             relative_change=v["relative_change"], ci_low=v["ci95"][0], ci_high=v["ci95"][1], p=v["p"],
                             p_holm=v["p_holm"], verdict=v["verdict"]))
        if complete:
            assert len(tests) == 10
            pattern = {m: {k: sum(tests[(p, m)]["verdict"] == k for p in c.PARTS) for k in ("better", "no_difference", "worse")}
                       for m in ("xgb", "p4")}
            res.setdefault("decision", {})[label] = dict(
                pattern=pattern, xgb_not_better_partitions=5 - pattern["xgb"]["better"],
                wording_rule_met=bool(5 - pattern["xgb"]["better"] >= 4), primary=label == "all")
    name = "evaluation.json" if complete else "evaluation_partial.json"
    if complete:
        with (c.OUT / name).open("x", encoding="utf-8") as h:
            json.dump(res, h, indent=1, ensure_ascii=False, default=str)
        pd.DataFrame(rows).to_csv(c.OUT / "final_table.csv", index=False)
    else:
        (c.OUT / name).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(json.dumps(res.get("decision"), indent=1), "written", name, flush=True)


if __name__ == "__main__":
    main()

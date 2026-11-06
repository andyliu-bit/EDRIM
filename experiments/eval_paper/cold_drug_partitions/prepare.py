"""Eval-paper supplement G, step 1: the 4 new split files + lookup HSA and P4 on all 5 partitions (no selection).

1. The copied split rule must regenerate the saved D11 split.json exactly (drugs, names, counts); then the 4 new
   partitions are written once to runs/eval_paper/cold_drug_partitions/p<seed>/split.json (an existing file must equal
   the regenerated record).
2. Per partition: P4 fitted on that partition's training role only; lookup HSA and P4 scored on new-drug validation and
   test (combination-cell pooled RMSE, all pairs and excluding pralatrexate pairs). P4 predictions saved for score.py.
   The D11 lookup HSA test RMSE must reproduce cold_test_evaluation.json.

  .venv/Scripts/python.exe -I experiments/eval_paper/cold_drug_partitions/prepare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

KEYS = ("seed", "n_drugs", "test_drugs", "val_drugs", "test_drug_names", "val_drug_names", "counts")


def scores(frame, preds):
    keep = ~frame.pair_id.str.split("|").map(lambda p: c.PRALA in p).to_numpy()
    y = frame.response_percent.to_numpy(float)
    return {m: dict(all=c.rmse(p - y), excluding_pralatrexate=c.rmse((p - y)[keep])) for m, p in preds.items()}


def main():
    if not sys.flags.isolated:
        raise SystemExit("run with python -I")
    out_json = c.OUT / "baselines.json"
    if out_json.exists():
        raise SystemExit(f"{out_json} exists")
    both = c.read_points()
    assert np.abs(c.lookup_hsa(both) - both.hsa_lookup_percent.to_numpy(float)).max() <= 1e-9
    saved = json.loads(c.split_path(c.D11_SEED).read_text(encoding="utf-8"))
    again = c.make_split(c.D11_SEED, both)
    assert all(again[k] == saved[k] for k in KEYS), "copied split rule does not regenerate D11"
    splits = {c.D11_SEED: saved}
    for seed in c.NEW_SEEDS:
        rec, path = c.make_split(seed, both), c.split_path(seed)
        if path.exists():
            assert all(json.loads(path.read_text(encoding="utf-8"))[k] == rec[k] for k in KEYS), path
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as h:
                h.write(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        splits[seed] = rec
    tests = {s: set(r["test_drugs"]) for s, r in splits.items()}
    res = {"card": "bisurfgnn/docs/EVAL_COLD_DRUG_PARTITIONS_20261018.md", "lambda": c.LAM, "blocks": c.BLOCKS,
           "test_drug_overlap_with_D11": {s: len(t & tests[c.D11_SEED]) for s, t in tests.items()},
           "pralatrexate_role": {s: ("test" if c.PRALA in r["test_drugs"] else "val" if c.PRALA in r["val_drugs"] else "train")
                                 for s, r in splits.items()},
           "partitions": {}}
    all_drugs = sorted(set(both.drug_a_id) | set(both.drug_b_id))
    for seed in c.PARTS:
        f = c.frames(seed, both)
        model = c.fit_p4(f["train"], all_drugs)
        train_rmse = c.rmse(c.predict_p4(model, f["train"]) - f["train"].response_percent.to_numpy(float))
        row = dict(counts=splits[seed]["counts"], test_drugs=splits[seed]["test_drug_names"],
                   p4=dict(lsqr_iterations=model["iters"], coefficients=model["coefficients"],
                           unseen_columns=model["unseen_columns"], train_rmse=train_rmse))
        save = {}
        for role in ("cold_val", "cold_test"):
            fr = f[role]
            p4 = c.predict_p4(model, fr)
            row[role] = scores(fr, {"lookup_hsa": c.lookup_hsa(fr), "p4": p4})
            save[f"{role}_source"], save[f"{role}_p4_percent"] = fr.source_combodrugseq.to_numpy(), p4
        (c.OUT / f"p{seed}").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(c.OUT / f"p{seed}/p4_predictions.npz", **save)
        res["partitions"][seed] = row
        print(seed, json.dumps({k: row[k] for k in ("cold_val", "cold_test")}), flush=True)
        del f, model, save
    d11 = json.loads((c.D11_DIR / "cold_test_evaluation.json").read_text(encoding="utf-8"))["results"]
    for label in ("all", "excluding_pralatrexate"):
        got = res["partitions"][c.D11_SEED]["cold_test"]["lookup_hsa"][label]
        assert abs(got - d11[label]["point"]["lookup_hsa"]) < 1e-9, (label, got)
    with out_json.open("x", encoding="utf-8") as h:
        json.dump(res, h, indent=1, ensure_ascii=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

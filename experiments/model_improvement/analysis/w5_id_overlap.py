"""W5 (KBS paper outline): ID-only overlap / recoverability counts for DeepSynBa public split 3.

Reads IDENTIFIER and DOSE columns only. No response, growth, label or synergy column is loaded; the DeepSynBa
dose-response pickle is never opened.
  (1) indirect overlap: split-3 drug pairs / (cell, pair) triplets / raw ALMANAC combination rows that sit in the
      train/validation partitions of the project's early NCI benchmarks (OneNewScaffoldAnchored-v1,
      WarmPairOutDevelopment-v1);
  (2) S54 dose-level single-agent lookup coverage with the gate-S1 split-1 lookup and the model's backoff rule
      (cell, drug, dose) -> (drug, dose) -> global.

Inputs read (columns):
  nci_synergy_metadata.csv                      cell, drug1, drug2, split
  gate s1 drug/cell_mapping_used.csv            deepsynba_drug, structure_ids / deepsynba_cell, cell_id
  gate s1 lookup_split1_exact.csv               cell, drug, dose_key          (keys only, mean column skipped)
  gate s1 lookup_split1_drug_dose.csv           drug, dose_key
  combokr2 public_split/query_test.npz          matrix_key, c1, c2            (doses; file holds no labels)
  OneNewScaffold group/surface assignments      ids + role_fold_k
  OneNewScaffold learned_point_shards/fold_k/source_partition.parquet   source_combodrugseq, role
  nci_almanac_unclipped/points.parquet          source_combodrugseq, prediction_group_id, cell_id, drug_a_id, drug_b_id
  warm_pairout_development*/fold_k/group_assignments.parquet            prediction_group_id, warm_role

  .venv/Scripts/python.exe -I -B experiments/model_improvement/analysis/w5_id_overlap.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ME = Path("C:/LHZ/research/bisurfgnn-model-experiments")
MAIN = Path("C:/LHZ/research/bisurfgnn")
META = MAIN / "data/raw/external_benchmarks/deepsynba/data/nci_synergy_metadata.csv"
S1 = ME / "runs/model_improvement/public_split_gate_v1/s1"
QUERY = ME / "runs/model_improvement/combokr2_original_v1/public_split/query_test.npz"
TASK_E = ME / "runs/eval_paper/deepsynba_public_extra_rows/result.json"
ONS = MAIN / "data/processed/one_new_scaffold_benchmark_unclipped"
ONS_CLIPPED = MAIN / "data/processed/one_new_scaffold_benchmark"
POINTS = MAIN / "data/processed/nci_almanac_unclipped/points.parquet"
WARM = {"unclipped": MAIN / "data/processed/warm_pairout_development_unclipped",
        "clipped": MAIN / "data/processed/warm_pairout_development"}
EXCL = "MDA-MB-468"
FOLDS = range(5)


def pct(n, d):
    return {"n": int(n), "of": int(d), "pct": round(100.0 * n / d, 2)}


def deepsynba_ids():
    meta = pd.read_csv(META, usecols=["cell", "drug1", "drug2", "split"])
    meta["pair"] = ["|".join(sorted(p)) for p in zip(meta.drug1, meta.drug2)]  # string keys: isin() and tuples do not mix
    meta["triplet"] = meta.cell + "||" + meta.pair
    s55 = meta[meta.split == 3].reset_index(drop=True)
    s54 = s55[s55.cell != EXCL].reset_index(drop=True)
    assert len(s55) == 55120 and len(s54) == 54196 and s55.pair.nunique() == s54.pair.nunique() == 979
    assert s55.triplet.is_unique
    assert not set(s55.pair) & set(meta.pair[meta.split == 1])  # new-combination split
    return meta, {"S55": s55, "S54": s54}


def id_maps():
    drugs = pd.read_csv(S1 / "drug_mapping_used.csv", usecols=["deepsynba_drug", "structure_ids"])
    sid = {s: d for d, ids in zip(drugs.deepsynba_drug, drugs.structure_ids) for s in str(ids).split(";")}
    cells = pd.read_csv(S1 / "cell_mapping_used.csv", usecols=["deepsynba_cell", "cell_id"])
    cid = dict(zip(cells.cell_id, cells.deepsynba_cell))  # MDA-MB-468 keeps its cell_id (status excluded)
    return sid, cid


def to_triplet(frame, sid, cid):
    a, b, c = frame.drug_a_id.map(sid), frame.drug_b_id.map(sid), frame.cell_id.map(cid)
    ok = a.notna() & b.notna() & c.notna()
    return pd.Series([cc + "||" + "|".join(sorted((x, y))) if k else None for cc, x, y, k in zip(c, a, b, ok)], index=frame.index)


def overlap(sets, trip_in, pts):
    """sets: S54/S55 frames; trip_in: set of DeepSynBa triplets in the partition;
    pts: raw combination rows of split-3 triplets with boolean column 'in_part'."""
    out = {}
    for name, s in sets.items():
        hit = s.triplet.isin(trip_in)
        p = pts[pts.triplet.isin(set(s.triplet))]
        out[name] = {"pairs": pct(s.pair[hit].nunique(), s.pair.nunique()),
                     "triplets": pct(hit.sum(), len(s)),
                     "raw_combination_rows": pct(p.in_part.sum(), len(p))}
    return out


def part1(sets):
    sid, cid = id_maps()
    test_trips = set(sets["S55"].triplet)
    roles = [f"role_fold_{k}" for k in FOLDS]
    g = pd.read_parquet(ONS / "group_assignments.parquet", columns=["prediction_group_id", "drug_a_id", "drug_b_id", "cell_id"] + roles)
    gc = pd.read_parquet(ONS_CLIPPED / "group_assignments.parquet", columns=["prediction_group_id"] + roles)
    assert g[["prediction_group_id"] + roles].equals(gc), "clipped and unclipped OneNewScaffold splits differ"
    del gc
    g["triplet"] = to_triplet(g, sid, cid)
    pts = pd.read_parquet(POINTS, columns=["source_combodrugseq", "prediction_group_id", "cell_id", "drug_a_id", "drug_b_id"])
    pts["triplet"] = to_triplet(pts, sid, cid)
    assert set(pts.cell_id.map(cid).dropna()) == set(sets["S55"].cell)  # every DeepSynBa cell found in the raw file
    pts = pts[pts.triplet.isin(test_trips)][["source_combodrugseq", "prediction_group_id", "triplet"]].reset_index(drop=True)
    gt = g[g.triplet.isin(test_trips)]
    res = {"raw_rows_of_split3_triplets": {k: int(pts.triplet.isin(set(s.triplet)).sum()) for k, s in sets.items()},
           "split3_triplets_without_raw_rows": {k: int((~s.triplet.isin(set(pts.triplet))).sum()) for k, s in sets.items()},
           "split3_triplets_in_benchmark_any_role": {k: pct(s.triplet.isin(set(gt.triplet)).sum(), len(s)) for k, s in sets.items()}}

    ons, union_trips, union_rows = {}, set(), set()
    for k in FOLDS:
        sp = pd.read_parquet(ONS / f"learned_point_shards/fold_{k}/source_partition.parquet", columns=["source_combodrugseq", "role"])
        role_of = dict(zip(sp.source_combodrugseq, sp.role))
        prow = pts.source_combodrugseq.map(role_of)
        fold = {}
        for label, keep in (("train", {"train"}), ("validation", {"validation"}), ("train_or_validation", {"train", "validation"})):
            trips = set(gt.triplet[gt[f"role_fold_{k}"].isin(keep)])
            fold[label] = overlap(sets, trips, pts.assign(in_part=prow.isin(keep)))
            if label == "train_or_validation":
                union_trips |= trips
                union_rows |= set(pts.source_combodrugseq[prow.isin(keep)])
        ons[f"fold_{k}"] = fold
    ons["union_folds_0_4_train_or_validation"] = overlap(sets, union_trips, pts.assign(in_part=pts.source_combodrugseq.isin(union_rows)))
    res["OneNewScaffoldAnchored-v1"] = ons

    for variant, root in WARM.items():
        warm = {}
        for k in FOLDS:
            w = pd.read_parquet(root / f"fold_{k}/group_assignments.parquet", columns=["prediction_group_id", "warm_role"])
            base = set(g.prediction_group_id[g[f"role_fold_{k}"].isin(["train", "validation"])])
            assert set(w.prediction_group_id) == base, ("warm pair-out fold is not the base fold train+validation", variant, k)
            fold = {}
            for label, keep in (("train", {"train"}), ("validation", {"validation"}), ("train_or_validation", {"train", "validation"})):
                ids = set(w.prediction_group_id[w.warm_role.isin(keep)])
                trips = set(g.triplet[g.prediction_group_id.isin(ids)].dropna())
                fold[label] = overlap(sets, trips, pts.assign(in_part=pts.prediction_group_id.isin(ids)))
            warm[f"fold_{k}"] = fold
            if k == 0:
                warm["warm_role_values_fold_0"] = w.warm_role.value_counts().to_dict()
        res[f"WarmPairOutDevelopment-v1 ({variant})"] = warm
    return res


def part2(s54):
    q = np.load(QUERY, allow_pickle=False)
    assert set(q.files) == {"cdd", "c1", "c2", "matrix_key"}, q.files  # IDs and doses only
    keys = q["matrix_key"].astype(str)
    s54 = s54.assign(key=s54.drug1 + "_" + s54.drug2 + "_" + s54.cell).set_index("key")
    assert len(keys) == len(s54) and set(keys) == set(s54.index)
    m = s54.loc[keys]
    da, db = q["c1"][:, [0, 3, 6]], q["c2"][:, :3]  # row-major 3x3: drug1 dose by row i, drug2 dose by column j
    assert np.allclose(q["c1"].reshape(-1, 3, 3), da[:, :, None]) and np.allclose(q["c2"].reshape(-1, 3, 3), db[:, None, :])
    ex = pd.read_csv(S1 / "lookup_split1_exact.csv", usecols=["cell", "drug", "dose_key"], dtype=str)
    dd = pd.read_csv(S1 / "lookup_split1_drug_dose.csv", usecols=["drug", "dose_key"], dtype=str)
    assert (len(ex), len(dd)) == (28544, 487)  # s1_report.json lookup.exact_keys / drug_dose_keys
    exact, by_drug = set(zip(ex.cell, ex.drug, ex.dose_key)), set(zip(dd.drug, dd.dose_key))
    fmt = np.vectorize(lambda x: f"{float(x):.6g}")  # = s1_prepare.dose_key
    level = {}
    lv = np.empty((len(m), 6), dtype=object)
    for side, drugs, doses in ((0, m.drug1.to_numpy(), fmt(da)), (1, m.drug2.to_numpy(), fmt(db))):
        for i in range(3):
            for r, (c, d, x) in enumerate(zip(m.cell.to_numpy(), drugs, doses[:, i])):
                lv[r, 3 * side + i] = "exact" if (c, d, x) in exact else ("drug_dose" if (d, x) in by_drug else "global")
    flat = pd.Series(lv.ravel())
    level["entries"] = flat.value_counts().to_dict() | {"total": int(flat.size)}
    uniq = pd.DataFrame({"cell": np.repeat(m.cell.to_numpy(), 6),
                         "drug": np.concatenate([np.repeat(m.drug1.to_numpy()[:, None], 3, 1), np.repeat(m.drug2.to_numpy()[:, None], 3, 1)], 1).ravel(),
                         "dose": np.concatenate([fmt(da), fmt(db)], 1).ravel(), "level": lv.ravel()}).drop_duplicates()
    assert not uniq.duplicated(["cell", "drug", "dose"]).any()
    level["unique_cell_drug_dose_keys"] = uniq.level.value_counts().to_dict() | {"total": int(len(uniq))}
    both = (lv[:, :3] == "exact")[:, :, None] & (lv[:, 3:] == "exact")[:, None, :]
    level["combination_cells_both_anchors_exact"] = pct(both.sum(), both.size)
    stored = json.loads(TASK_E.read_text(encoding="utf-8"))["lookup_fallbacks"]["S54"]
    level["task_E_stored_tally_S54"] = stored
    assert {k: v for k, v in level["entries"].items() if k != "total"} == stored, "ID-only recount differs from task E"
    return level


def main():
    _, sets = deepsynba_ids()
    out = {"note": "ID and dose columns only; no response/label column read; DeepSynBa pickle not opened",
           "part1_indirect_overlap": part1(sets), "part2_S54_mono_lookup": part2(sets["S54"])}
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

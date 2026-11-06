"""O'Neil comparison (docs/ONEIL_COMPARISON_CARD_20261020.md v2, DEVIATIONS D13): point tables.

train/validation: responses and monotherapy anchors from the old project's v2 development surfaces.
test: IDs, doses and anchors only. Doses come from surface_roles.jsonl, anchors from the AACR single-agent
workbook through the old v2 code (workbook_rows / finite_replicates / interpolate). The combination workbook
is never opened here; an audit hook refuses it and every file the card forbids.
Same schema as public_split_gate/s1_prepare.py plus `batch`; one row per combination cell (4x4, no zero dose).

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/data/prepare_points.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode = True            # the old repo is read-only: no __pycache__ writes there
ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT.parent / "bisurfgnn"
AR = OLD / "runs/audit_revision"
FORBIDDEN = ("oneil_confirmation_eval_v1", "oneil_per_surface.csv", "external_combokr2_oneil_newcell_v1",
             "oneil_fitted_Braids.npy", "supp_1_w2lrww.xls")       # last one: AACR combination workbook


def _guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, Path)):
        path = str(args[0]).replace("\\", "/")
        if any(f in path for f in FORBIDDEN):
            raise PermissionError(f"Card forbids opening {path}")


sys.addaudithook(_guard)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OLD / "experiments/audit_revision"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402

import prepare_oneil_development_v2 as v2  # noqa: E402
from bisurfgnn.data.cellminer import normalize_cell_label  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "runs/oneil_v1/data"
DOSE_REFERENCE_M = 1e-12                  # s1_prepare.py: dose_model = log10(1 + D / 1e-12 M)
MONO_UPPER = 3.0
ROLE = {"train": "train", "validation": "validation", "confirmation": "test"}
EXPECTED = {"train": (349, 10590), "validation": (116, 3528), "test": (118, 3624)}   # pairs, surfaces


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def mono_curves(contexts):
    """v2 monotherapy curves (oneil_response_v2.json): pool finite replicates per (batch, cell, drug, dose)."""
    source = {r["role"]: r for r in json.loads(v2.SOURCE.read_text(encoding="utf-8"))["records"]}
    pools = defaultdict(list)
    for _, cells, strings in v2.workbook_rows(source["single_agent"]):
        meta = [v2.value(cells.get(c), strings) for c in ("A", "B", "C", "D")]
        v2.require(all(v is not None for v in meta), "Missing monotherapy metadata")
        if tuple(meta[:3]) not in contexts:
            continue
        reps = v2.finite_replicates(cells, strings, ("E", "F", "G", "H", "I", "J"))
        if reps:
            pools[(*meta[:3], v2.positive_dose(meta[3]))].extend(reps)
    curves = defaultdict(dict)
    for key, reps in pools.items():
        curves[key[:3]][key[3]] = math.fsum(reps) / len(reps)
    return curves


def surface_points(s, dev, curves):
    """16 (dose_a, dose_b, mono_a, mono_b, response, own_mono_a, own_mono_b) in canonical drug order (a < b)."""
    a, b = s["drugs"]
    own = [(Decimal(da), Decimal(db), v2.interpolate(curves[(s["batch"], s["cell"], a)], Decimal(da)),
            v2.interpolate(curves[(s["batch"], s["cell"], b)], Decimal(db))) for da, db in s["dose_pairs"]]
    if dev is None:
        return [(da, db, ma, mb, None, ma, mb) for da, db, ma, mb in own]
    points = []
    for p in dev["points"]:
        flip = p["drug_a"] != a
        da, db = (p["dose_b_uM"], p["dose_a_uM"]) if flip else (p["dose_a_uM"], p["dose_b_uM"])
        ma, mb = (p["mono_b"], p["mono_a"]) if flip else (p["mono_a"], p["mono_b"])
        points.append((Decimal(da), Decimal(db), ma, mb, p["response"]))
    by_dose = {(da, db): (ma, mb) for da, db, ma, mb in own}
    return [(*p, *by_dose[p[:2]]) for p in points]


def main():
    expression = np.load(AR / "oneil_depmap_coverage_v1/expression.npz", allow_pickle=False)
    model_id = dict(zip(expression["cells"].tolist(), expression["model_ids"].tolist()))
    surfaces = read_jsonl(AR / "oneil_pairout_metadata_v1/surface_roles.jsonl")
    dev = {tuple(s["source_record_ordinals"]): s for role in ("train", "validation")
           for s in read_jsonl(AR / f"oneil_development_v2/{role}_surfaces.jsonl")}
    outside = Counter(ROLE[s["role"]] for s in surfaces if normalize_cell_label(s["cell"]) not in model_id)
    surfaces = [s for s in surfaces if normalize_cell_label(s["cell"]) in model_id]
    for role, expected in EXPECTED.items():     # card counts: all role surfaces in the 30 cells, before eligibility
        keys = [(s["batch"], s["cell"], tuple(s["drugs"])) for s in surfaces if ROLE[s["role"]] == role]
        got = (len({k[2] for k in keys}), len(set(keys)))
        if got != expected or len(keys) != len(set(keys)):
            raise SystemExit(f"STOP {role}: pairs/surfaces {got} ({len(keys)} records) != card {expected}")
    curves = mono_curves({(s["batch"], s["cell"], d) for s in surfaces for d in s["drugs"]})

    rows, report, anchor_diff = defaultdict(list), {}, []
    excluded = defaultdict(Counter)
    for s in surfaces:
        role = ROLE[s["role"]]
        a, b = s["drugs"]
        assert a < b and len(s["dose_pairs"]) == 16, s
        d = dev.get(tuple(s["source_record_ordinals"])) if role != "test" else None
        assert (d is None) == (role == "test"), "development surface missing or test surface decoded"
        if d is not None:
            assert (d["batch"], d["cell"], d["drugs"], d["role"]) == (s["batch"], s["cell"], s["drugs"], s["role"])
        points = surface_points(s, d, curves)
        if d is not None:
            anchor_diff += [(p[2] == p[5]) and (p[3] == p[6]) for p in points]
            eligible = d["primary_eligible"]
            reasons = {r for p in d["points"] for r in p["reasons"]}
        else:   # input half of the v2 rule; the 16-finite-responses half is applied at joint scoring
            reasons = {f"mono_{arm}_not_estimable" for p in points for arm, m in (("a", p[2]), ("b", p[3])) if m is None}
            eligible = not reasons
        if not eligible:
            excluded[role].update(reasons | {"surfaces"})
            continue
        rank_a = {x: i + 1 for i, x in enumerate(sorted({p[0] for p in points}))}
        rank_b = {x: j + 1 for j, x in enumerate(sorted({p[1] for p in points}))}
        assert len(rank_a) == len(rank_b) == 4 and len({p[:2] for p in points}) == 16, s
        cell = normalize_cell_label(s["cell"])
        pair = f"{a}|{b}"
        for da, db, ma, mb, y, _, _ in sorted(points, key=lambda p: (rank_a[p[0]], rank_b[p[1]])):
            rows[role].append(dict(matrix_key=f"{s['batch']}|{cell}|{pair}", pair_id=pair, deepsynba_cell=s["cell"],
                                   cell_id=cell, biological_cell_id=model_id[cell], deepsynba_drug_a=a,
                                   deepsynba_drug_b=b, drug_a_id=a, drug_b_id=b, row_i=rank_a[da], col_j=rank_b[db],
                                   dose_a_M=float(da * Decimal("1e-6")), dose_b_M=float(db * Decimal("1e-6")),
                                   mono_a_ratio_raw=ma, mono_b_ratio_raw=mb, response=y, batch=s["batch"]))

    OUT.mkdir(parents=True, exist_ok=True)
    tables = {}
    for role in ("train", "validation", "test"):
        f = pd.DataFrame(rows[role])
        f.insert(1, "surface_index", pd.factorize(f.matrix_key)[0])
        f["mono_a_percent"], f["mono_b_percent"] = 100.0 * f.mono_a_ratio_raw, 100.0 * f.mono_b_ratio_raw
        f["dose_a_model"] = np.log10(1.0 + f.dose_a_M / DOSE_REFERENCE_M)
        f["dose_b_model"] = np.log10(1.0 + f.dose_b_M / DOSE_REFERENCE_M)
        for arm in ("a", "b"):
            f[f"mono_response_{arm}"] = f[f"mono_{arm}_ratio_raw"].clip(0.0, MONO_UPPER)
        if role == "test":     # card: no response and no HSA on test before the joint scoring script
            f = f.drop(columns=["response"])
        else:
            f["response_percent"] = 100.0 * f.response
            f["hsa_lookup_percent"] = np.minimum(f.mono_a_percent, f.mono_b_percent)
        n_pairs, n_surf = f.pair_id.nunique(), f.matrix_key.nunique()
        f.to_parquet(OUT / f"points_{'test_inputs' if role == 'test' else role}.parquet", index=False)
        tables[role] = f
        mono = np.concatenate([f.mono_a_ratio_raw, f.mono_b_ratio_raw])
        anchors = pd.concat([f[["batch", "cell_id", f"drug_{x}_id", f"dose_{x}_M", f"mono_{x}_ratio_raw"]].set_axis(
            ["batch", "cell", "drug", "dose", "mono"], axis=1) for x in ("a", "b")]).drop_duplicates()
        report[role] = dict(eligible_pairs=int(n_pairs), eligible_surfaces=int(n_surf), rows=len(f),
                            surfaces_outside_30_cells=outside[role], excluded=dict(excluded[role]),
                            anchors_clipped_rows=int(((mono < 0) | (mono > MONO_UPPER)).sum()),
                            anchors_clipped_unique=int(((anchors.mono < 0) | (anchors.mono > MONO_UPPER)).sum()),
                            anchor_ratio_range=[float(mono.min()), float(mono.max())],
                            dose_model_range=[float(f[["dose_a_model", "dose_b_model"]].min().min()),
                                              float(f[["dose_a_model", "dose_b_model"]].max().max())],
                            surfaces_by_batch=dict(sorted(Counter(f.drop_duplicates("matrix_key").batch).items())))
        print(role, json.dumps(report[role], ensure_ascii=False), flush=True)

    # Response-free anchor path reproduces the sealed v2 development anchors exactly
    report["anchor_path_reproduces_v2_dev"] = dict(points=len(anchor_diff), identical=int(sum(anchor_diff)))
    assert all(anchor_diff), report["anchor_path_reproduces_v2_dev"]

    # ID-only statistic (card): test (batch, cell, drug) contexts not seen in train surfaces
    ctx = {r: {(b, c, d) for b, c, x, y in tables[r][["batch", "cell_id", "drug_a_id", "drug_b_id"]].drop_duplicates()
               .itertuples(index=False) for d in (x, y)} for r in tables}
    test_s = tables["test"].drop_duplicates("matrix_key")
    for ref, seen in (("train", ctx["train"]), ("train_or_validation", ctx["train"] | ctx["validation"])):
        unseen = ctx["test"] - seen
        hit = test_s[[((b, c, x) in unseen) or ((b, c, y) in unseen) for b, c, x, y in
                      test_s[["batch", "cell_id", "drug_a_id", "drug_b_id"]].itertuples(index=False)]]
        report[f"test_contexts_unseen_in_{ref}"] = dict(
            unseen=len(unseen), total=len(ctx["test"]), surfaces_affected=len(hit),
            surfaces_affected_by_batch=dict(sorted(Counter(hit.batch).items())))

    # cells, drugs, pairs
    cells = sorted(set(tables["train"].cell_id) | set(tables["validation"].cell_id) | set(tables["test"].cell_id))
    labels = {normalize_cell_label(s["cell"]): s["cell"] for s in surfaces}
    pd.DataFrame(dict(cell_id=cells, source_label=[labels[c] for c in cells],
                      depmap_model_id=[model_id[c] for c in cells])).to_csv(OUT / "cells_oneil.csv", index=False)
    identity = json.loads((AR / "oneil_identity_audit_v1/validated_identity.json").read_text(encoding="utf-8"))
    drugs = []
    for entry in identity["drugs"]:
        smiles = {c["standard"]["canonical_smiles"] for c in entry["candidates"]}
        keys = {c["standard"]["inchikey"] for c in entry["candidates"]}
        assert len(smiles) == len(keys) == 1, (entry["name"], smiles)   # ABT-888: 2 candidates, one parent
        smi = smiles.pop()
        assert Chem.MolFromSmiles(smi) is not None, (entry["name"], smi)   # Pt complexes keep their fragments
        drugs.append(dict(drug_id=entry["name"], canonical_smiles=smi, inchikey=keys.pop(),
                          identity_candidates=len(entry["candidates"]),
                          removed_fragments=";".join(f for c in entry["candidates"] for f in c["standard"]["removed_fragments"])))
    drugs = pd.DataFrame(drugs).sort_values("drug_id")
    drugs.to_csv(OUT / "drugs_oneil.csv", index=False)
    used = set().union(*(set(t.drug_a_id) | set(t.drug_b_id) for t in tables.values()))
    assert len(drugs) == 38 and set(drugs.drug_id) == used, sorted(used ^ set(drugs.drug_id))
    pairs = pd.concat([t.groupby("pair_id").agg(drug_a_id=("drug_a_id", "first"), drug_b_id=("drug_b_id", "first"),
                                                surfaces=("matrix_key", "nunique")).reset_index().assign(role=r)
                       for r, t in tables.items()])
    assert pairs.pair_id.is_unique, "a drug pair occurs in more than one role"
    pairs.to_csv(OUT / "pairs_by_role.csv", index=False)
    report.update(cells=len(cells), drugs=len(drugs),
                  abt888_candidates=int(drugs.set_index("drug_id").loc["ABT-888", "identity_candidates"]))
    (OUT / "prepare_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ROLE.values()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

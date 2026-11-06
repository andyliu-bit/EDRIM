"""NCI60-Pretrain-v1 step 3: fold-0 leakage exclusion list (protocol §3), frozen BEFORE pretraining.

Uses only split ID files (group_assignments.parquet: IDs/roles; cluster_assignments.csv; drugs.csv structures).
No response column is read; test.parquet / points.parquet are guarded.

Excluded ALMANAC set E = held-out drugs (heldout_drug_id) of every fold-0 validation and test group, plus every
ALMANAC drug in their chemical clusters. An NCI-60 compound is excluded if, relative to ANY drug in E, it has
  (a) the same NSC (including frozen name aliases), (b) the same identity group (connectivity key) or the same
  canonical SMILES, (c) the same non-empty Bemis-Murcko scaffold, or (d) Morgan r=2/2048 Tanimoto >= 0.70
  (inclusive, 1e-12 tolerance) -- the edge rules of chemical_splits.build_chemical_clusters.
Outputs: runs/model_improvement/nci60_pretrain_v1/freeze/{exclusion_list.csv, excluded_almanac_drugs.csv,
         exclusion_freeze.json}; appends FROZEN_RECORD.md and STEP_LOG.md.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, DataStructs, rdBase  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

from bisurfgnn.pretrain.nci60 import DATA_ROOT, append_log, forbid_test_responses, sha256_file  # noqa: E402

SPLIT = DATA_ROOT / "data/processed/one_new_scaffold_benchmark_unclipped"
ALM = DATA_ROOT / "data/processed/nci_almanac_unclipped"
OUT = ROOT / "data/interim/nci60_pretrain_v1"
RUN = ROOT / "runs/model_improvement/nci60_pretrain_v1"
FRZ = RUN / "freeze"
THRESHOLD = 0.70


def main():
    FRZ.mkdir(parents=True, exist_ok=True)
    if (FRZ / "exclusion_freeze.json").exists():
        raise FileExistsError("exclusion already frozen")
    t0 = time.monotonic()
    cols = ["prediction_group_id", "heldout_drug_id", "partner_drug_id", "heldout_cluster_id", "partner_cluster_id",
            "heldout_cluster_bin", "partner_cluster_bin", "role_fold_0"]
    with forbid_test_responses() as guard:
        g = pd.read_parquet(SPLIT / "group_assignments.parquet", columns=cols)
    clusters = pd.read_csv(SPLIT / "cluster_assignments.csv", keep_default_na=False)
    drugs = pd.read_csv(ALM / "drugs.csv", keep_default_na=False)
    held = g[g.role_fold_0.isin(["validation", "test"])]
    roles = {r: sorted(set(held.loc[held.role_fold_0 == r, "heldout_drug_id"])) for r in ("validation", "test")}
    held_clusters = sorted(set(held.heldout_cluster_id))
    E = clusters[clusters.cluster_id.isin(held_clusters)].copy()
    cmap = dict(zip(clusters.drug_id, clusters.cluster_id))
    assert all(cmap[d] == c for d, c in zip(held.heldout_drug_id, held.heldout_cluster_id)), "cluster map drift"
    assert set(roles["validation"]) | set(roles["test"]) <= set(E.drug_id)
    assert not (set(roles["validation"]) & set(roles["test"]))
    train_groups = g[g.role_fold_0 == "train"]
    train_drugs = set(train_groups.heldout_drug_id) | set(train_groups.partner_drug_id)
    assert not (set(E.drug_id) & train_drugs), "held-out cluster drug appears in a training group"
    E["role"] = np.where(E.drug_id.isin(roles["validation"]), "validation_target",
                         np.where(E.drug_id.isin(roles["test"]), "test_target", "cluster_member"))
    vclu = set(held.loc[held.role_fold_0 == "validation", "heldout_cluster_id"])
    E["cluster_role"] = np.where(E.cluster_id.isin(vclu), "validation", "test")
    # audit: clusters in bins 0/1 (test/validation bins) that hold no legal held-out group
    bins = pd.concat([g[["heldout_cluster_id", "heldout_cluster_bin"]].set_axis(["c", "b"], axis=1),
                      g[["partner_cluster_id", "partner_cluster_bin"]].set_axis(["c", "b"], axis=1)]).drop_duplicates()
    assert not bins.c.duplicated().any()
    bin01 = set(bins.c[bins.b.isin([0, 1])])
    extra_bin01 = sorted(bin01 - set(held_clusters))
    # NSC set (frozen name aliases resolved both ways)
    nsc_of = dict(zip(drugs.drug_id, drugs.nsc.astype(int)))
    alias = {r.drug_id: r.structure_alias_of for r in drugs.itertuples() if r.structure_alias_of}
    e_ids = set(E.drug_id)
    for a, b in alias.items():
        if a in e_ids or b in e_ids:
            e_ids |= {a, b}
    excluded_nsc = sorted(nsc_of[d] for d in e_ids)
    d_e = drugs[drugs.drug_id.isin(e_ids)]
    assert (d_e.set_index("drug_id").canonical_smiles.reindex(E.drug_id).fillna("") == E.set_index("drug_id").canonical_smiles).all()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    efp = {r.drug_id: gen.GetFingerprint(Chem.MolFromSmiles(r.canonical_smiles)) for r in E.itertuples()}
    comp = pd.read_parquet(OUT / "nci60_compounds.parquet")
    fps = [DataStructs.CreateFromBinaryText(b) for b in comp.morgan2048]
    # fingerprint consistency: compounds identical to an ALMANAC drug must reproduce its fingerprint
    sims = np.zeros((len(comp), len(E)))
    for j, d in enumerate(E.drug_id):
        sims[:, j] = DataStructs.BulkTanimotoSimilarity(efp[d], fps)
    best = sims.argmax(1); max_sim = sims.max(1)
    scaf = set(E.murcko_scaffold[E.murcko_scaffold != ""])
    flags = pd.DataFrame({
        "same_nsc": comp.nsc.isin(excluded_nsc).to_numpy(),
        "same_identity_group": comp.identity_group.isin(set(E.identity_group)).to_numpy(),
        "same_canonical_smiles": comp.canonical_smiles.isin(set(E.canonical_smiles)).to_numpy(),
        "same_murcko_scaffold": ((comp.murcko_scaffold != "") & comp.murcko_scaffold.isin(scaf)).to_numpy(),
        "tanimoto_ge_070": (max_sim + 1e-12 >= THRESHOLD)})
    excl = flags.any(axis=1).to_numpy()
    lst = comp.loc[excl, ["nsc", "canonical_smiles", "identity_group", "murcko_scaffold"]].copy()
    lst["reasons"] = flags[excl].apply(lambda r: "|".join(k for k, v in r.items() if v), axis=1).to_numpy()
    lst["max_tanimoto"] = np.round(max_sim[excl], 6)
    lst["nearest_almanac_drug"] = E.drug_id.to_numpy()[best[excl]]
    lst["nearest_cluster_role"] = E.cluster_role.to_numpy()[best[excl]]
    # excluded NSCs of E that are not (or no longer) in the structured NCI-60 table are listed too
    absent = sorted(set(excluded_nsc) - set(lst.nsc))
    lst = pd.concat([lst, pd.DataFrame({"nsc": absent, "reasons": "almanac_heldout_nsc_not_in_structured_nci60"})], ignore_index=True)
    lst = lst.sort_values("nsc").reset_index(drop=True)
    lst.to_csv(FRZ / "exclusion_list.csv", index=False)
    E.merge(drugs[["drug_id", "nsc", "preferred_name"]], on="drug_id").to_csv(FRZ / "excluded_almanac_drugs.csv", index=False)
    h = sha256_file(FRZ / "exclusion_list.csv")
    pts = pd.read_parquet(OUT / "nci60_points.parquet", columns=["nsc"])
    pts_excl = pts.nsc.isin(set(lst.nsc))
    # audit: NCI-60 compounds matching TRAINING-role ALMANAC drugs (allowed, protocol §3.4)
    T = clusters[clusters.drug_id.isin(train_drugs)]
    tr_match = comp.nsc.isin([nsc_of[d] for d in T.drug_id]) | comp.identity_group.isin(set(T.identity_group))
    rec = dict(
        protocol="NCI60-Pretrain-v1", fold=0, rdkit=rdBase.rdkitVersion, threshold_inclusive=THRESHOLD,
        inputs={p.name: sha256_file(p) for p in (SPLIT / "group_assignments.parquet", SPLIT / "cluster_assignments.csv",
                                                 ALM / "drugs.csv", OUT / "nci60_compounds.parquet", OUT / "nci60_points.parquet")},
        parquet_reads=guard.accesses, response_columns_read=0,
        heldout_validation_drugs=roles["validation"], heldout_test_drugs=roles["test"],
        heldout_clusters=len(held_clusters), excluded_almanac_drugs=sorted(e_ids), excluded_almanac_drug_count=len(e_ids),
        excluded_almanac_by_role=E.role.value_counts().to_dict(), excluded_nsc=excluded_nsc,
        bin01_clusters_without_heldout_groups=len(extra_bin01),
        bin01_drugs_without_heldout_groups=sorted(clusters.drug_id[clusters.cluster_id.isin(extra_bin01)]),
        nci60_compounds_structured=len(comp), nci60_compounds_excluded=int(excl.sum()),
        nci60_excluded_by_reason={k: int(v) for k, v in flags.sum().items()},
        nci60_excluded_only_by_reason={k: int((flags[k] & (flags.sum(axis=1) == 1)).sum()) for k in flags},
        heldout_nsc_present_in_structured_nci60=sorted(set(excluded_nsc) & set(comp.nsc)),
        exclusion_list_rows=len(lst), exclusion_list_sha256=h,
        nci60_points_total=len(pts), nci60_points_excluded=int(pts_excl.sum()), nci60_points_remaining=int((~pts_excl).sum()),
        nci60_compounds_remaining=int(pts.nsc[~pts_excl].nunique()),
        training_role_almanac_drugs_matched_in_nci60_kept=int(tr_match.sum()), seconds=time.monotonic() - t0)
    (FRZ / "exclusion_freeze.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    msg = (f"STEP3 exclusion (fold 0): E = {len(e_ids)} ALMANAC drugs ({len(roles['validation'])} validation targets, "
           f"{len(roles['test'])} test targets, {len(held_clusters)} clusters); NCI-60 excluded compounds {int(excl.sum()):,} "
           f"of {len(comp):,} (by reason {rec['nci60_excluded_by_reason']}); points excluded {rec['nci60_points_excluded']:,}, "
           f"remaining {rec['nci60_points_remaining']:,} / {rec['nci60_compounds_remaining']:,} compounds; "
           f"exclusion_list.csv sha256 {h}")
    append_log(RUN / "STEP_LOG.md", msg)
    if not (RUN / "FROZEN_RECORD.md").exists():
        (RUN / "FROZEN_RECORD.md").write_text("# NCI60-Pretrain-v1 冻结记录（追加，不改写）\n", encoding="utf-8")
    append_log(RUN / "FROZEN_RECORD.md", "Leakage exclusion frozen BEFORE any pretraining: " + msg +
               f". Held-out validation drugs {roles['validation']}; test drugs {roles['test']}. "
               "Pretraining must assert this SHA-256 before loading data.")
    print(msg)


if __name__ == "__main__":
    main()

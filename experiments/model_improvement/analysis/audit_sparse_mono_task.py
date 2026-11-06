"""SparseMono-v1 feasibility + real-data leakage audit (no training, no test access)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import yaml

from bisurfgnn.sparse_mono.task import build_role, choose_supports, cluster_folds, correction_examples, monotherapy_table, with_target

spec = importlib.util.spec_from_file_location("fg", Path(__file__).with_name("diagnose_fit_vs_generalization.py"))
fitgen = importlib.util.module_from_spec(spec); spec.loader.exec_module(fitgen)


def main(cfg_path, out):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")); cfg["device"] = "cpu"
    orig = pd.read_parquet
    def guard(path, *a, **k):
        if Path(path).name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError("Forbidden response source")
        return orig(path, *a, **k)
    with patch.object(pd, "read_parquet", guard):
        frames, banks, _, _ = fitgen.frozen_tables(cfg)
        from bisurfgnn.benchmark_config import load_benchmark_config
        bench = load_benchmark_config((ROOT / cfg["data_root"]).resolve() / cfg["benchmark_config"])
        assign = pd.read_parquet(bench.output_dir / "surface_assignments.parquet")
    rep = {"test_response_rows_read": 0}
    for role, f in frames.items():
        m = with_target(f, assign); sup = choose_supports(m)
        v1, t1 = build_role(f, assign, 1); v0, _ = build_role(f, assign, 0)
        p = v1.points
        per_surface_query = p.groupby("surface_replicate_id").primary_query.sum()
        rep[role] = dict(points=len(p), surfaces=p.surface_replicate_id.nunique(), groups=p.prediction_group_id.nunique(),
                         target_drugs=p.target_drug.nunique(), target_drug_cells=len(sup),
                         distinct_doses_per_drug_cell=sup.n_distinct_doses.value_counts().sort_index().to_dict(),
                         readings_at_support_dose=sup.readings_at_support_dose.value_counts().sort_index().to_dict(),
                         visible_points=int(p.visible_flag.sum()), primary_query_points=int(p.primary_query.sum()),
                         surfaces_without_query=int((per_surface_query == 0).sum()),
                         query_points_per_surface=per_surface_query.value_counts().sort_index().to_dict(),
                         excluded_drug_cells=0, view_sha256_k1=v1.sha256(), view_sha256_k0=v0.sha256(),
                         k0_k1_same_queries=bool((v0.points.primary_query == p.primary_query).all()),
                         support_table_sha256=hashlib.sha256(pd.util.hash_pandas_object(sup, index=False).values.tobytes()).hexdigest())
        # Real-data scramble: every target reading except the chosen support reading -> extreme values
        keep = set(sup.support_source_row); g = f.copy(); rng = np.random.default_rng(20261010)
        for arm in ("a", "b"):
            sel = (m.target_arm == arm).to_numpy() & ~m.source_combodrugseq.isin(keep).to_numpy()
            g.loc[sel, f"mono_response_{arm}"] = rng.choice([-50.0, 99.0, 1e6], size=int(sel.sum()))
        for k, ref in ((1, v1), (0, v0)):
            vs, _ = build_role(g, assign, k)
            rep[role][f"scrambled_hidden_view_identical_k{k}"] = vs.sha256() == ref.sha256()
        hidden = (~p.visible_flag).to_numpy()
        rep[role]["hidden_target_references"] = int(hidden.sum())
        p[["source_combodrugseq", "surface_replicate_id", "target_drug", "cell_id", "target_dose", "visible_flag", "primary_query"]] \
            .to_csv(out / f"{role}_query_manifest.csv", index=False)
        sup.to_csv(out / f"{role}_supports.csv", index=False)
    mono = monotherapy_table(frames["train"])
    folds = cluster_folds(assign, frames["train"].surface_replicate_id, k=cfg["oof_folds"])
    rep["training_monotherapy_observations"] = len(mono)
    rep["training_correction_examples"] = len(correction_examples(mono))
    rep["oof_fold_drug_counts"] = pd.Series(folds).value_counts().sort_index().to_dict()
    vdrugs = set(rep_drugs := build_role(frames["validation"], assign, 1)[0].points.target_drug)
    rep["validation_target_drugs_in_training_monotherapy"] = len(vdrugs & set(mono.drug))
    va = with_target(frames["validation"], assign)
    rep["validation_groups_with_both_drugs_heldout"] = int(va.partner_drug.isin(vdrugs).sum())
    clusters = assign[assign.surface_replicate_id.isin(set(va.surface_replicate_id))].heldout_cluster_id.nunique()
    rep["validation_target_clusters"] = int(clusters)
    (out / "task_audit.json").write_text(json.dumps(rep, indent=1, default=int), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=int))


if __name__ == "__main__":
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    main(sys.argv[1], out)

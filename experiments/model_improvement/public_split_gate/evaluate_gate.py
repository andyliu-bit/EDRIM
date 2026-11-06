"""PublicSplitGate-v1 step S5: the single test evaluation (launch card sections 5-7).

  --check   verify every precondition and rebuild/validate everything that needs no test data; exits before
            touching split 3. Safe to run any number of times.
  (default) create S5_TEST_EVALUATED.marker exclusively, then read split 3 once and score every method on the
            same evaluable set: lookup HSA/Bliss, XGBoost residual (5 seeds), BiSurfGNN (chosen form, 5 seeds),
            DeepSynBa reproduction v2. Refuses to run if the marker exists.

Must run under python -I (the released pickle is only opened in isolated mode):
  .venv/Scripts/python.exe -I experiments/model_improvement/public_split_gate/evaluate_gate.py --check
  .venv/Scripts/python.exe -I experiments/model_improvement/public_split_gate/evaluate_gate.py
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import pickle
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_common as gate  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from gate_common import learned, search  # noqa: E402

run_gate = gate.load_module("gate_run_for_eval", HERE / "run_gate.py")
sys.stdout.reconfigure(encoding="utf-8")
RUN, S1 = gate.RUN, gate.S1
OUT, MARKER = RUN / "s5", RUN / "S5_TEST_EVALUATED.marker"
SRC = gate.DATA / "data/raw/external_benchmarks/deepsynba/data"
META, RESP = SRC / "nci_synergy_metadata.csv", SRC / "nci_synergy_dose_response.pickle"
DSB = gate.DATA / "runs/audit_revision/deepsynba_repro_v2_20261012/full"
HASHES = {"metadata": "42b00d74c376d35a5f38c6a63d9a2369f8888d45d37d5b500e334e72ee6ae5b1",
          "responses": "de56596840262950b1b390d40d610789d46af19c2f162d45fa41a7fc23ea5ad1",
          "deepsynba_evaluation": "d73d362f9d18d94cc19dc2764e67ba32817e6390e006f203a20dbdbcc69480bd"}
SEEDS = (41, 42, 43, 44, 45)
DRAWS, BOOT_SEED = 2000, 20261009


def dose_key(x) -> str:
    return f"{float(x):.6g}"


def frozen_npz_hash() -> str:
    text = (RUN / "FROZEN_RECORD.md").read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if "test_predictions.npz" in l and "SHA-256" in l)
    return line.split("SHA-256 `")[1].split("`")[0]


class Lookup:
    """The S1 split-1 single-agent lookup.

    The default constructor reloads S1's CSV outputs (used by --check; CSV round-trips can differ by ~1e-14).
    `from_matrices` recomputes it from the split-1 matrices with S1's exact aggregation order, so the test anchors
    are bitwise identical to what S1 used for training and validation (checked against every S1 validation anchor).
    """

    def __init__(self, exact=None, drug=None, control=None, glob=None):
        if exact is None:
            e = pd.read_csv(S1 / "lookup_split1_exact.csv", dtype={"dose_key": str, "cell": str, "drug": str})
            d = pd.read_csv(S1 / "lookup_split1_drug_dose.csv", dtype={"dose_key": str, "drug": str})
            report = json.loads((S1 / "s1_report.json").read_text(encoding="utf-8"))
            exact = dict(zip(zip(e.cell, e.drug, e.dose_key), e.mean_percent.astype(float)))
            drug = dict(zip(zip(d.drug, d.dose_key), d.mean_percent.astype(float)))
            control, glob = float(report["lookup"]["control_mean_percent"]), float(report["lookup"]["global_mean_percent"])
        self.exact, self.drug, self.control, self.glob = exact, drug, control, glob

    @classmethod
    def from_matrices(cls, train_rows: pd.DataFrame, entries: dict):
        """Same loop, append order and np.mean as s1_prepare.py (evaluable split-1 rows in metadata order)."""
        from collections import defaultdict
        exact, by_drug, control = defaultdict(list), defaultdict(list), []
        for a, b, c, k in zip(train_rows.drug1, train_rows.drug2, train_rows.cell, train_rows.key):
            e = entries[k]; m = np.asarray(e["matrix"], float)
            control.append(m[0, 0])
            for i in range(1, 4):
                exact[(c, a, dose_key(e["x1_dose"][i]))].append(m[i, 0]); by_drug[(a, dose_key(e["x1_dose"][i]))].append(m[i, 0])
                exact[(c, b, dose_key(e["x2_dose"][i]))].append(m[0, i]); by_drug[(b, dose_key(e["x2_dose"][i]))].append(m[0, i])
        return cls({k: float(np.mean(v)) for k, v in exact.items()}, {k: float(np.mean(v)) for k, v in by_drug.items()},
                   float(np.mean(control)), float(np.mean([v for vs in exact.values() for v in vs])))

    def __call__(self, cell, drug, dose) -> float:
        k = (cell, drug, dose_key(dose))
        if k in self.exact:
            return self.exact[k]
        return self.drug.get(k[1:], self.glob)


def locate_runs(form: str) -> dict:
    runs = {"bisurf": {}, "xgb": {}}
    for seed in SEEDS:
        hits = [RUN / f"s3_{form}_r2/bisurf_{form}_seed_41"] if seed == 41 else sorted(RUN.glob(f"s4_bisurf*/bisurf_{form}_seed_{seed}"))
        if len(hits) != 1 or not (hits[0] / "model.pt").exists():
            raise RuntimeError(f"BiSurfGNN {form} seed {seed}: expected exactly one finished run, found {hits}")
        runs["bisurf"][seed] = hits[0]
        x = RUN / f"s4_xgb_r2/xgb_residual_d10_m1_seed_{seed}"
        if not (x / "model.json").exists():
            raise RuntimeError(f"XGBoost seed {seed} has no saved model")
        runs["xgb"][seed] = x
    for path in {p.parent for group in runs.values() for p in group.values()}:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "passed":
            raise RuntimeError(f"Run did not pass its end-of-run checks: {path}")
    return runs


def preconditions():
    bindings = gate.check_bindings()
    for name, path in (("metadata", META), ("responses", RESP), ("deepsynba_evaluation", DSB / "evaluation.json")):
        if gate.sha256(path) != HASHES[name]:
            raise RuntimeError(f"Input hash changed: {name}")
    if gate.sha256(DSB / "test_predictions.npz") != frozen_npz_hash():
        raise RuntimeError("DeepSynBa prediction file differs from the hash recorded in FROZEN_RECORD.md")
    dsb_eval = json.loads((DSB / "evaluation.json").read_text(encoding="utf-8"))
    selection = json.loads((RUN / "s3_selection.json").read_text(encoding="utf-8"))
    form = selection["chosen_form"]
    runs = locate_runs(form)
    cfg = gate.load_config(HERE / "configs/gate.yaml")
    return bindings, dsb_eval, selection, form, runs, cfg


def dev_setup(cfg):
    frames = {r: gate.load_points(r) for r in ("train", "validation")}
    banks = gate.build_banks(frames)
    return frames, banks


def check_lookup(lookup: Lookup, frames, tolerance: float) -> dict:
    """The lookup must reproduce every S1 validation anchor (exactly when rebuilt from the matrices)."""
    va = frames["validation"]
    a = np.array([lookup(c, d, x) for c, d, x in zip(va.deepsynba_cell, va.deepsynba_drug_a, va.dose_a_M)])
    b = np.array([lookup(c, d, x) for c, d, x in zip(va.deepsynba_cell, va.deepsynba_drug_b, va.dose_b_M)])
    diff = float(max(np.abs(a - va.mono_a_percent.to_numpy()).max(), np.abs(b - va.mono_b_percent.to_numpy()).max()))
    if diff > tolerance:
        raise RuntimeError(f"Lookup does not reproduce the S1 validation anchors (max diff {diff} > {tolerance})")
    return {"validation_anchor_points_checked": int(2 * len(va)), "max_abs_difference": diff, "tolerance": tolerance}


def evaluable_rows(meta: pd.DataFrame, split: int, cells: dict, drugs: dict) -> pd.DataFrame:
    part = meta[meta.split == split]
    return part[part.cell.isin(cells) & part.drug1.isin(drugs) & part.drug2.isin(drugs)].reset_index(drop=True)


def load_models(form, runs, banks, cfg):
    models = {}
    for seed, path in runs["bisurf"].items():
        model = gate.build_model(banks, form, seed, cfg["device"])
        model.load_state_dict(torch.load(path / "model.pt", weights_only=True, map_location="cpu"))
        model.eval(); models[seed] = model
    trees = {s: json.loads((p / "metrics.json").read_text(encoding="utf-8"))["best_epoch"] for s, p in runs["xgb"].items()}
    return models, trees


def test_points(meta, entries, lookup, cells, drugs):
    rows, matrices = [], []
    part = meta[(meta.split == 3)].reset_index(drop=True)
    part = part[part.cell.isin(cells) & part.drug1.isin(drugs) & part.drug2.isin(drugs)].reset_index(drop=True)
    for surface, (a, b, c, k) in enumerate(zip(part.drug1, part.drug2, part.cell, part.key)):
        e = entries[k]; m = np.asarray(e["matrix"], float)
        d1, d2 = np.asarray(e["x1_dose"], float), np.asarray(e["x2_dose"], float)
        if m.shape != (4, 4) or d1[0] != 0 or d2[0] != 0:
            raise RuntimeError(f"Unexpected test matrix layout: {k}")
        ma = [lookup(c, a, d1[i]) for i in range(1, 4)]
        mb = [lookup(c, b, d2[j]) for j in range(1, 4)]
        p16 = np.empty((4, 4)); p16[0, 0] = lookup.control; p16[1:, 0] = ma; p16[0, 1:] = mb
        matrices.append((k, m, p16))
        pair = "|".join(sorted((drugs[a], drugs[b])))
        for i in range(1, 4):
            for j in range(1, 4):
                rows.append((k, pair, c, cells[c]["cell_id"], cells[c]["biological_cell_id"], a, b, drugs[a], drugs[b], i, j,
                             d1[i], d2[j], ma[i - 1], mb[j - 1], m[i, j]))
    frame = pd.DataFrame(rows, columns=["matrix_key", "pair_id", "deepsynba_cell", "cell_id", "biological_cell_id",
                                        "deepsynba_drug_a", "deepsynba_drug_b", "drug_a_id", "drug_b_id", "row_i", "col_j",
                                        "dose_a_M", "dose_b_M", "mono_a_percent", "mono_b_percent", "response_percent"])
    frame["dose_a_model"] = np.log10(1.0 + frame.dose_a_M / 1e-12)
    frame["dose_b_model"] = np.log10(1.0 + frame.dose_b_M / 1e-12)
    for arm in ("a", "b"):
        frame[f"mono_response_{arm}"] = (frame[f"mono_{arm}_percent"] / 100.0).clip(0.0, 3.0)
    frame["response"] = frame.response_percent / 100.0
    frame["source_combodrugseq"] = gate.ROLE_OFFSET["test"] + np.arange(len(frame), dtype=np.int64)
    frame["surface_replicate_id"] = frame.matrix_key
    frame["prediction_group_id"] = frame.matrix_key
    frame["heldout_cluster_id"] = frame.pair_id
    frame["response_clipped_high"] = False
    return frame, matrices


def deepsynba_combination(matrices):
    """DeepSynBa v2 predictions on the evaluable matrices, layout-checked against the pickle truth."""
    data = np.load(DSB / "test_predictions.npz", allow_pickle=False)
    keys = ["_".join(map(str, k)) for k in data["keys"]]
    pred = np.asarray(data["predictions"], float).reshape(len(keys), -1)
    truth = np.asarray(data["truth"], float).reshape(len(keys), -1)
    if pred.shape[1] != 16:
        raise RuntimeError(f"DeepSynBa predictions have {pred.shape[1]} entries per matrix")
    index = {k: i for i, k in enumerate(keys)}
    missing = [k for k, _, _ in matrices if k not in index]
    if missing:
        raise RuntimeError(f"{len(missing)} evaluable test matrices lack DeepSynBa predictions")
    rows = np.array([index[k] for k, _, _ in matrices])
    ours = np.stack([m for _, m, _ in matrices])
    direct = np.abs(truth[rows].reshape(-1, 4, 4) - ours).max()
    transposed = np.abs(truth[rows].reshape(-1, 4, 4).transpose(0, 2, 1) - ours).max()
    if direct <= 1e-3:
        p = pred[rows].reshape(-1, 4, 4)
    elif transposed <= 1e-3:
        p = pred[rows].reshape(-1, 4, 4).transpose(0, 2, 1)
    else:
        raise RuntimeError(f"DeepSynBa truth does not match the release (max diff {direct}, transposed {transposed})")
    return p, {"truth_vs_release_max_abs": float(min(direct, transposed)), "layout": "as-is" if direct <= 1e-3 else "transposed",
               "matrices": int(len(rows)), "file_matrices": int(len(keys))}


def pair_sse(frame, pred_percent):
    err = (np.asarray(pred_percent, float) - frame.response_percent.to_numpy()) ** 2
    s = pd.DataFrame({"p": frame.pair_id.to_numpy(), "e": err}).groupby("p", sort=True).e.agg(["sum", "size"])
    return s["sum"].to_numpy(), s["size"].to_numpy(float), s.index.to_numpy()


def evaluate(frame, matrices, preds16, comb):
    """comb: {method: {seed: combination predictions (percent)}}; preds16: {method: {seed: (n,4,4)}}."""
    truth16 = np.stack([m for _, m, _ in matrices])
    rows, sse = [], {}
    pairs = None
    for method, seeds in comb.items():
        for seed, p in seeds.items():
            y = frame.response_percent.to_numpy()
            s, n, idx = pair_sse(frame, p)
            pairs = idx if pairs is None else pairs
            if not (idx == pairs).all():
                raise RuntimeError("Pair index mismatch")
            sse[(method, seed)] = s
            surf = learned.surface_macro_rmse(y / 100.0, np.asarray(p) / 100.0, frame.surface_replicate_id.to_numpy())
            e16 = preds16[method][seed] - truth16
            rows.append(dict(method=method, seed=seed, combination_rmse_percent=float(np.sqrt(np.mean((p - y) ** 2))),
                             combination_pearson=float(np.corrcoef(p, y)[0, 1]),
                             combination_surface_macro_rmse_percent=100.0 * surf,
                             all16_rmse_percent=float(np.sqrt(np.mean(e16 ** 2))),
                             all16_pearson=float(np.corrcoef(preds16[method][seed].ravel(), truth16.ravel())[0, 1])))
    return pd.DataFrame(rows), sse, n, pairs


def bootstrap(sse, n, methods):
    rng = np.random.default_rng(BOOT_SEED)
    P = len(n)
    weights = np.stack([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(DRAWS)]).astype(float)
    denom = weights @ n
    out = {}
    for method in methods:
        seeds = sorted(s for m, s in sse if m == method)
        per_seed = np.stack([np.sqrt((weights @ sse[(method, s)]) / denom) for s in seeds])
        out[method] = per_seed.mean(axis=0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not sys.flags.isolated:
        raise RuntimeError("Run under python -I")
    if MARKER.exists():
        raise RuntimeError("S5 has already been run; the test set is evaluated only once")
    bindings, dsb_eval, selection, form, runs, cfg = preconditions()
    if not dsb_eval.get("reproduced"):
        raise RuntimeError("Card section 7: DeepSynBa v2 failed acceptance, so G1 must use the substitute threshold; not implemented")
    frames, banks = dev_setup(cfg)
    csv_lookup_check = check_lookup(Lookup(), frames, tolerance=1e-9)
    cells_tab = pd.read_csv(S1 / "cell_mapping_used.csv")
    drugs_tab = pd.read_csv(S1 / "drug_mapping_used.csv")
    cells = {r.deepsynba_cell: {"cell_id": r.cell_id, "biological_cell_id": r.biological_cell_id}
             for r in cells_tab.itertuples() if r.status == "mapped"}
    drugs = {r.deepsynba_drug: r.canonical_drug_id for r in drugs_tab.itertuples() if bool(r.mapped)}
    with threadpool_limits(limits=cfg["xgboost"]["threads"]):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        models, trees = load_models(form, runs, banks, cfg)
        if args.check:
            print(json.dumps({"check": "passed", "form": form, "bisurf_runs": {s: str(p) for s, p in runs["bisurf"].items()},
                              "xgb_trees": trees, "csv_lookup_check": csv_lookup_check, "bindings": bindings}, indent=1, default=str))
            return
        OUT.mkdir()
        with MARKER.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "form": form}) + "\n")
        meta = pd.read_csv(META)
        meta["key"] = meta.drug1 + "_" + meta.drug2 + "_" + meta.cell
        train_rows = evaluable_rows(meta, 1, cells, drugs)
        data = pickle.load(RESP.open("rb"))
        train_entries = {k: data[k] for k in train_rows.key}
        entries = {k: data[k] for k in meta.key[meta.split == 3]}
        del data
        lookup = Lookup.from_matrices(train_rows, train_entries)
        del train_entries
        lookup_check = check_lookup(lookup, frames, tolerance=0.0)
        frame, matrices = test_points(meta, entries, lookup, cells, drugs)
        table = learned.build_compact_point_table(frame, banks, role="test", fold=0, monotherapy_upper_bound=3.0)
        hsa = np.minimum(frame.mono_a_percent, frame.mono_b_percent).to_numpy()
        bliss = (frame.mono_a_percent * frame.mono_b_percent / 100.0).to_numpy()
        comb = {"lookup_hsa": {0: hsa}, "lookup_bliss": {0: bliss}, "xgb": {}, "bisurf": {}}
        for seed, model in models.items():
            comb["bisurf"][seed] = 100.0 * np.asarray(search.predict(model, table, banks, cfg), float)
        import xgboost as xgb
        build = run_gate.xgb_inputs(frames, banks, gate.morgan(banks), cfg)
        dtest = (xgb.DMatrix(build(frame, False)), xgb.DMatrix(build(frame, True)))
        hsa_ratio = np.minimum(frame.mono_response_a, frame.mono_response_b).to_numpy(float)
        for seed, path in runs["xgb"].items():
            booster = xgb.Booster(); booster.load_model(path / "model.json")
            k = trees[seed]
            delta = 0.5 * (booster.predict(dtest[0], iteration_range=(0, k)) + booster.predict(dtest[1], iteration_range=(0, k))) if k else 0.0
            comb["xgb"][seed] = 100.0 * (hsa_ratio + delta)
        dsb16, dsb_check = deepsynba_combination(matrices)
        comb["deepsynba_v2"] = {10: np.concatenate([dsb16[s, 1:, 1:].ravel() for s in range(len(matrices))])}
        lookup16 = np.stack([p16 for _, _, p16 in matrices])
        preds16 = {}
        for method, seeds in comb.items():
            preds16[method] = {}
            for seed, p in seeds.items():
                if method == "deepsynba_v2":
                    preds16[method][seed] = dsb16
                else:
                    full = lookup16.copy(); full[:, 1:, 1:] = np.asarray(p).reshape(-1, 3, 3); preds16[method][seed] = full
        table_rows, sse, n, pairs = evaluate(frame, matrices, preds16, comb)
        table_rows.to_csv(OUT / "per_seed_metrics.csv", index=False)
        boots = bootstrap(sse, n, list(comb))
        point = {m: float(table_rows[table_rows.method == m].combination_rmse_percent.mean()) for m in comb}
        rel = lambda a, b: (point[a] - point[b]) / point[b]
        rel_draws = lambda a, b: (boots[a] - boots[b]) / boots[b]
        def interval(a, b):
            d = rel_draws(a, b); return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        g1_rel, g1_ci = rel("bisurf", "deepsynba_v2"), interval("bisurf", "deepsynba_v2")
        g2_rel, g2_ci = rel("bisurf", "xgb"), interval("bisurf", "xgb")
        g1, g2 = g1_rel <= 0.0, (g2_rel <= -0.05 and g2_ci[1] < 0.0)
        comparisons = {f"{a}_vs_{b}": {"relative_change": rel(a, b), "ci95": interval(a, b)}
                       for a, b in [("bisurf", "deepsynba_v2"), ("bisurf", "xgb"), ("bisurf", "lookup_hsa"), ("xgb", "lookup_hsa"),
                                    ("deepsynba_v2", "lookup_hsa"), ("xgb", "deepsynba_v2"), ("lookup_bliss", "lookup_hsa")]}
        for method, seeds in comb.items():
            np.savez_compressed(OUT / f"test_combination_predictions_{method}.npz", **{f"seed_{s}": np.asarray(p, float) for s, p in seeds.items()},
                                point_ids=frame.source_combodrugseq.to_numpy(), matrix_keys=frame.matrix_key.to_numpy().astype(str))
        result = {"protocol": "PublicSplitGate-v1", "step": "S5", "form": form, "selection": selection,
                  "evaluable": {"matrices": len(matrices), "combination_points": int(len(frame)), "pairs": int(len(pairs))},
                  "point_estimates_combination_rmse_percent": point,
                  "seed_sd_combination_rmse_percent": {m: float(table_rows[table_rows.method == m].combination_rmse_percent.std(ddof=1))
                                                       if (table_rows.method == m).sum() > 1 else None for m in comb},
                  "all16_rmse_percent": {m: float(table_rows[table_rows.method == m].all16_rmse_percent.mean()) for m in comb},
                  "comparisons": comparisons,
                  "gates": {"G1": {"rule": "BiSurfGNN combination RMSE <= DeepSynBa v2 (point estimate)", "relative_change": g1_rel,
                                   "ci95": g1_ci, "passed": bool(g1)},
                            "G2": {"rule": "BiSurfGNN vs XGBoost <= -5% and CI upper < 0", "relative_change": g2_rel,
                                   "ci95": g2_ci, "passed": bool(g2)},
                            "overall": "pass" if (g1 and g2) else "fail"},
                  "deepsynba_check": dsb_check, "deepsynba_reported_full_test": dsb_eval, "lookup_check": lookup_check,
                  "xgb_trees": trees, "bootstrap": {"unit": "test drug pair", "draws": DRAWS, "seed": BOOT_SEED},
                  "bindings": bindings, "test_response_reads": 1}
        with (OUT / "evaluation.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, default=float)
        print(json.dumps({k: result[k] for k in ("evaluable", "point_estimates_combination_rmse_percent", "all16_rmse_percent", "gates")},
                         indent=1, default=float))


if __name__ == "__main__":
    main()

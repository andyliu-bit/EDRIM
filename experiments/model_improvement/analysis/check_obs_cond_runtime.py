"""ObsCond-v1 runtime checks on the real development data with a trained checkpoint.

Usage: python check_obs_cond_runtime.py <job_dir> <config.yaml> <out.json>
<job_dir> = .../OC_<kind>_<r>_k1_seed_<s>_primary (contains model.pt, metrics.json, predictions.csv).
Priors are recomputed with the job's seed and must reproduce the recorded prior hash.
Validation hidden truth is touched only to scramble it (check 1) — never as an input.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
import yaml

from bisurfgnn.data.batch import GraphItem, pad_graph_items
from bisurfgnn.sparse_mono.task import build_role, choose_supports, with_target

spec = importlib.util.spec_from_file_location("oc_runner", ROOT / "experiments/model_improvement/run_obs_cond.py")
oc = importlib.util.module_from_spec(spec); spec.loader.exec_module(oc)


def main():
    job, cfg_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m = json.loads((job / "metrics.json").read_text(encoding="utf-8"))
    _, kind, key = m["pipeline"].split("_"); seed = m["seed"]; c = cfg["obs"][key]; dev = torch.device(cfg["device"])
    ctx = oc.Context(cfg)
    pri, _ = oc.stage1(ctx, kind, c, seed, dev)
    res = dict(job=str(job), prior_hash_reproduced=hashlib.sha256(np.round(pri["validation"]["tq"], 8).tobytes()).hexdigest() == m["prior_validation_hash"])
    model = oc.build_model(ctx, kind, c, None, (0.0, 1.0)).to(dev)
    model.load_state_dict(torch.load(job / "model.pt", weights_only=True, map_location="cpu")); model.eval()
    banks_in = oc.banks_for(ctx, kind, dev)
    vv = ctx.views[("validation", 1)]
    a, b, cell, is_a = oc.arm_inputs(ctx, vv, pri["validation"])
    base = oc.predict(model, banks_in, cell, a, b, dev)
    saved = pd.read_csv(job / "predictions.csv").prediction.to_numpy()
    res["saved_vs_reloaded_max_abs"] = float(np.max(np.abs(saved - base["pred"])))
    res["input_hash_matches_job"] = oc.input_hash(a, b, cell) == m["validation_input_sha256"]

    # 1 hidden target monotherapy scrambled in the raw validation frame -> identical view, inputs, predictions
    frames, _, _, _ = oc.sm.fitgen.frozen_tables(cfg)
    fr = frames["validation"].copy()
    mt = with_target(fr, ctx.assign); keep = set(choose_supports(mt).support_source_row)
    rng = np.random.default_rng(0); hide = ~fr.source_combodrugseq.isin(keep).to_numpy()
    ta = (mt.target_arm == "a").to_numpy()
    for arm, sel in (("a", ta), ("b", ~ta)):
        col = f"mono_response_{arm}"; rows = hide & sel
        fr.loc[rows, col] = rng.uniform(0, 2, rows.sum())
    v2, _ = build_role(fr, ctx.assign, 1)
    a2, b2, c2, _ = oc.arm_inputs(ctx, v2, pri["validation"])
    p2 = oc.predict(model, banks_in, c2, a2, b2, dev)["pred"]
    res["hidden_scramble"] = dict(rows_scrambled=int(hide.sum()), view_sha_equal=v2.sha256() == vv.sha256(),
                                  input_hash_equal=oc.input_hash(a2, b2, c2) == oc.input_hash(a, b, cell),
                                  pred_max_abs_diff=float(np.max(np.abs(p2 - base["pred"]))))
    del frames, fr
    q = vv.points.primary_query.to_numpy()

    # 2 permitted support responses changed -> representation and prediction change
    pa, pb = copy.copy(a), copy.copy(b)
    for arm, tgt in ((pa, is_a), (pb, ~is_a)):
        arm.sup_value = arm.sup_value + 0.1 * tgt[:, None].float()
        arm.mono = arm.mono + 0.1 * (tgt & (arm.flag > 0.5)).float()
    pp = oc.predict(model, banks_in, cell, pa, pb, dev)
    res["support_plus_0p1"] = dict(frac_query_pred_changed=float(np.mean(np.abs(pp["pred"] - base["pred"])[q] > 1e-6)),
                                   mean_abs_pred_change=float(np.mean(np.abs(pp["pred"] - base["pred"])[q])),
                                   mean_abs_target_mono_change=float(np.mean(np.abs(oc.target_of(pp, is_a) - oc.target_of(base, is_a))[q])))

    # 3 k=0 view: target update is exactly zero
    a0, b0, c0, ia0 = oc.arm_inputs(ctx, ctx.views[("validation", 0)], pri["validation"])
    p0 = oc.predict(model, banks_in, c0, a0, b0, dev)
    res["k0_target_relative_update_max"] = float(np.max(oc.target_of(p0, ia0, "dz")))
    res["k1_target_relative_update_mean"] = float(np.mean(oc.target_of(base, is_a, "dz")))

    # 4 drug exchange
    ps = oc.predict(model, banks_in, cell, b, a, dev)["pred"]
    res["swap_max_abs_diff"] = float(np.max(np.abs(ps - base["pred"])))

    # 5 support-independent bank cache vs per-chunk recompute; distinct supports never share an update
    sel = np.random.default_rng(1).choice(len(cell), 64, replace=False)
    pc = np.concatenate([oc.predict(model, banks_in, cell[[i]], a.index([i]), b.index([i]), dev)["pred"] for i in sel])
    res["cache_vs_recompute_max_abs"] = float(np.max(np.abs(pc - base["pred"][sel])))
    with torch.no_grad():
        banks = model.encode_banks(*banks_in)
        t = a.index(slice(0, 2)); t = copy.copy(t)
        t.drug = torch.full_like(t.drug, int(t.drug[0])); t.dose = torch.full_like(t.dose, float(t.dose[0])); t.prior = torch.full_like(t.prior, float(t.prior[0]))
        t.sup_mask = torch.ones_like(t.sup_mask); t.sup_value = torch.tensor([[0.4], [0.9]]); t.sup_dose = torch.full_like(t.sup_dose, 4.0)
        st, _ = model.arm_state(banks, torch.zeros(2, dtype=torch.long, device=dev) + cell[0].to(dev), t.to(dev))
        res["same_drug_cell_different_support_update_diff"] = float((st.z_updated[0] - st.z_updated[1]).abs().max())

    # 6 atom renumbering of every real drug graph
    if kind == "gnn":
        g = torch.Generator().manual_seed(5); perm = []
        for it in ctx.banks.drug_graphs:
            p = torch.randperm(it.node_features.shape[0], generator=g); perm.append(GraphItem(it.node_features[p], it.adjacency[p][:, p]))
        pr = oc.predict(model, (pad_graph_items(perm).to(dev), banks_in[1]), cell, a, b, dev)["pred"]
        res["atom_permutation_max_abs_diff"] = float(np.max(np.abs(pr - base["pred"])))

    # 7 combination loss reaches the molecular encoder (trained weights, real training batch)
    tv = ctx.views[("train", 1)]
    at, bt, ct, _ = oc.arm_inputs(ctx, tv, pri["train"]); idx = torch.arange(512)
    model.train()
    o = model(model.encode_banks(*banks_in), ct[idx].to(dev), at.index(idx).to(dev), bt.index(idx).to(dev))
    y = torch.as_tensor(tv.points.response.to_numpy()[:512], dtype=torch.float32, device=dev)
    gr = torch.autograd.grad(((o.prediction - y) ** 2).mean(), list(model.drug_encoder.parameters()), allow_unused=True)
    res["combo_grad_norm_drug_encoder"] = float(torch.sqrt(sum((x.double() ** 2).sum() for x in gr if x is not None)))
    res["combo_grad_all_finite"] = all(bool(torch.isfinite(x).all()) for x in gr if x is not None)
    model.eval()

    # 8 range / non-trivial output
    pred = base["pred"]
    res["prediction_range"] = [float(pred.min()), float(pred.max())]
    res["prediction_finite"] = bool(np.isfinite(pred).all())
    ref = np.minimum(np.where(a.flag.numpy() > 0.5, a.mono.numpy(), np.clip(base["mono_a"], 0, 3)),
                     np.where(b.flag.numpy() > 0.5, b.mono.numpy(), np.clip(base["mono_b"], 0, 3)))
    res["frac_pred_differs_from_reference"] = float(np.mean(np.abs(pred - ref) > 1e-6))
    checks = dict(prior=res["prior_hash_reproduced"], reload=res["saved_vs_reloaded_max_abs"] < 1e-6, input=res["input_hash_matches_job"],
                  scramble=res["hidden_scramble"]["view_sha_equal"] and res["hidden_scramble"]["input_hash_equal"] and res["hidden_scramble"]["pred_max_abs_diff"] == 0.0,
                  support=res["support_plus_0p1"]["frac_query_pred_changed"] > 0.5, k0=res["k0_target_relative_update_max"] == 0.0,
                  swap=res["swap_max_abs_diff"] < 1e-5, cache=res["cache_vs_recompute_max_abs"] < 1e-5,
                  distinct=res["same_drug_cell_different_support_update_diff"] > 1e-6,
                  atoms=res.get("atom_permutation_max_abs_diff", 0.0) < 1e-4,
                  grad=res["combo_grad_norm_drug_encoder"] > 0 and res["combo_grad_all_finite"], finite=res["prediction_finite"],
                  nontrivial=res["frac_pred_differs_from_reference"] > 0.5)
    res["checks"] = checks; res["all_passed"] = all(checks.values())
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    if not res["all_passed"]:
        raise SystemExit("runtime checks failed")


if __name__ == "__main__":
    main()

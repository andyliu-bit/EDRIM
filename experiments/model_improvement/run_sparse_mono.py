"""SparseMono-v1 runner. Protocol: docs/SPARSE_MONO_V1_PROTOCOL.md.

Pipelines (stage 1 = target monotherapy reconstruction, stage 2 = combination head)
  A_ref      training dose-bin x cell mean prior; k=1 additive support offset; combination = estimated HSA
  B_xgb      XGBoost prior [drug PCA16, cell PCA16, dose]; XGBoost correction; XGBoost combination head
  C_tabm     stage 1 identical to B; FeatureTabM combination head (surface-RMSE objective)
  D_gnn_<g>  MonoPriorNet over molecular graphs; CorrectionMLP; FeatureTabM head
  D_morgan_<g> same as D_gnn with a Morgan-512 MLP drug encoder
Training combinations always receive OUT-OF-FOLD reconstructions (4 folds over training
chemical clusters); validation receives the full-training-data reconstruction.
Models only ever see ``RoleView`` (no hidden target readings).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import rtdl_num_embeddings as rne
import torch
import yaml

from bisurfgnn.models.sparse_mono import CorrectionMLP, FeatureTabM, MonoPriorNet
from bisurfgnn.sparse_mono.task import build_role, cluster_folds, correction_examples, monotherapy_table

spec = importlib.util.spec_from_file_location("sm_fitgen", ROOT / "experiments/model_improvement/analysis/diagnose_fit_vs_generalization.py")
fitgen = importlib.util.module_from_spec(spec); spec.loader.exec_module(fitgen)
pilot, graph, learned = fitgen.pilot, fitgen.graph, fitgen.learned
MONO_MAX = 3.0
SWAP = [1, 0, 3, 2, 5, 4, 6, 7, 8] + list(range(25, 41)) + list(range(9, 25)) + list(range(41, 57))


# ------------------------------------------------------------------ data
class Context:
    def __init__(self, cfg):
        self.cfg = cfg
        frames, banks, _, fingerprints = fitgen.frozen_tables(cfg)
        data_root = (ROOT / cfg["data_root"]).resolve()
        from bisurfgnn.benchmark_config import load_benchmark_config
        bench = load_benchmark_config(data_root / cfg["benchmark_config"])
        self.assign = pd.read_parquet(bench.output_dir / "surface_assignments.parquet")
        self.banks = banks
        self.drug_index = {d: i for i, d in enumerate(banks.drug_ids)}
        self.cell_index = {c: i for i, c in enumerate(banks.cell_ids)}
        tr = frames["train"]
        # Training role: full monotherapy allowed. Validation frame is dropped after the views are built.
        self.mono = monotherapy_table(tr)
        self.examples = correction_examples(self.mono)
        self.folds = cluster_folds(self.assign, tr.surface_replicate_id, k=cfg["oof_folds"])
        self.views, self.truth = {}, {}
        for k in (0, 1):
            for role, frame in frames.items():
                self.views[(role, k)], self.truth[(role, k)] = build_role(frame, self.assign, k)
        del frames
        from sklearn.decomposition import PCA
        fp = fingerprints.numpy().astype(float)
        tr_drugs = sorted(set(self.mono.drug))
        self.drug_codes = PCA(16, random_state=0).fit(fp[[self.drug_index[d] for d in tr_drugs]]).transform(fp)
        cell_vec = np.stack([g.node_features[:, 0].numpy() for g in banks.cell_graphs]).astype(float)
        tr_cells = sorted(set(self.mono.cell_id))
        self.cell_codes = PCA(16, random_state=0).fit(cell_vec[[self.cell_index[c] for c in tr_cells]]).transform(cell_vec)
        self.fingerprints = fingerprints.float()
        tv = self.views[("train", 1)].points
        self.residual_scale = float((tv.response - np.minimum(tv.partner_mono, self.truth[("train", 1)].target_true_mono)).std())


# ------------------------------------------------------------------ stage 1 learners
class BinMeanPrior:
    """A_ref: mean training monotherapy per (cell, dose rounded to 0.5), falling back to dose bin, then global."""
    def fit(self, mono, ctx, seed):
        b = (mono.dose * 2).round() / 2
        self.cell_bin = mono.assign(b=b).groupby(["cell_id", "b"]).mono.mean().to_dict()
        self.bin = mono.assign(b=b).groupby("b").mono.mean().to_dict(); self.glob = float(mono.mono.mean()); return self
    def predict(self, drugs, cells, doses):
        b = np.round(np.asarray(doses) * 2) / 2
        return np.array([self.cell_bin.get((c, x), self.bin.get(x, self.glob)) for c, x in zip(cells, b)])


class XGBPrior:
    def fit(self, mono, ctx, seed):
        import xgboost as xgb
        self.ctx, c = ctx, ctx.cfg["xgb_prior"]
        self.model = xgb.train(dict(objective="reg:squarederror", eta=c["eta"], max_depth=c["depth"], subsample=0.8,
                                    colsample_bytree=0.8, seed=seed, nthread=c["threads"], tree_method="hist"),
                               xgb.DMatrix(self.x(mono.drug, mono.cell_id, mono.dose), label=mono.mono.to_numpy()), c["trees"])
        return self
    def x(self, drugs, cells, doses):
        return np.column_stack([self.ctx.drug_codes[[self.ctx.drug_index[d] for d in drugs]],
                                self.ctx.cell_codes[[self.ctx.cell_index[c] for c in cells]], np.asarray(doses, float)])
    def predict(self, drugs, cells, doses):
        import xgboost as xgb
        return self.model.predict(xgb.DMatrix(self.x(drugs, cells, doses)))


class NetPrior:
    def __init__(self, drug_input, cfg_key):
        self.drug_input, self.cfg_key = drug_input, cfg_key
    def fit(self, mono, ctx, seed):
        c = ctx.cfg["net_prior"][self.cfg_key]; dev = torch.device(ctx.cfg["device"]); self.ctx, self.dev = ctx, dev
        learned._seed_everything(seed)
        self.model = MonoPriorNet(self.drug_input, ctx.banks.atom_feature_dim, ctx.banks.cell_feature_dim, hidden=c["hidden"],
                                  dropout=c["dropout"], dose_mean=float(mono.dose.mean()), dose_sd=float(mono.dose.std()),
                                  mono_mean=float(mono.mono.mean())).to(dev)
        self.drug_in = (graph.pad_graph_items(ctx.banks.drug_graphs).to(dev) if self.drug_input == "gnn" else ctx.fingerprints.to(dev))
        self.cell_in = graph.pad_graph_items(ctx.banks.cell_graphs).to(dev)
        d = torch.as_tensor([ctx.drug_index[x] for x in mono.drug], device=dev)
        cc = torch.as_tensor([ctx.cell_index[x] for x in mono.cell_id], device=dev)
        z = torch.as_tensor(mono.dose.to_numpy(), dtype=torch.float32, device=dev)
        y = torch.as_tensor(mono.mono.to_numpy(), dtype=torch.float32, device=dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        g = torch.Generator().manual_seed(seed)
        for step in range(c["steps"]):
            self.model.train()
            i = torch.randint(0, len(y), (c["batch"],), generator=g).to(dev)
            drugs, cells = self.model.encode(self.drug_in, self.cell_in)
            loss = torch.mean((self.model.predict(drugs, cells, d[i], cc[i], z[i]) - y[i]) ** 2)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite prior loss")
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step()
        self.final_train_mse = float(loss.detach())
        return self
    @torch.no_grad()
    def predict(self, drugs, cells, doses):
        self.model.eval()
        dr, ce = self.model.encode(self.drug_in, self.cell_in)
        d = torch.as_tensor([self.ctx.drug_index[x] for x in drugs], device=self.dev)
        c = torch.as_tensor([self.ctx.cell_index[x] for x in cells], device=self.dev)
        z = torch.as_tensor(np.asarray(doses, float), dtype=torch.float32, device=self.dev)
        return self.model.predict(dr, ce, d, c, z).cpu().numpy()


class AdditiveCorrection:
    def fit(self, ex, ctx, seed): return self
    def predict(self, pq, ps, sv, dq, ds): return np.asarray(sv) - np.asarray(ps)


class XGBCorrection:
    def fit(self, ex, ctx, seed):
        import xgboost as xgb
        c = ctx.cfg["xgb_correction"]
        self.m = xgb.train(dict(objective="reg:squarederror", eta=c["eta"], max_depth=c["depth"], subsample=0.8, seed=seed,
                                nthread=c["threads"], tree_method="hist"),
                           xgb.DMatrix(self.x(ex.prior_q, ex.prior_s, ex.support_value, ex.dose, ex.support_dose),
                                       label=(ex.mono - ex.prior_q).to_numpy()), c["trees"])
        return self
    @staticmethod
    def x(pq, ps, sv, dq, ds):
        pq, ps, sv, dq, ds = (np.asarray(v, float) for v in (pq, ps, sv, dq, ds))
        return np.column_stack([sv - ps, dq - ds, pq, ps])
    def predict(self, pq, ps, sv, dq, ds):
        import xgboost as xgb
        return self.m.predict(xgb.DMatrix(self.x(pq, ps, sv, dq, ds)))


class MLPCorrection:
    def fit(self, ex, ctx, seed):
        c = ctx.cfg["mlp_correction"]; learned._seed_everything(seed)
        self.m = CorrectionMLP(c["hidden"], c["dropout"])
        t = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float32)
        args = [t(ex.prior_q), t(ex.prior_s), t(ex.support_value), t(ex.dose), t(ex.support_dose)]
        y = t(ex.mono - ex.prior_q)
        opt = torch.optim.AdamW(self.m.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        g = torch.Generator().manual_seed(seed)
        for _ in range(c["steps"]):
            self.m.train(); i = torch.randint(0, len(y), (c["batch"],), generator=g)
            loss = torch.mean((self.m(*[a[i] for a in args]) - y[i]) ** 2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return self
    @torch.no_grad()
    def predict(self, pq, ps, sv, dq, ds):
        self.m.eval(); t = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float32)
        return self.m(t(pq), t(ps), t(sv), t(dq), t(ds)).numpy()


def stage1(ctx, prior_factory, corr_factory, k, seed):
    """Reconstructed target monotherapy for every train (out-of-fold) and validation point."""
    ex = ctx.examples.copy()
    ex["fold"] = ex.drug.map(ctx.folds)
    tv = ctx.views[("train", k)].points.copy(); vv = ctx.views[("validation", k)].points.copy()
    tv["fold"] = tv.target_drug.map(ctx.folds)
    if ex.fold.isna().any() or tv.fold.isna().any():
        raise RuntimeError("Training drug without a cluster fold")
    ex["prior_q"] = np.nan; ex["prior_s"] = np.nan; tv["prior_q"] = np.nan; tv["prior_s"] = np.nan
    nf = ctx.cfg["oof_folds"]
    for f in range(nf):  # pass 1: out-of-fold priors
        prior = prior_factory().fit(ctx.mono[ctx.mono.drug.map(ctx.folds) != f], ctx, seed + f)
        for tab, drug in ((ex, "drug"), (tv, "target_drug")):
            sel = tab.fold == f
            dose_col = "dose" if tab is ex else "target_dose"
            tab.loc[sel, "prior_q"] = prior.predict(tab.loc[sel, drug], tab.loc[sel, "cell_id"], tab.loc[sel, dose_col])
            if k == 1:
                tab.loc[sel, "prior_s"] = prior.predict(tab.loc[sel, drug], tab.loc[sel, "cell_id"], tab.loc[sel, "support_dose"])
    full_prior = prior_factory().fit(ctx.mono, ctx, seed + 97)
    vv["prior_q"] = full_prior.predict(vv.target_drug, vv.cell_id, vv.target_dose)
    recon = {}
    if k == 1:
        vv["prior_s"] = full_prior.predict(vv.target_drug, vv.cell_id, vv.support_dose)
        tv["correction"] = np.nan
        for f in range(nf):  # pass 2: corrections fitted on other folds' out-of-fold priors
            corr = corr_factory().fit(ex[ex.fold != f], ctx, seed + 11 * (f + 1))
            sel = tv.fold == f
            tv.loc[sel, "correction"] = corr.predict(tv.loc[sel, "prior_q"], tv.loc[sel, "prior_s"], tv.loc[sel, "support_value"],
                                               tv.loc[sel, "target_dose"], tv.loc[sel, "support_dose"])
        full_corr = corr_factory().fit(ex, ctx, seed + 131)
        vv["correction"] = full_corr.predict(vv.prior_q, vv.prior_s, vv.support_value, vv.target_dose, vv.support_dose)
        for name, tab in (("train", tv), ("validation", vv)):
            est = np.clip(tab["prior_q"] + tab["correction"], 0.0, MONO_MAX)
            recon[name] = np.where(tab.visible_flag, tab.visible_value, est)
    else:
        for name, tab in (("train", tv), ("validation", vv)):
            recon[name] = np.clip(tab.prior_q.to_numpy(), 0.0, MONO_MAX)
    if any(np.isnan(r).any() for r in recon.values()):
        raise RuntimeError("Missing reconstruction")
    return recon, {"prior_validation_hash": hashlib.sha256(np.round(vv.prior_q.to_numpy(), 8).tobytes()).hexdigest()}


# ------------------------------------------------------------------ stage 2
def features(ctx, view, recon):
    p = view.points
    is_a = (p.target_arm == "a").to_numpy()
    t_flag = p.visible_flag.to_numpy().astype(float)
    m_a = np.where(is_a, recon, p.partner_mono); m_b = np.where(is_a, p.partner_mono, recon)
    f_a = np.where(is_a, t_flag, 1.0); f_b = np.where(is_a, 1.0, t_flag)
    d_a = np.where(is_a, p.target_dose, p.partner_dose); d_b = np.where(is_a, p.partner_dose, p.target_dose)
    drug_a = np.where(is_a, p.target_drug, p.partner_drug); drug_b = np.where(is_a, p.partner_drug, p.target_drug)
    da = ctx.drug_codes[[ctx.drug_index[d] for d in drug_a]]; db = ctx.drug_codes[[ctx.drug_index[d] for d in drug_b]]
    cc = ctx.cell_codes[[ctx.cell_index[c] for c in p.cell_id]]
    x = np.column_stack([m_a, m_b, f_a, f_b, d_a, d_b, np.minimum(m_a, m_b), m_a * m_b, np.abs(m_a - m_b), da, db, cc])
    if x.shape[1] != 57 or not np.isfinite(x).all():
        raise RuntimeError("Invalid combination feature matrix")
    return x.astype(np.float64)


def macro_rmse(y, pred, surfaces, mask):
    f = pd.DataFrame({"s": surfaces[mask], "e": (np.asarray(pred)[mask] - np.asarray(y)[mask]) ** 2})
    return float(np.sqrt(f.groupby("s").e.mean()).mean())


def head_hsa(ctx, xt, xv, tv, vv, seed, out):
    return np.minimum(xv[:, 0], xv[:, 1]), {"head": "estimated_hsa"}


def head_xgb(ctx, xt, xv, tv, vv, seed, out):
    import xgboost as xgb
    c = ctx.cfg["xgb_head"]
    y = tv.points.response.to_numpy() - np.minimum(xt[:, 0], xt[:, 1])
    w = 1.0 / tv.points.groupby("surface_replicate_id").surface_replicate_id.transform("size").to_numpy()
    b = xgb.train(dict(objective="reg:squarederror", eta=c["eta"], max_depth=c["depth"], min_child_weight=c["min_child"],
                       subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=seed, nthread=c["threads"]),
                  xgb.DMatrix(np.vstack([xt, xt[:, SWAP]]), label=np.concatenate([y, y]), weight=np.concatenate([w, w])), c["max_trees"])
    dv = (xgb.DMatrix(xv), xgb.DMatrix(xv[:, SWAP])); hsa = np.minimum(xv[:, 0], xv[:, 1])
    pred_at = lambda n: hsa if n == 0 else hsa + 0.5 * (b.predict(dv[0], iteration_range=(0, n)) + b.predict(dv[1], iteration_range=(0, n)))
    yv, s, q = vv.points.response.to_numpy(), vv.points.surface_replicate_id.to_numpy(), vv.points.primary_query.to_numpy()
    hist, best = [], (macro_rmse(yv, hsa, s, q), 0)
    for n in range(c["eval_every"], c["max_trees"] + 1, c["eval_every"]):
        sc = macro_rmse(yv, pred_at(n), s, q); hist.append(dict(trees=n, val_primary=sc))
        if sc < best[0] - 1e-5:
            best = (sc, n)
    b.save_model(out / "combo_head.json")
    return pred_at(best[1]), {"head": "xgb", "trees": best[1], "history": hist}


def head_tabm(ctx, xt, xv, tv, vv, seed, out):
    c = ctx.cfg["tabm_head"]; dev = torch.device(ctx.cfg["device"]); learned._seed_everything(seed)
    bins = rne.compute_bins(torch.as_tensor(xt), n_bins=16)
    model = FeatureTabM([b.float() for b in bins], SWAP, ctx.residual_scale, d_embedding=16, k=c["k"],
                        width=c["width"], blocks=c["blocks"], dropout=c["dropout"]).to(dev)
    X = torch.as_tensor(xt, dtype=torch.float32, device=dev); XV = torch.as_tensor(xv, dtype=torch.float32, device=dev)
    y = torch.as_tensor(tv.points.response.to_numpy(), dtype=torch.float32, device=dev)
    surf = tv.points.surface_replicate_id.to_numpy()
    groups = pd.Series(np.arange(len(surf))).groupby(surf).apply(list)
    keys = list(groups.index); eps = (0.01 * ctx.residual_scale) ** 2
    yv, s, q = vv.points.response.to_numpy(), vv.points.surface_replicate_id.to_numpy(), vv.points.primary_query.to_numpy()
    @torch.no_grad()
    def predict(model):
        model.eval(); return torch.cat([model.predict(XV[i:i + 4096]) for i in range(0, len(XV), 4096)]).cpu().numpy()
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    steps = c["steps"]; sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + np.cos(np.pi * min(t, steps) / steps)))
    best = dict(score=macro_rmse(yv, predict(model), s, q), state=copy.deepcopy(model.state_dict()), step=0); hist = []
    g = np.random.default_rng(seed); step = 0
    while step < steps:
        order = g.permutation(len(keys))
        for start in range(0, len(order), c["batch_surfaces"]):
            if step >= steps:
                break
            sel = [groups[keys[j]] for j in order[start:start + c["batch_surfaces"]]]
            idx = torch.as_tensor(np.concatenate(sel), device=dev)
            codes = torch.as_tensor(np.repeat(np.arange(len(sel)), [len(x) for x in sel]), device=dev)
            onehot = torch.zeros(len(sel), len(idx), device=dev); onehot[codes, torch.arange(len(idx), device=dev)] = 1.0
            model.train(); pred = model.predict(X[idx], members=True)
            mse_s = onehot @ (pred - y[idx, None]) ** 2 / onehot.sum(1, keepdim=True)    # [S, k]
            loss = torch.sqrt(mse_s + eps).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite head loss")
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % c["eval_every"] == 0 or step == steps:
                sc = macro_rmse(yv, predict(model), s, q); hist.append(dict(step=step, val_primary=sc))
                if sc < best["score"] - 1e-5:
                    best.update(score=sc, state=copy.deepcopy(model.state_dict()), step=step)
    model.load_state_dict(best["state"]); pred = predict(model)
    torch.save(best["state"], out / "combo_head.pt")
    clone = FeatureTabM([b.float() for b in bins], SWAP, ctx.residual_scale, d_embedding=16, k=c["k"], width=c["width"],
                        blocks=c["blocks"], dropout=c["dropout"]).to(dev)
    clone.load_state_dict(torch.load(out / "combo_head.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(predict(clone), pred, atol=0, rtol=0)
    return pred, {"head": "tabm", "selected_step": best["step"], "history": hist}


PIPELINES = {
    "A_ref": (BinMeanPrior, AdditiveCorrection, head_hsa),
    "B_xgb": (XGBPrior, XGBCorrection, head_xgb),
    "C_tabm": (XGBPrior, XGBCorrection, head_tabm),
}
for g in ("g1", "g2"):
    PIPELINES[f"D_gnn_{g}"] = (lambda g=g: NetPrior("gnn", g), MLPCorrection, head_tabm)
    PIPELINES[f"D_morgan_{g}"] = (lambda g=g: NetPrior("morgan", g), MLPCorrection, head_tabm)


def run_job(ctx, name, seed, k, out):
    prior_f, corr_f, head = PIPELINES[name]
    t0 = time.monotonic()
    recon, s1 = stage1(ctx, prior_f, corr_f, k, seed)
    tv, vv = ctx.views[("train", k)], ctx.views[("validation", k)]
    xt, xv = features(ctx, tv, recon["train"]), features(ctx, vv, recon["validation"])
    pred, s2 = head(ctx, xt, xv, tv, vv, seed, out)
    p = vv.points
    q = p.primary_query.to_numpy()
    truth_v = ctx.truth[("validation", k)].target_true_mono  # evaluation only
    hidden = ~p.visible_flag.to_numpy()
    res = dict(pipeline=name, seed=seed, k=k,
               primary_rmse=macro_rmse(p.response, pred, p.surface_replicate_id.to_numpy(), q),
               all_points_rmse=macro_rmse(p.response, pred, p.surface_replicate_id.to_numpy(), np.ones(len(p), bool)),
               recon_rmse_hidden_query=float(np.sqrt(np.mean((recon["validation"][hidden & q] - truth_v[hidden & q]) ** 2))),
               wall_seconds=time.monotonic() - t0, **{k2: v for k2, v in s1.items()})
    pd.DataFrame(dict(source_row=p.source_combodrugseq, surface_id=p.surface_replicate_id, target_drug=p.target_drug,
                      primary_query=q, target=p.response, prediction=pred, recon_target_mono=recon["validation"])) \
        .to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps({**res, **{k2: v for k2, v in s2.items()}}, indent=1, default=float), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs/model_improvement"):
        raise ValueError("Output must be a fresh subdirectory of runs/model_improvement")
    output.mkdir(parents=True)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(cfg["threads"])
    accesses = []; original = pd.read_parquet
    def guard(path, *a, **kw):
        resolved = Path(path).resolve()
        if resolved.name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError(f"Forbidden response source: {resolved}")
        accesses.append(str(resolved)); return original(path, *a, **kw)
    results = []
    try:
        with patch.object(pd, "read_parquet", guard):
            ctx = Context(cfg)
            manifest = {f"{r}_k{k}": dict(view_sha256=v.sha256(), points=len(v.points), primary_query=int(v.points.primary_query.sum()),
                                           visible=int(v.points.visible_flag.sum()), supports=len(v.supports))
                        for (r, k), v in ctx.views.items()}
            for (r, k), v in ctx.views.items():
                if k == 1:
                    v.supports.to_csv(output / f"supports_{r}.csv", index=False)
            (output / "task_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            jobs = [(n, s, k, "primary") for n in cfg["pipelines"] for k in cfg["budgets"] for s in (cfg["seeds"] if n != "A_ref" else [0])]
            jobs += [(n, cfg["seeds"][0] if n != "A_ref" else 0, cfg["budgets"][0], "repeat") for n in cfg["pipelines"]]
            primary = {}
            for n, s, k, run in jobs:
                out = output / f"{n}_k{k}_seed_{s}_{run}"; out.mkdir()
                r = run_job(ctx, n, s, k, out); r["run"] = run; results.append(r)
                pred = pd.read_csv(out / "predictions.csv").prediction.to_numpy()
                if run == "primary":
                    primary[(n, s, k)] = pred
                else:
                    diff = float(np.max(np.abs(primary[(n, s, k)] - pred))); r["repeat_max_abs_diff"] = diff
                    if diff > 1e-7:
                        raise RuntimeError(f"Repeat drift {n} {diff}")
                pd.DataFrame(results).to_csv(output / "results.csv", index=False)
                print(f"DONE {n} k={k} seed={s} {run} primary={r['primary_rmse']:.6f} recon={r['recon_rmse_hidden_query']:.5f}", flush=True)
        (output / "summary.json").write_text(json.dumps(dict(status="passed", test_response_rows_read=0, jobs=len(results)), indent=1), encoding="utf-8")
    except Exception as error:
        (output / "failure.json").write_text(json.dumps(dict(status="failed", error=repr(error))), encoding="utf-8"); raise
    finally:
        (output / "input_access.json").write_text(json.dumps(dict(parquet_reads=accesses, test_response_rows_read=0), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""ObsCond-v1 runner. Protocol: docs/OBS_COND_V1_PROTOCOL.md.

Same task, views, supports and queries as SparseMono-v1 (its ``Context`` is reused unchanged).
Pipelines  OC_<gnn|morgan>_<r1|r2>:
  stage 1  structural prior (StructPrior, same encoder family) — 4 out-of-fold copies over training
           chemical clusters (training targets) + 1 full copy (validation targets, all partners);
           the full copy's encoders initialise the main model;
  stage 2  "adapt": encoders frozen; support update, interaction and heads trained;
  stage 3  "joint": everything (incl. the drug encoder) fine-tuned at a small learning rate.
Loss: surface-RMSE on combination responses + aux_weight * RMSE of the target monotherapy
estimate at hidden training query doses (training truth = supervision only, never an input).
Checkpoint: validation primary-query RMSE every ``eval_every`` steps over stages 2+3 (step 0
included) — the same selection rule as the SparseMono-v1 neural heads.
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
import torch
import yaml

from bisurfgnn.data.batch import pad_graph_items
from bisurfgnn.learned_benchmark import _seed_everything
from bisurfgnn.models.graph_encoder import GraphEncoder
from bisurfgnn.models.obs_cond import ArmInputs, BitTokens, ObsCondBiSurf, StructPrior, make_drug_encoder

spec = importlib.util.spec_from_file_location("sm_runner", ROOT / "experiments/model_improvement/run_sparse_mono.py")
sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)
Context, macro_rmse, MONO_MAX = sm.Context, sm.macro_rmse, sm.MONO_MAX


# ------------------------------------------------------------------ banks and stage 1
def banks_for(ctx, kind, dev):
    drug = pad_graph_items(ctx.banks.drug_graphs).to(dev) if kind == "gnn" else BitTokens.from_fingerprints(ctx.fingerprints).to(dev)
    return drug, pad_graph_items(ctx.banks.cell_graphs).to(dev)


def new_encoder(ctx, kind, c):
    return make_drug_encoder(kind, ctx.banks.atom_feature_dim, ctx.fingerprints.shape[1], c["hidden"], c["layers"], c["dropout"])


def fit_prior(ctx, kind, c, mono, seed, dev):
    _seed_everything(seed)
    model = StructPrior(new_encoder(ctx, kind, c), ctx.banks.cell_feature_dim, c["hidden"], c["layers"], c["dropout"],
                        float(mono.dose.mean()), float(mono.dose.std()), float(mono.mono.mean())).to(dev)
    drug_in, cell_in = banks_for(ctx, kind, dev)
    d = torch.as_tensor([ctx.drug_index[x] for x in mono.drug], device=dev)
    cc = torch.as_tensor([ctx.cell_index[x] for x in mono.cell_id], device=dev)
    z = torch.as_tensor(mono.dose.to_numpy(), dtype=torch.float32, device=dev)
    y = torch.as_tensor(mono.mono.to_numpy(), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=c["prior_lr"], weight_decay=c["weight_decay"])
    g = torch.Generator().manual_seed(seed)
    for _ in range(c["prior_steps"]):
        model.train(); i = torch.randint(0, len(y), (c["prior_batch"],), generator=g).to(dev)
        loss = torch.mean((model(drug_in, cell_in, d[i], cc[i], z[i]) - y[i]) ** 2)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite prior loss")
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    @torch.no_grad()
    def predict(drugs, cells, doses):
        dd = torch.as_tensor([ctx.drug_index[x] for x in drugs], device=dev)
        ce = torch.as_tensor([ctx.cell_index[x] for x in cells], device=dev)
        zz = torch.as_tensor(np.nan_to_num(np.asarray(doses, float)), dtype=torch.float32, device=dev)
        return model(drug_in, cell_in, dd, ce, zz).cpu().numpy()
    return model, predict


def stage1(ctx, kind, c, seed, dev):
    """Prior predictions aligned with the k=1 views (k=0 shares target/partner doses; supports are masked)."""
    tv = ctx.views[("train", 1)].points; vv = ctx.views[("validation", 1)].points
    fold = tv.target_drug.map(ctx.folds)
    if fold.isna().any():
        raise RuntimeError("Training target without a cluster fold")
    out = {"train": dict(tq=np.full(len(tv), np.nan), ts=np.full(len(tv), np.nan)), "validation": {}}
    for f in range(ctx.cfg["oof_folds"]):
        _, pred = fit_prior(ctx, kind, c, ctx.mono[ctx.mono.drug.map(ctx.folds) != f], seed + f, dev)
        sel = (fold == f).to_numpy()
        out["train"]["tq"][sel] = pred(tv.target_drug[sel], tv.cell_id[sel], tv.target_dose[sel])
        out["train"]["ts"][sel] = pred(tv.target_drug[sel], tv.cell_id[sel], tv.support_dose[sel])
    full, pred = fit_prior(ctx, kind, c, ctx.mono, seed + 97, dev)
    out["validation"]["tq"] = pred(vv.target_drug, vv.cell_id, vv.target_dose)
    out["validation"]["ts"] = pred(vv.target_drug, vv.cell_id, vv.support_dose)
    for role, p in (("train", tv), ("validation", vv)):
        out[role]["pp"] = pred(p.partner_drug, p.cell_id, p.partner_dose)
    if any(np.isnan(v).any() for r in out.values() for v in r.values()):
        raise RuntimeError("Missing prior prediction")
    state = {k: v.detach().cpu().clone() for k, v in full.state_dict().items()}
    return out, state


# ------------------------------------------------------------------ inputs (RoleView + priors only)
def arm_inputs(ctx, view, pri):
    p = view.points
    is_a = (p.target_arm == "a").to_numpy()
    sv = p.support_value.to_numpy(float); smask = ~np.isnan(sv)
    idx = lambda s: np.array([ctx.drug_index[x] for x in s])
    T = dict(drug=idx(p.target_drug), dose=p.target_dose.to_numpy(float), mono=np.nan_to_num(p.visible_value.to_numpy(float)),
             flag=p.visible_flag.to_numpy(float), prior=pri["tq"], sup_dose=np.nan_to_num(p.support_dose.to_numpy(float)),
             sup_value=np.nan_to_num(sv), sup_prior=np.where(smask, pri["ts"], 0.0), sup_mask=smask)
    pm = p.partner_mono.to_numpy(float)
    if not np.isfinite(pm).all():
        raise RuntimeError("Partner monotherapy must be observed")
    P = dict(drug=idx(p.partner_drug), dose=p.partner_dose.to_numpy(float), mono=pm, flag=np.ones(len(p)), prior=pri["pp"],
             sup_dose=p.partner_dose.to_numpy(float), sup_value=pm, sup_prior=pri["pp"], sup_mask=np.ones(len(p), bool))
    def build(first, second):
        f = {k: np.where(is_a, first[k], second[k]) for k in T}
        ten = lambda k, v: torch.as_tensor(v, dtype=torch.long if k == "drug" else (torch.bool if k == "sup_mask" else torch.float32))
        return ArmInputs(**{k: (ten(k, v)[:, None] if k.startswith("sup_") else ten(k, v)) for k, v in f.items()})
    cell = torch.as_tensor([ctx.cell_index[x] for x in p.cell_id], dtype=torch.long)
    return build(T, P), build(P, T), cell, torch.as_tensor(is_a)


def input_hash(a, b, cell):
    h = hashlib.sha256()
    for arm in (a, b):
        for f in arm.__dataclass_fields__:
            h.update(getattr(arm, f).cpu().numpy().tobytes())
    h.update(cell.numpy().tobytes()); return h.hexdigest()


# ------------------------------------------------------------------ main model
def build_model(ctx, kind, c, prior_state, dose_stats):
    model = ObsCondBiSurf(new_encoder(ctx, kind, c), GraphEncoder(ctx.banks.cell_feature_dim, c["hidden"], c["layers"], c["dropout"]),
                          c["hidden"], c["heads"], c["dropout"], dose_stats[0], dose_stats[1], ctx.residual_scale, MONO_MAX)
    if prior_state is not None:
        for name in ("drug_encoder", "cell_encoder"):
            sub = {k[len(name) + 1:]: v for k, v in prior_state.items() if k.startswith(name + ".")}
            getattr(model, name).load_state_dict(sub, strict=True)
    return model


@torch.no_grad()
def predict(model, banks_in, cell, a, b, dev, chunk=2048, banks=None):
    model.eval()
    banks = banks or model.encode_banks(*banks_in)
    outs = [model(banks, cell[i:i + chunk].to(dev), a.index(slice(i, i + chunk)).to(dev), b.index(slice(i, i + chunk)).to(dev))
            for i in range(0, len(cell), chunk)]
    cat = lambda f: torch.cat([f(o) for o in outs]).cpu()
    return dict(pred=cat(lambda o: o.prediction).numpy(), mono_a=cat(lambda o: o.mono_hat_a).numpy(), mono_b=cat(lambda o: o.mono_hat_b).numpy(),
                dz_a=cat(lambda o: (o.arm_a.z_updated - o.arm_a.z).norm(dim=1) / o.arm_a.z.norm(dim=1).clamp_min(1e-8)).numpy(),
                dz_b=cat(lambda o: (o.arm_b.z_updated - o.arm_b.z).norm(dim=1) / o.arm_b.z.norm(dim=1).clamp_min(1e-8)).numpy())


def target_of(out, is_a, key="mono"):
    return np.where(is_a.numpy(), out[f"{key}_a"], out[f"{key}_b"])


def train_main(ctx, kind, cfg_key, seed, k, pri, prior_state, out_dir):
    c = ctx.cfg["obs"][cfg_key]; dev = torch.device(ctx.cfg["device"])
    tv, vv = ctx.views[("train", k)], ctx.views[("validation", k)]
    at, bt, ct, ia_t = arm_inputs(ctx, tv, pri["train"]); av, bv, cv, ia_v = arm_inputs(ctx, vv, pri["validation"])
    _seed_everything(seed)
    model = build_model(ctx, kind, c, prior_state, (float(ctx.mono.dose.mean()), float(ctx.mono.dose.std()))).to(dev)
    banks_in = banks_for(ctx, kind, dev)
    at, bt, ct, ia_dev = at.to(dev), bt.to(dev), ct.to(dev), ia_t.to(dev)
    y = torch.as_tensor(tv.points.response.to_numpy(), dtype=torch.float32, device=dev)
    aux_y = torch.as_tensor(np.nan_to_num(ctx.truth[("train", k)].target_true_mono), dtype=torch.float32, device=dev)  # supervision only
    aux_m = torch.as_tensor(tv.points.primary_query.to_numpy(), device=dev)
    surf = tv.points.surface_replicate_id.to_numpy()
    groups = pd.Series(np.arange(len(surf))).groupby(surf).apply(list); keys = list(groups.index)
    eps = (0.01 * ctx.residual_scale) ** 2
    yv, s_v, q_v = vv.points.response.to_numpy(), vv.points.surface_replicate_id.to_numpy(), vv.points.primary_query.to_numpy()
    score = lambda: macro_rmse(yv, predict(model, banks_in, cv, av, bv, dev)["pred"], s_v, q_v)
    best = dict(score=score(), state=copy.deepcopy(model.state_dict()), step=0, stage="init"); hist = [dict(step=0, stage="init", val_primary=best["score"])]
    rng = np.random.default_rng(seed); step = 0; grad_check = {}
    enc_params = list(model.drug_encoder.parameters()) + list(model.cell_encoder.parameters())

    def batch_loss(banks, idx, codes, n):
        o = model(banks, ct[idx], at.index(idx), bt.index(idx))
        onehot = torch.zeros(n, len(idx), device=dev); onehot[codes, torch.arange(len(idx), device=dev)] = 1.0
        mse_s = onehot @ (o.prediction - y[idx]) ** 2 / onehot.sum(1)
        combo = torch.sqrt(mse_s + eps).mean()
        mt = torch.where(ia_dev[idx], o.mono_hat_a, o.mono_hat_b); m = aux_m[idx]
        aux = torch.sqrt(((mt - aux_y[idx]) ** 2)[m].mean() + eps) if bool(m.any()) else combo * 0
        return combo, aux

    for stage, steps, lr in (("adapt", c["adapt_steps"], c["adapt_lr"]), ("joint", c["joint_steps"], c["joint_lr"])):
        for p in enc_params:
            p.requires_grad_(stage == "joint")
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=c["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t, s=steps: 0.5 * (1 + np.cos(np.pi * min(t, s) / s)))
        frozen_banks = None
        if stage == "adapt":
            model.drug_encoder.eval(); model.cell_encoder.eval()
            with torch.no_grad():
                frozen_banks = model.encode_banks(*banks_in)
        t = 0
        while t < steps:
            order = rng.permutation(len(keys))
            for start in range(0, len(order), c["batch_surfaces"]):
                if t >= steps:
                    break
                sel = [groups[keys[j]] for j in order[start:start + c["batch_surfaces"]]]
                idx = torch.as_tensor(np.concatenate(sel), device=dev)
                codes = torch.as_tensor(np.repeat(np.arange(len(sel)), [len(x) for x in sel]), device=dev)
                model.train()
                if stage == "adapt":
                    model.drug_encoder.eval(); model.cell_encoder.eval()
                banks = frozen_banks if stage == "adapt" else model.encode_banks(*banks_in)
                combo, aux = batch_loss(banks, idx, codes, len(sel))
                if stage == "joint" and t == 0:  # combination loss alone must reach the molecular encoder
                    gd = torch.autograd.grad(combo, list(model.drug_encoder.parameters()), retain_graph=True, allow_unused=True)
                    gc = torch.autograd.grad(combo, list(model.cell_encoder.parameters()), retain_graph=True, allow_unused=True)
                    norm = lambda gs: float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs if g is not None)))
                    grad_check = dict(combo_grad_norm_drug_encoder=norm(gd), combo_grad_norm_cell_encoder=norm(gc),
                                      drug_encoder_params_without_grad=sum(g is None for g in gd))
                    if not (np.isfinite(grad_check["combo_grad_norm_drug_encoder"]) and grad_check["combo_grad_norm_drug_encoder"] > 0):
                        raise RuntimeError(f"Combination loss does not reach the drug encoder: {grad_check}")
                loss = combo + c["aux_weight"] * aux
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite loss")
                opt.zero_grad(set_to_none=True); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not bool(torch.isfinite(gn)):
                    raise RuntimeError("Non-finite gradient")
                opt.step(); sched.step(); t += 1; step += 1
                if t % c["eval_every"] == 0 or t == steps:
                    sc = score(); hist.append(dict(step=step, stage=stage, val_primary=sc, train_combo=float(combo.detach()), train_aux=float(aux.detach())))
                    if sc < best["score"] - 1e-5:
                        best.update(score=sc, state=copy.deepcopy(model.state_dict()), step=step, stage=stage)
    for p in enc_params:
        p.requires_grad_(True)
    end_adapt = next(h["val_primary"] for h in hist if h["stage"] == "adapt" and h["step"] == c["adapt_steps"])
    model.load_state_dict(best["state"]); torch.save(best["state"], out_dir / "model.pt")
    clone = build_model(ctx, kind, c, None, (0.0, 1.0)).to(dev)
    clone.load_state_dict(torch.load(out_dir / "model.pt", weights_only=True, map_location="cpu"))
    res_v = predict(model, banks_in, cv, av, bv, dev)
    np.testing.assert_allclose(predict(clone, banks_in, cv, av, bv, dev)["pred"], res_v["pred"], atol=0, rtol=0)
    # ---- diagnostics (validation truth = evaluation only)
    p = vv.points; hidden_q = (~p.visible_flag.to_numpy()) & q_v
    truth_v = ctx.truth[("validation", k)].target_true_mono
    recon_v = np.where(p.visible_flag.to_numpy(), np.nan_to_num(p.visible_value.to_numpy(float)), np.clip(target_of(res_v, ia_v), 0, MONO_MAX))
    rm = lambda a_, b_, m_: float(np.sqrt(np.mean((a_[m_] - b_[m_]) ** 2))) if m_.any() else float("nan")
    diag = dict(selected_step=best["step"], selected_stage=best["stage"], val_primary_end_adapt=end_adapt,
                val_primary_init=hist[0]["val_primary"], **grad_check,
                recon_prior_only_hidden_query=rm(np.clip(pri["validation"]["tq"], 0, MONO_MAX), truth_v, hidden_q),
                train_recon_in_sample_hidden_query=None, train_prior_oof_hidden_query=None)
    if k == 1:
        above = (p.target_dose > p.support_dose).to_numpy(); below = (p.target_dose < p.support_dose).to_numpy()
        diag.update(recon_above_support=rm(recon_v, truth_v, hidden_q & above), recon_below_support=rm(recon_v, truth_v, hidden_q & below),
                    combo_above_support=macro_rmse(yv, res_v["pred"], s_v, q_v & above), combo_below_support=macro_rmse(yv, res_v["pred"], s_v, q_v & below))
        noup_a, noup_b = copy.copy(av), copy.copy(bv)
        for arm, tgt in ((noup_a, ia_v), (noup_b, ~ia_v)):
            arm.sup_mask = arm.sup_mask & ~tgt[:, None]
        r0 = predict(model, banks_in, cv, noup_a, noup_b, dev)
        pert_a, pert_b = copy.copy(av), copy.copy(bv)
        for arm, tgt in ((pert_a, ia_v), (pert_b, ~ia_v)):
            arm.sup_value = arm.sup_value + 0.1 * tgt[:, None].float()
            arm.mono = arm.mono + 0.1 * (tgt & (arm.flag > 0.5)).float()
        r1 = predict(model, banks_in, cv, pert_a, pert_b, dev)
        diag.update(primary_rmse_target_support_removed=macro_rmse(yv, r0["pred"], s_v, q_v),
                    recon_target_support_removed=rm(np.clip(target_of(r0, ia_v), 0, MONO_MAX), truth_v, hidden_q),
                    mean_abs_pred_change_support_removed=float(np.mean(np.abs(r0["pred"] - res_v["pred"])[q_v])),
                    mean_abs_pred_change_support_plus_0p1=float(np.mean(np.abs(r1["pred"] - res_v["pred"])[q_v])),
                    mean_abs_mono_change_support_plus_0p1=float(np.mean(np.abs(target_of(r1, ia_v) - target_of(res_v, ia_v))[hidden_q])),
                    target_relative_update_norm=float(np.mean(target_of(res_v, ia_v, "dz"))))
    else:
        diag.update(target_relative_update_norm=float(np.mean(target_of(res_v, ia_v, "dz"))))
    res_t = predict(model, banks_in, ct.cpu(), at, bt, dev)
    hq_t = (~tv.points.visible_flag.to_numpy()) & tv.points.primary_query.to_numpy(); truth_t = ctx.truth[("train", k)].target_true_mono
    diag.update(train_recon_in_sample_hidden_query=rm(np.clip(target_of(res_t, ia_t), 0, MONO_MAX), truth_t, hq_t),
                train_prior_oof_hidden_query=rm(np.clip(pri["train"]["tq"], 0, MONO_MAX), truth_t, hq_t),
                validation_input_sha256=input_hash(av, bv, cv),
                parameters=dict(total=sum(x.numel() for x in model.parameters()), drug_encoder=sum(x.numel() for x in model.drug_encoder.parameters())))
    return res_v["pred"], recon_v, diag, hist


def run_job(ctx, name, seed, k, out, prior_cache):
    _, kind, cfg_key = name.split("_")
    c = ctx.cfg["obs"][cfg_key]; dev = torch.device(ctx.cfg["device"]); t0 = time.monotonic()
    key = (kind, cfg_key, seed)
    if prior_cache is not None and key in prior_cache:
        pri, state = prior_cache[key]
    else:
        pri, state = stage1(ctx, kind, c, seed, dev)
        if prior_cache is not None:
            prior_cache[key] = (pri, state)
    pred, recon, diag, hist = train_main(ctx, kind, cfg_key, seed, k, pri, state, out)
    p = ctx.views[("validation", k)].points; q = p.primary_query.to_numpy()
    truth_v = ctx.truth[("validation", k)].target_true_mono; hidden = ~p.visible_flag.to_numpy()
    res = dict(pipeline=name, seed=seed, k=k, primary_rmse=macro_rmse(p.response, pred, p.surface_replicate_id.to_numpy(), q),
               all_points_rmse=macro_rmse(p.response, pred, p.surface_replicate_id.to_numpy(), np.ones(len(p), bool)),
               recon_rmse_hidden_query=float(np.sqrt(np.mean((recon[hidden & q] - truth_v[hidden & q]) ** 2))),
               prior_validation_hash=hashlib.sha256(np.round(pri["validation"]["tq"], 8).tobytes()).hexdigest(),
               wall_seconds=time.monotonic() - t0, **diag)
    pd.DataFrame(dict(source_row=p.source_combodrugseq, surface_id=p.surface_replicate_id, target_drug=p.target_drug, primary_query=q,
                      target=p.response, prediction=pred, recon_target_mono=recon)).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(hist).to_csv(out / "curve.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
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
            (output / "task_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            expected = cfg.get("expected_view_sha256", {})
            for key, h in expected.items():
                if manifest[key]["view_sha256"] != h:
                    raise RuntimeError(f"View hash mismatch for {key}")
            jobs = [(n, s, k, "primary") for n in cfg["pipelines"] for s in cfg["seeds"] for k in cfg["budgets"]]
            if cfg.get("repeat", True):
                jobs += [(n, cfg["seeds"][0], cfg["budgets"][0], "repeat") for n in cfg["pipelines"]]
            primary, cache = {}, {}
            for n, s, k, run in jobs:
                out = output / f"{n}_k{k}_seed_{s}_{run}"; out.mkdir()
                r = run_job(ctx, n, s, k, out, cache if run == "primary" else None); r["run"] = run; results.append(r)
                pred = pd.read_csv(out / "predictions.csv").prediction.to_numpy()
                if run == "primary":
                    primary[(n, s, k)] = pred
                else:
                    diff = float(np.max(np.abs(primary[(n, s, k)] - pred))); r["repeat_max_abs_diff"] = diff
                    if diff > 1e-7:
                        raise RuntimeError(f"Repeat drift {n} {diff}")
                pd.DataFrame(results).to_csv(output / "results.csv", index=False)
                print(f"DONE {n} k={k} seed={s} {run} primary={r['primary_rmse']:.6f} recon={r['recon_rmse_hidden_query']:.5f} "
                      f"step={r['selected_step']}({r['selected_stage']}) wall={r['wall_seconds']:.0f}s", flush=True)
        (output / "summary.json").write_text(json.dumps(dict(status="passed", test_response_rows_read=0, jobs=len(results)), indent=1), encoding="utf-8")
    except Exception as error:
        (output / "failure.json").write_text(json.dumps(dict(status="failed", error=repr(error))), encoding="utf-8"); raise
    finally:
        (output / "input_access.json").write_text(json.dumps(dict(parquet_reads=accesses, test_response_rows_read=0), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

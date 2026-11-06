"""NumericSurface-v1 stage-3 structural diagnostics (training/validation roles only).

Static part reads frozen data and the saved AlgorithmSearch-v1 checkpoints.
Dynamic part retrains v2 / mdx seed 41 with extra logging (eval-mode training
RMSE on the fixed training table, per-module gradient and update norms, last and
lowest-training-error states) and runs one fixed small-subset fit check.
No test responses are read. Output: one JSON + CSVs in --output.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
import yaml

HERE = ROOT / "experiments/model_improvement"
spec = importlib.util.spec_from_file_location("fitgen", HERE / "analysis/diagnose_fit_vs_generalization.py")
fitgen = importlib.util.module_from_spec(spec); spec.loader.exec_module(fitgen)
search, pilot, graph, learned = fitgen.search, fitgen.pilot, fitgen.graph, fitgen.learned
SEARCH = ROOT / "runs/model_improvement/algorithm_search_v1"
GROUPS = ["drug_encoder", "context_encoder", "bixt", "context_fusion",
          "interaction_decoder.shared_dose_arm", "interaction_decoder.interaction_network",
          "interaction_decoder.residual_head", "monotherapy_dose_branch"]


def macro(table, pred):
    return pilot.metric(table.target.numpy(), pred, table.surface_ids)["surface_macro_rmse"]


def group_of(name):
    for g in GROUPS:
        if name.startswith(g + "."):
            return g
    return "other"


def static_checks(frames, tables, banks, cfg, device):
    tr, va = frames["train"], frames["validation"]
    out = {}
    d = pd.concat([tr[["dose_a_model", "dose_b_model"]], va[["dose_a_model", "dose_b_model"]]])
    span = tr.groupby("surface_replicate_id").dose_a_model.agg(lambda x: x.max() - x.min())
    levels = tr.groupby("surface_replicate_id").dose_a_model.nunique()
    out["dose_model"] = dict(min=float(d.values.min()), max=float(d.values.max()),
                             train_q01_q99=[float(tr.dose_a_model.quantile(.01)), float(tr.dose_a_model.quantile(.99))],
                             within_surface_span_median=float(span.median()), within_surface_levels=levels.value_counts().to_dict(),
                             definition="log10(1 + dose_M / 1e-12)")
    freqs = (np.pi * 2.0 ** np.arange(4)).tolist()
    out["dose_fourier"] = dict(frequencies=freqs, periods_in_dose_units=[2 * np.pi / f for f in freqs],
                               note="sin/cos(f*dose) with dose in [0,9]: periods 2,1,0.5,0.25 log10 units")
    for role, f in frames.items():
        hsa = np.minimum(f.mono_response_a, f.mono_response_b)
        out[f"{role}_ranges"] = dict(target_min=float(f.response.min()), target_max=float(f.response.max()),
            target_above_one=float((f.response > 1).mean()), mono_max=float(max(f.mono_response_a.max(), f.mono_response_b.max())),
            hsa_below_0p05=float((hsa < .05).mean()), residual_sd=float((f.response - hsa).std()))
    gate = (1 - np.exp(-2 * va.dose_a_model)) * (1 - np.exp(-2 * va.dose_b_model))
    out["dose_gate_validation"] = dict(min=float(gate.min()), q01=float(np.quantile(gate, .01)), fraction_below_0p99=float((gate < .99).mean()))
    n_surf = tr.surface_replicate_id.nunique()
    out["optimisation_steps"] = dict(train_surfaces=n_surf, batch_surfaces=128, steps_per_epoch=int(np.ceil(n_surf / 128)),
                                     steps_40_epochs=int(np.ceil(n_surf / 128)) * 40)
    # loss vs metric weighting at the v2 seed-41 predictions
    p = pd.read_csv(SEARCH / "screen_b1/bisurf_mono_v2_seed_41_primary/predictions.csv")
    rmse_s = np.sqrt(((p.prediction - p.target) ** 2).groupby(p.surface_id).mean())
    w = 1 / (2 * rmse_s.clip(lower=1e-6))
    out["loss_vs_metric"] = dict(note="loss=mean_s MSE_s; metric=mean_s sqrt(MSE_s); d metric/d MSE_s = 1/(2 RMSE_s)",
                                 surface_rmse_q10_q50_q90=[float(rmse_s.quantile(q)) for q in (.1, .5, .9)],
                                 metric_weight_ratio_q90_over_q10=float(w.quantile(.9) / w.quantile(.1)))
    # Sigmoid slope, needed logit shift, and LayerNorm share of the monotherapy column, per checkpoint
    for name in ("bisurf_mono_v2", "bisurf_search_mdx_v2"):
        model = search.build_model(banks, name, 41, cfg)
        model.load_state_dict(torch.load(SEARCH / f"screen_b1/{name}_seed_41_primary/model.pt", weights_only=True, map_location="cpu"))
        model.eval()
        va_t = tables["validation"]
        hsa = np.minimum(va_t.mono_a.numpy(), va_t.mono_b.numpy()).astype(float)
        base = np.log(np.clip(hsa / 3, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(hsa / 3, 1e-6, 1 - 1e-6))
        pred = search.predict(model, va_t, banks, cfg).astype(float)
        sig = pred / 3
        slope = 3 * sig * (1 - sig)
        t = np.clip(va_t.target.numpy() / 3, 1e-6, 1 - 1e-6)
        needed = np.log(t) - np.log1p(-t) - base
        learned_shift = np.log(np.clip(sig, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(sig, 1e-6, 1 - 1e-6)) - base
        err = (pred - va_t.target.numpy()) ** 2
        low = slope < 0.05
        out[f"{name}_output_layer"] = dict(slope_fraction_below_0p05=float(low.mean()),
            squared_error_share_slope_below_0p05=float(err[low].sum() / err.sum()),
            needed_logit_shift_abs_q50_q90=[float(np.quantile(np.abs(needed), q)) for q in (.5, .9)],
            learned_logit_shift_abs_q50_q90=[float(np.quantile(np.abs(learned_shift), q)) for q in (.5, .9)],
            target_outside_open_unit_bound=float(((va_t.target.numpy() <= 0) | (va_t.target.numpy() >= 3)).mean()))
        # Per-column contribution after the arm LayerNorm (first Linear weights x normalised inputs)
        arm = model.interaction_decoder.shared_dose_arm
        ln, lin = arm[0], arm[1]
        idx = torch.arange(min(4096, len(va_t)))
        captured = {}
        h = ln.register_forward_hook(lambda m, i, o: captured.setdefault("x", o.detach()))
        with torch.no_grad():
            graph._model_forward(model, va_t, banks, idx, torch.device("cpu"))
        h.remove()
        x = captured["x"]  # [N, width] normalised
        contrib = (x[:, :, None] * lin.weight.T[None]).pow(2).mean(dim=(0, 2))
        width = x.shape[1]
        out[f"{name}_arm_layernorm"] = dict(input_width=width,
            mono_column_share_of_first_layer_energy=float(contrib[-1] / contrib.sum()),
            uniform_share=1.0 / width,
            dose_feature_share=float(contrib[-11:-1].sum() / contrib.sum()))
        # gradient of the learned logit w.r.t. monotherapy: decoder path vs explicit branch
        mono_a = va_t.mono_a[idx].clone().requires_grad_(True)
        drug_bank = model.encode_drugs(graph.pad_graph_items(banks.drug_graphs))
        ctx = model.encode_context(graph.pad_graph_items(banks.cell_graphs))
        m = torch.ones(len(idx), dtype=torch.bool)
        o = model.forward_pooled(drug_bank, ctx, va_t.drug_a_index[idx].long(), va_t.drug_b_index[idx].long(),
                                 va_t.cell_index[idx].long(), va_t.doses[idx], mono_a, va_t.mono_b[idx], m, m)
        g, = torch.autograd.grad(o.interaction_logit.sum(), mono_a)
        out[f"{name}_dlogit_dmono_abs_mean"] = float(g.abs().mean())
    return out


def train_logged(name, seed, tables, banks, cfg, device, epochs, subset_surfaces=None, lr=None):
    model = search.build_model(banks, name, seed, cfg)
    tr, va = tables["train"], tables["validation"]
    if subset_surfaces is not None:
        keep = sorted(set(tr.surface_ids))[:subset_surfaces]
        rows = np.where(np.isin(tr.surface_ids, keep))[0]
        tr = sub_table(tr, rows)
    opt = torch.optim.AdamW(model.parameters(), lr=lr or cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    hist, best = [], dict(val=macro(va, search.predict(model, va, banks, cfg)), epoch=-1)
    min_train = dict(train=np.inf)
    for epoch in range(epochs):
        model.train(); grads = {g: 0.0 for g in GROUPS + ["other"]}; steps = 0
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        batch = len(set(tr.surface_ids)) if subset_surfaces else cfg["training"]["batch_surfaces"]
        for idx in graph._surface_batch_indices(tr, batch, shuffle_seed=seed * 1000003 + epoch):
            opt.zero_grad(set_to_none=True)
            out = graph._model_forward(model, tr, banks, idx, device)
            loss = pilot.loss_equal_surface(out.prediction, tr.target[idx].to(device), tr.surface_ids[idx.numpy()])
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    grads[group_of(n)] += float(p.grad.detach().pow(2).sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            opt.step(); steps += 1
        upd = {g: 0.0 for g in grads}; size = {g: 0.0 for g in grads}
        for n, p in model.named_parameters():
            upd[group_of(n)] += float((p.detach() - before[n]).pow(2).sum()); size[group_of(n)] += float(before[n].pow(2).sum())
        tr_rmse = macro(tr, search.predict(model, tr, banks, cfg))
        va_rmse = macro(va, search.predict(model, va, banks, cfg))
        row = dict(epoch=epoch, steps=steps, train_eval_rmse=tr_rmse, val_rmse=va_rmse)
        row.update({f"gradnorm_{g}": (v / steps) ** .5 for g, v in grads.items() if size[g] > 0})
        row.update({f"rel_update_{g}": (upd[g] / size[g]) ** .5 for g in grads if size[g] > 0})
        hist.append(row)
        if va_rmse < best["val"]:
            best = dict(val=va_rmse, epoch=epoch, train=tr_rmse)
        if tr_rmse < min_train["train"]:
            min_train = dict(train=tr_rmse, epoch=epoch, val=va_rmse)
        print(f"DIAG {name} sub={subset_surfaces} ep={epoch} train={tr_rmse:.5f} val={va_rmse:.5f}", flush=True)
    hsa_tr = macro(tr, np.minimum(tr.mono_a.numpy(), tr.mono_b.numpy()))
    return pd.DataFrame(hist), dict(best_val=best, min_train=min_train, last=dict(train=hist[-1]["train_eval_rmse"], val=hist[-1]["val_rmse"]),
                                     hsa_train=hsa_tr, train_surfaces=len(set(tr.surface_ids)))


def sub_table(table, rows):
    rows_t = torch.as_tensor(rows)
    kw = {}
    for field in table.__dataclass_fields__:
        v = getattr(table, field)
        if isinstance(v, torch.Tensor) and v.shape[:1] == (len(table),):
            kw[field] = v[rows_t]
        elif isinstance(v, np.ndarray) and v.shape[:1] == (len(table),):
            kw[field] = v[rows]
        else:
            kw[field] = v
    return type(table)(**kw)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--subset-epochs", type=int, default=600)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((ROOT / "configs/model_improvement/algorithm_search_v1/screen_b1.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); cfg["device"] = str(device)
    torch.use_deterministic_algorithms(True)
    orig = pd.read_parquet
    def guard(path, *a, **k):
        if Path(path).name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError("Forbidden response source")
        return orig(path, *a, **k)
    start = time.monotonic()
    with patch.object(pd, "read_parquet", guard):
        frames, banks, tables, _ = fitgen.frozen_tables(cfg)
        cpu_cfg = dict(cfg, device="cpu")
        report = {"static": static_checks(frames, tables, banks, cpu_cfg, torch.device("cpu"))}
        for name in ("bisurf_mono_v2", "bisurf_search_mdx_v2"):
            h, s = train_logged(name, 41, tables, banks, cfg, device, args.epochs)
            h.to_csv(args.output / f"{name}_full_history.csv", index=False); report[f"{name}_full"] = s
            h, s = train_logged(name, 41, tables, banks, cfg, device, args.subset_epochs, subset_surfaces=128, lr=1e-3)
            h.to_csv(args.output / f"{name}_subset128_history.csv", index=False); report[f"{name}_subset128"] = s
    report["wall_seconds"] = time.monotonic() - start; report["test_response_rows_read"] = 0
    (args.output / "structure_diagnostic.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()

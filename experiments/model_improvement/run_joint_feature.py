"""JointFeature-v1 runner. Protocol: docs/JOINT_FEATURE_V1_PROTOCOL.md.

Model names (all routed through pilot.execute's per-seed "bisurf" branch)
  bisurf_joint_<enc>[_rmse]          candidate A, enc in {pleall, plelin}
  bisurf_jointcorr_<cfg>[_rmse]      candidate B, cfg in {b1, b2}
``_rmse`` = surface-RMSE training objective (candidate C); the reported metric is unchanged.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits

from bisurfgnn.models.joint_feature import JointModel, JointTabM, StateConditionedCorrector, fit_joint_stats, raw_features
from bisurfgnn.models.numeric_surface import TabMNumeric

spec = importlib.util.spec_from_file_location("jf_ns", Path(__file__).with_name("run_numeric_surface.py"))
NS = importlib.util.module_from_spec(spec); spec.loader.exec_module(NS)
search, pilot, graph, learned = NS.search, NS.pilot, NS.graph, NS.learned
A = re.compile(r"bisurf_joint_(pleall|plelin)(_rmse)?")
B = re.compile(r"bisurf_jointcorr_(b1|b2)(_rmse)?")


def surface_objective(pred, data, idx, kind, eps):
    """Surface-equal MSE (default) or mean over surfaces of sqrt(MSE_s + eps). Deterministic (dense one-hot)."""
    i = idx.to(pred.device)
    y = data.target[i]
    codes = np.unique(data.surface_ids[idx.cpu().numpy()], return_inverse=True)[1]
    onehot = torch.zeros(codes.max() + 1, len(codes), device=pred.device)
    onehot[torch.as_tensor(codes, device=pred.device), torch.arange(len(codes), device=pred.device)] = 1.0
    counts = onehot.sum(1)
    def one(p):
        mse_s = onehot @ (p - y) ** 2 / counts
        return mse_s.mean() if kind == "mse" else torch.sqrt(mse_s + eps).mean()
    if pred.ndim == 2:  # TabM members keep their own losses
        return torch.stack([one(pred[:, j]) for j in range(pred.shape[1])]).mean()
    return one(pred)


@torch.no_grad()
def predict(model, data):
    model.eval()
    out = [model.predict(data.mono_a[s:s + 4096], data.mono_b[s:s + 4096], data.doses[s:s + 4096],
                         data.a[s:s + 4096], data.b[s:s + 4096], data.c[s:s + 4096]).cpu()
           for s in range(0, len(data), 4096)]
    return torch.cat(out).numpy()


def optimise(model, params, train, val, opt, steps, seed, deadline, history, tag, best, loss_kind, eps, frozen=None):
    optimizer = torch.optim.AdamW(params, lr=opt["lr"], weight_decay=opt["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    step, epoch = 0, 0
    while step < steps:
        for idx in NS.batches(train, opt["batch_surfaces"], seed * 1000003 + epoch + 7919 * len(history)):
            if step >= steps:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("JointFeature wall budget exceeded")
            model.train()
            if frozen is not None:
                frozen.eval()
            optimizer.zero_grad(set_to_none=True)
            i = idx.to(train.mono_a.device)
            pred = model.predict(train.mono_a[i], train.mono_b[i], train.doses[i], train.a[i], train.b[i], train.c[i], members=True)
            loss = surface_objective(pred, train, idx, loss_kind, eps)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite JointFeature loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, opt["gradient_clip"])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite JointFeature gradient")
            optimizer.step(); sched.step(); step += 1
            if step % opt["eval_every"] == 0 or step == steps:
                score = NS.metric(val, predict(model, val))
                history.append(dict(stage=tag, step=step, val_rmse=score, grad_norm=float(norm),
                                    train_eval_rmse=NS.metric(train, predict(model, train))))
                if score < best["score"] - 1e-5:
                    best.update(score=score, state=copy.deepcopy(model.state_dict()), where=f"{tag}:{step}")
        epoch += 1


def feature_check(tables, banks, cfg, train, stats, out):
    """Runtime proof that our 55 columns equal the full-XGBoost construction column by column."""
    from sklearn.decomposition import PCA
    frame = pd.DataFrame({
        "mono_response_a": tables["validation"].mono_a.numpy().astype(float), "mono_response_b": tables["validation"].mono_b.numpy().astype(float),
        "dose_a_model": tables["validation"].doses[:, 0].numpy().astype(float), "dose_b_model": tables["validation"].doses[:, 1].numpy().astype(float),
        "drug_a_id": [banks.drug_ids[i] for i in tables["validation"].drug_a_index.tolist()],
        "drug_b_id": [banks.drug_ids[i] for i in tables["validation"].drug_b_index.tolist()],
        "cell_id": [banks.cell_ids[i] for i in tables["validation"].cell_index.tolist()]})
    tr = tables["train"]
    drug_index = {d: i for i, d in enumerate(banks.drug_ids)}; cell_index = {c: i for i, c in enumerate(banks.cell_ids)}
    fp = pilot.morgan512((ROOT / cfg["data_root"]).resolve(), banks).numpy().astype(float)
    tr_drugs = sorted({banks.drug_ids[i] for i in tr.drug_a_index.tolist()} | {banks.drug_ids[i] for i in tr.drug_b_index.tolist()})
    drug_feat = PCA(16, random_state=0).fit(fp[[drug_index[d] for d in tr_drugs]]).transform(fp)
    cell_vec = np.stack([g.node_features[:, 0].numpy() for g in banks.cell_graphs]).astype(float)
    tr_cells = sorted({banks.cell_ids[i] for i in tr.cell_index.tolist()})
    cell_feat = PCA(16, random_state=0).fit(cell_vec[[cell_index[c] for c in tr_cells]]).transform(cell_vec)
    report = {"columns": 55}
    for swap in (False, True):
        ref = search.xgb_features(frame, drug_feat, cell_feat, drug_index, cell_index, swap)
        va = tables["validation"]
        ours = raw_features(va.mono_a, va.mono_b, va.doses, va.drug_a_index, va.drug_b_index, va.cell_index,
                            stats.drug_codes, stats.cell_codes, swap=swap).double().numpy()
        diff = np.abs(ref - ours).max(0)
        report[f"swap_{swap}_max_abs_column_diff"] = float(diff.max())
        if ref.shape != ours.shape or diff.max() > 1e-4:
            raise RuntimeError(f"Feature matrix differs from XGBoost construction (swap={swap}): {diff.max()}")
    learned._write_json_atomic(out / "feature_check.json", report)
    return report


def run(name, tables, banks, cfg, seed, out, deadline):
    device = torch.device(cfg["device"])
    train, val = NS.Data(tables["train"], device), NS.Data(tables["validation"], device)
    ma, mb = A.fullmatch(name), B.fullmatch(name)
    loss_kind = "rmse" if (ma or mb).group(2) else "mse"
    drug, cell = NS.lowdim_codes(banks, train, cfg)
    cpu = lambda t: t.detach().cpu()
    stats = fit_joint_stats(cpu(train.mono_a), cpu(train.mono_b), cpu(train.doses), cpu(train.a), cpu(train.b), cpu(train.c),
                            cpu(train.target), drug, cell)
    info = {"feature_check": feature_check(tables, banks, cfg, train, stats, out), "loss": loss_kind}
    eps = (cfg["rmse_eps_fraction"] * stats.residual_scale) ** 2
    info["rmse_eps"] = eps
    history = []
    learned._seed_everything(seed)
    if ma:
        enc = {"pleall": "ple_all", "plelin": "ple_mono_linear_pca"}[ma.group(1)]
        build = lambda: JointModel(JointTabM(stats, encoding=enc, **cfg["joint_tabm"]))
    else:
        num_stats = NS.fit_stats(train)
        cc = cfg["corrector"][mb.group(1)]
        build = lambda: JointModel(StateConditionedCorrector(stats, dropout=cc["dropout"]),
                                   TabMNumeric(num_stats, **cfg["numeric_arch"]["tabm"]))
    model = build().to(device)
    np.testing.assert_allclose(predict(model, val), np.minimum(val.mono_a.cpu(), val.mono_b.cpu()), atol=1e-6)
    best = dict(score=NS.metric(val, predict(model, val)), state=copy.deepcopy(model.state_dict()), where="init")
    if ma:
        opt = cfg["joint_opt"][ma.group(1)]
        optimise(model, list(model.parameters()), train, val, opt, opt["steps"], seed, deadline, history, "joint", best, loss_kind, eps)
    else:
        src = Path(cfg["numeric_source_runs"][str(seed)]) / f"bisurf_num_tabm_o1_seed_{seed}_primary"
        state = torch.load(src / "model.pt", weights_only=True, map_location="cpu")
        model.base.load_state_dict({k[len("numeric."):]: v for k, v in state.items() if k.startswith("numeric.")})
        base_only = JointModel(StateConditionedCorrector(stats), model.base).to(device)  # zero corrector
        reproduced = predict(base_only, val)
        saved = pd.read_csv(src / "predictions.csv").prediction.to_numpy()
        if np.max(np.abs(reproduced - saved)) > 1e-6:
            raise RuntimeError("Base numeric checkpoint does not reproduce its saved predictions")
        info.update(base_source=learned._file_record(src / "model.pt"), base_val_rmse=NS.metric(val, reproduced),
                    base_train_rmse=NS.metric(train, predict(base_only, train)))
        best = dict(score=info["base_val_rmse"], state=copy.deepcopy(model.state_dict()), where="stage1")
        for p in model.base.parameters():
            p.requires_grad_(False)
        s2 = cc["stage2"]
        optimise(model, list(model.joint.parameters()), train, val, s2, s2["steps"], seed, deadline, history, "stage2",
                 best, loss_kind, eps, frozen=model.base)
        for p in model.base.parameters():
            p.requires_grad_(True)
        s3 = cc["stage3"]
        optimise(model, list(model.parameters()), train, val, s3, s3["steps"], seed + 1, deadline, history, "stage3",
                 best, loss_kind, eps)
    model.load_state_dict(best["state"]); prediction = predict(model, val)
    torch.save(best["state"], out / "model.pt")
    clone = build().to(device); clone.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(predict(clone, val), prediction, atol=0, rtol=0)
    info.update(selected=best["where"], train_eval_rmse_selected=NS.metric(train, predict(model, train)))
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pilot.write_history(out, history, selected=best["where"])
    return prediction, {"best_epoch": best["where"], "history": history, **info,
                        "parameters": sum(p.numel() for p in model.parameters())}


def route(tables, banks, cfg, seed, mono, out, deadline):
    del mono
    name = out.name.split("_seed_")[0]
    if not (A.fullmatch(name) or B.fullmatch(name)):
        raise ValueError(f"Unregistered JointFeature model {name}")
    search.check_reference(out, cfg, banks)
    return run(name, tables, banks, cfg, seed, out, deadline)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); output = args.output_dir.resolve()
    if not all(A.fullmatch(m) or B.fullmatch(m) for m in cfg["models"]) or cfg["fold"] != 0 \
            or cfg["claim_scope"] != "engineering_pilot_no_performance_ranking":
        raise ValueError("Unregistered or non-development JointFeature configuration")
    allowed = ROOT / "runs/model_improvement"
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Output must be a fresh experiment subdirectory")
    output.mkdir(parents=True, exist_ok=False)
    reference = ROOT / cfg["reference_run"]
    sources = [learned._file_record(reference / n) for n in ("train_ids.csv", "validation_ids.csv", "data_binding.json", "summary.json")]
    learned._write_json_atomic(output / "reference_binding.json", {"sources": sources, "written_before_response_loading": True})
    accesses = []; original = pd.read_parquet
    def guard(path, *a, **k):
        resolved = Path(path).resolve()
        if resolved.name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError(f"Forbidden response source: {resolved}")
        accesses.append(str(resolved)); return original(path, *a, **k)
    try:
        with threadpool_limits(limits=cfg["threads"]), patch.object(pd, "read_parquet", guard), patch.object(pilot, "train_graph", route):
            result = pilot.execute(output, cfg, args.config.resolve())
        for record in sources:
            if learned._file_record(Path(record["path"]))["sha256"] != record["sha256"]:
                raise RuntimeError("Reference run drifted during training")
        search.assess(output, cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    except Exception as error:
        learned._write_json_atomic(output / "failure.json", {"status": "failed", "error": repr(error)})
        learned._write_json_atomic(output / "progress.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        learned._write_json_atomic(output / "input_access.json", {"parquet_reads": accesses, "test_response_rows_read": 0})


if __name__ == "__main__":
    main()

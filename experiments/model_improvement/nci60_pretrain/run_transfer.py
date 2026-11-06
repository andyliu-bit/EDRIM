"""NCI60-Pretrain-v1 transfer runner (docs/NCI60_PRETRAIN_V1_PROTOCOL.md §4-5).

Imports experiments/model_improvement/run_sparse_mono.py UNCHANGED (data context, stage 1/2, features, metrics,
run_job) and registers:
  D_gnn_pt_<g>         MonoPriorNet drug + cell encoders initialised from the NCI-60 pretrained checkpoint of
                       architecture <g> (g1 / g2); everything else identical to D_gnn_<g>: same seed -> identical
                       head initialisation, batch order, steps, optimiser, correction MLP and FeatureTabM head.
  D_gnn_pt_frozen_<g>  as above with the drug encoder frozen (eval mode, no updates).
``PretrainedNetPrior.fit`` is a copy of ``run_sparse_mono.NetPrior.fit`` with one marked insertion.
``main`` is a copy of ``run_sparse_mono.main`` with an explicit job list (cfg["jobs"] = [pipeline, seed, k, run]),
so that reproduction controls (run="control") of existing D_gnn jobs can be scheduled and compared byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

spec = importlib.util.spec_from_file_location("sm_runner", ROOT / "experiments/model_improvement/run_sparse_mono.py")
sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)

from bisurfgnn.pretrain.nci60 import sha256_file  # noqa: E402

FROZEN_SAMPLE = {"train_ids.csv": "5752b4675ef81a9ccd021545a8d1072b965125bbfca7d67684a8e019b5b5d81e",
                 "validation_ids.csv": "eb3b88ca488f038e3792ed5cd08d547023648137bbd22be39da65449d117ca5e"}
_CKPT: dict[str, dict] = {}


def load_pretrained(entry: dict) -> dict:
    path = (ROOT / entry["path"]).resolve()
    if str(path) not in _CKPT:
        h = sha256_file(path)
        if h != entry["sha256"]:
            raise RuntimeError(f"Pretrained checkpoint hash mismatch {path}: {h}")
        _CKPT[str(path)] = torch.load(path, map_location="cpu", weights_only=True)
    return _CKPT[str(path)]


class PretrainedNetPrior(sm.NetPrior):
    def __init__(self, cfg_key, freeze_drug=False):
        super().__init__("gnn", cfg_key); self.freeze_drug = freeze_drug

    def fit(self, mono, ctx, seed):
        c = ctx.cfg["net_prior"][self.cfg_key]; dev = torch.device(ctx.cfg["device"]); self.ctx, self.dev = ctx, dev
        sm.learned._seed_everything(seed)
        self.model = sm.MonoPriorNet(self.drug_input, ctx.banks.atom_feature_dim, ctx.banks.cell_feature_dim, hidden=c["hidden"],
                                     dropout=c["dropout"], dose_mean=float(mono.dose.mean()), dose_sd=float(mono.dose.std()),
                                     mono_mean=float(mono.mono.mean())).to(dev)
        # >>> inserted (NCI60-Pretrain-v1): pretrained drug + cell encoders; head, dose stats, RNG stream untouched
        ck = load_pretrained(ctx.cfg["pretrained"][self.cfg_key])
        if ck["meta"]["hidden"] != c["hidden"] or ck["meta"]["arch"] != self.cfg_key:
            raise RuntimeError("Pretrained architecture does not match the D_gnn configuration")
        self.model.drug_encoder.load_state_dict(ck["drug_encoder"]); self.model.cell_encoder.load_state_dict(ck["cell_encoder"])
        if self.freeze_drug:
            for p in self.model.drug_encoder.parameters():
                p.requires_grad_(False)
        # <<< inserted
        self.drug_in = sm.graph.pad_graph_items(ctx.banks.drug_graphs).to(dev)
        self.cell_in = sm.graph.pad_graph_items(ctx.banks.cell_graphs).to(dev)
        d = torch.as_tensor([ctx.drug_index[x] for x in mono.drug], device=dev)
        cc = torch.as_tensor([ctx.cell_index[x] for x in mono.cell_id], device=dev)
        z = torch.as_tensor(mono.dose.to_numpy(), dtype=torch.float32, device=dev)
        y = torch.as_tensor(mono.mono.to_numpy(), dtype=torch.float32, device=dev)
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=c["lr"], weight_decay=c["weight_decay"])
        g = torch.Generator().manual_seed(seed)
        for step in range(c["steps"]):
            self.model.train()
            if self.freeze_drug:
                self.model.drug_encoder.eval()
            i = torch.randint(0, len(y), (c["batch"],), generator=g).to(dev)
            drugs, cells = self.model.encode(self.drug_in, self.cell_in)
            loss = torch.mean((self.model.predict(drugs, cells, d[i], cc[i], z[i]) - y[i]) ** 2)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite prior loss")
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step()
        self.final_train_mse = float(loss.detach())
        return self


for _g in ("g1", "g2"):
    sm.PIPELINES[f"D_gnn_pt_{_g}"] = (lambda g=_g: PretrainedNetPrior(g), sm.MLPCorrection, sm.head_tabm)
    sm.PIPELINES[f"D_gnn_pt_frozen_{_g}"] = (lambda g=_g: PretrainedNetPrior(g, freeze_drug=True), sm.MLPCorrection, sm.head_tabm)

CODE = ["experiments/model_improvement/run_sparse_mono.py", "experiments/model_improvement/nci60_pretrain/run_transfer.py",
        "experiments/model_improvement/analysis/diagnose_fit_vs_generalization.py", "experiments/model_improvement/run_algorithm_search.py",
        "experiments/model_improvement/run_algorithm_search_b2.py", "experiments/model_improvement/run_comparator_pilot.py",
        "src/bisurfgnn/models/sparse_mono.py", "src/bisurfgnn/sparse_mono/task.py", "src/bisurfgnn/models/graph_encoder.py",
        "src/bisurfgnn/pretrain/nci60.py", "src/bisurfgnn/pretrain/trainer.py"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs/model_improvement/nci60_pretrain_v1"):
        raise ValueError("Output must be a fresh subdirectory of runs/model_improvement/nci60_pretrain_v1")
    output.mkdir(parents=True)
    (output / "code_manifest.json").write_text(json.dumps({p: sha256_file(ROOT / p) for p in CODE}, indent=1), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(cfg["threads"])
    for name, h in FROZEN_SAMPLE.items():
        got = sha256_file(Path(cfg["reference_run"]) / name)
        if got != h:
            raise RuntimeError(f"Frozen sample drift {name}: {got}")
    accesses = []; original = pd.read_parquet
    def guard(path, *a, **kw):
        resolved = Path(path).resolve()
        if resolved.name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError(f"Forbidden response source: {resolved}")
        accesses.append(str(resolved)); return original(path, *a, **kw)
    results = []
    try:
        with patch.object(pd, "read_parquet", guard):
            ctx = sm.Context(cfg)
            checks = {}
            for gk, entry in cfg.get("pretrained", {}).items():
                meta = load_pretrained(entry)["meta"]
                ok = all(meta["cell_graph_hashes"].get(b) == h for b, h in zip(ctx.banks.biological_cell_ids, ctx.banks.cell_graph_hashes))
                if not ok or meta["atom_feature_dim"] != ctx.banks.atom_feature_dim:
                    raise RuntimeError(f"Pretraining inputs differ from the SparseMono banks ({gk})")
                checks[gk] = dict(cell_graphs_identical=True, bank_cells=len(ctx.banks.cell_ids), best_step=meta["best_step"],
                                  best_holdout_rmse=meta["best_holdout_rmse"], exclusion_sha256=meta["exclusion_sha256"])
            manifest = {f"{r}_k{k}": dict(view_sha256=v.sha256(), points=len(v.points), primary_query=int(v.points.primary_query.sum()),
                                           visible=int(v.points.visible_flag.sum()), supports=len(v.supports))
                        for (r, k), v in ctx.views.items()}
            manifest["pretrained_checks"] = checks
            for (r, k), v in ctx.views.items():
                if k == 1:
                    v.supports.to_csv(output / f"supports_{r}.csv", index=False)
            (output / "task_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            primary = {}
            for n, s, k, run in [tuple(j) for j in cfg["jobs"]]:
                out = output / f"{n}_k{k}_seed_{s}_{run}"; out.mkdir()
                r = sm.run_job(ctx, n, s, k, out); r["run"] = run; results.append(r)
                pred = pd.read_csv(out / "predictions.csv").prediction.to_numpy()
                if run == "primary":
                    primary[(n, s, k)] = pred
                elif run == "repeat":
                    diff = float(np.max(np.abs(primary[(n, s, k)] - pred))); r["repeat_max_abs_diff"] = diff
                    if diff > 1e-7:
                        raise RuntimeError(f"Repeat drift {n} {diff}")
                elif run == "control":
                    ref_dir = ROOT / cfg["control_reference"][n] / f"{n}_k{k}_seed_{s}_primary"
                    ref = pd.read_csv(ref_dir / "predictions.csv")
                    r["control_max_abs_diff"] = float(np.max(np.abs(ref.prediction.to_numpy() - pred)))
                    r["control_reference_primary_rmse"] = json.loads((ref_dir / "metrics.json").read_text())["primary_rmse"]
                pd.DataFrame(results).to_csv(output / "results.csv", index=False)
                print(f"DONE {n} k={k} seed={s} {run} primary={r['primary_rmse']:.6f} recon={r['recon_rmse_hidden_query']:.5f}"
                      + (f" control_diff={r['control_max_abs_diff']:.3g}" if run == "control" else ""), flush=True)
        (output / "summary.json").write_text(json.dumps(dict(status="passed", test_response_rows_read=0, jobs=len(results)), indent=1), encoding="utf-8")
    except Exception as error:
        (output / "failure.json").write_text(json.dumps(dict(status="failed", error=repr(error))), encoding="utf-8"); raise
    finally:
        (output / "input_access.json").write_text(json.dumps(dict(parquet_reads=accesses, test_response_rows_read=0), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

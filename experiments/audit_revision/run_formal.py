"""Prepared-only by default: freeze, worker, dispatch, aggregate and preflight.

Full launch requires the explicit `launch` command and a passing source-bound
readiness report. Preflight uses the same worker but a declared small cohort.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

import bisurfgnn.learned_benchmark as learned
import bisurfgnn.independent_graph_benchmark as graph
from bisurfgnn.benchmark_config import load_benchmark_config
from bisurfgnn.data.benchmark_context import load_benchmark_cell_context
from bisurfgnn.data.cellminer import load_cellminer_config
from bisurfgnn.data.chemical_splits import load_one_new_scaffold_benchmark
from bisurfgnn.models.factorial_bisurf import FactorialBiSurfGNN, VARIANTS as GRAPH
from bisurfgnn.models.baselines import MorganHallmarkResidualMLP
from bisurfgnn.models.independent_graph import CompactIndependentPooledGNN
from bisurfgnn import lightweight_controls as light
from bisurfgnn.streaming_ridge import fit_streaming, predict_streaming
from bisurfgnn.resumable_training import train_resumable, _latest, exclusive_job
from bisurfgnn.process_dispatch import dispatch
from bisurfgnn.launch_plan import make_inventory, verify_receipts, validate_paired_points
from bisurfgnn.factorial_statistics import factorial_bootstrap
from bisurfgnn.paired_statistics import paired_block_bootstrap
from run_factorial_smoke import sample_groups, guarded_read

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/"configs/audit_revision/formal_launch_v1.yaml"
CACHE = {}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seal(path, payload):
    data = dict(payload)
    data["manifest_sha256"] = learned._canonical_hash(data)
    learned._write_json_exclusive(path, data)
    return data


def check_seal(data):
    if data.get("manifest_sha256") != learned._canonical_hash({k:v for k,v in data.items() if k != "manifest_sha256"}):
        raise ValueError("Manifest self hash mismatch")


def verify(records):
    for record in records:
        if learned._file_record(Path(record["path"])) != record:
            raise ValueError(f"Bound artifact drift: {record['path']}")


def source_paths():
    return sorted((ROOT/"src/bisurfgnn").rglob("*.py")) + [Path(__file__),
        ROOT/"experiments/audit_revision/complete_prelaunch.py",
        ROOT/"experiments/audit_revision/run_factorial_smoke.py", CONFIG,
        ROOT/"configs/audit_revision/bisurf_mono_v2.yaml", ROOT/"configs/cell_context.yaml",
        ROOT/"docs/PRELAUNCH_PROTOCOL.md"]


def configure_cuda():
    if not torch.cuda.is_available() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUDA and CUBLAS_WORKSPACE_CONFIG=:4096:8 are required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prepare(output, *, preflight=False):
    output = Path(output).resolve()
    allowed = (ROOT/"runs/audit_revision").resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Fresh output child of runs/audit_revision required")
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    inventory = make_inventory(config)
    if preflight:
        inventory = {key:[j for j in rows if j["fold"] == 0 and j["seed"] == 41] for key, rows in inventory.items()}
        config.update(max_epochs=3, min_epochs=3, patience=3, checkpoint_stride=2,
                      batch_surfaces=32, evaluation_batch_points=512, isotime_seconds=.05,
                      bootstrap_iterations=1000)
    elif shutil.disk_usage(output).free < config["estimated_run_bytes"]+config["disk_reserve_bytes"]:
        raise RuntimeError("Insufficient disk for frozen run estimate plus reserve")
    sources, snapshots = [], []
    for source in source_paths():
        record = learned._file_record(source)
        dest = output/"source_snapshot"/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        sources.append(record)
        snapshots.append(learned._file_record(dest))
        if sources[-1]["sha256"] != snapshots[-1]["sha256"]:
            raise RuntimeError("Snapshot mismatch")
    benchmark = load_benchmark_config(ROOT/config["benchmark_config"])
    split = load_one_new_scaffold_benchmark(benchmark)
    shards = learned.load_learned_point_shards(benchmark, verify_roles=("train", "validation"))
    inputs = {str(fold): {role: learned._file_record(shards.role_path(fold, role))
              for role in ("train", "validation")} for fold in sorted({j["fold"] for j in inventory["jobs"]})}
    return seal(output/"plan.json", dict(version="formal_worker_plan_v1", preflight=preflight,
        config=config, inventory=inventory, sources=sources, snapshots=snapshots, inputs=inputs,
        split_hash=split.split_hash, endpoint=benchmark.endpoint,
        allowed_response_roles=["train", "validation"], full_test_exposure_disclaimer="development cohort reused across folds",
        runtime_packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}))


def load_data(plan, fold):
    key = (plan["manifest_sha256"], fold)
    if key in CACHE:
        return CACHE[key]
    benchmark = load_benchmark_config(ROOT/plan["config"]["benchmark_config"])
    split = load_one_new_scaffold_benchmark(benchmark)
    context = load_benchmark_cell_context(benchmark, load_cellminer_config(ROOT/"configs/cell_context.yaml"))
    if split.split_hash != plan["split_hash"] or context.split_hash != split.split_hash:
        raise ValueError("Frozen split/context mismatch")
    shards = learned.load_learned_point_shards(benchmark, verify_roles=("train", "validation"))
    frames = {role: learned.load_learned_role_frame(shards, split.surfaces, fold, role, sample_seed=41)
              for role in ("train", "validation")}
    if plan["preflight"]:
        frames = {role:sample_groups(frame, 900 if role == "train" else 1800, seed=41, salt="P1-v1")
                  for role, frame in frames.items()}
    if set(frames["train"].prediction_group_id) & set(frames["validation"].prediction_group_id):
        raise ValueError("Group leakage")
    assignment = learned._fold_surface_roles(split.surfaces, fold)
    drugs, cells = sorted(set(assignment.drug_a_id)|set(assignment.drug_b_id)), sorted(set(assignment.cell_id))
    mbanks = learned.build_feature_banks(drugs, cells, context, fold, benchmark)
    gbanks = graph.build_graph_feature_banks(drugs, cells, context, fold, benchmark)
    tables = {role:learned.build_compact_point_table(frame, mbanks, role=role, fold=fold, monotherapy_upper_bound=3)
              for role,frame in frames.items()}
    if mbanks.drug_ids != gbanks.drug_ids or mbanks.cell_ids != gbanks.cell_ids:
        raise ValueError("Graph/Morgan entity order mismatch")
    CACHE[key] = (mbanks, gbanks, tables, split.surfaces)
    return CACHE[key]


def worker(plan_path, job_id):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    check_seal(plan)
    job = next((j for j in plan["inventory"]["jobs"] if j["id"] == job_id), None)
    if job is None:
        raise ValueError("Job is outside frozen inventory")
    verify(plan["sources"]+plan["snapshots"]+list(plan["inputs"][str(job["fold"])].values()))
    configure_cuda()
    config = plan["config"]
    out = plan_path.parent/"jobs"/job_id
    out.mkdir(parents=True, exist_ok=True)
    if (out/"receipt.json").exists():
        raise ValueError("Job receipt exists; use dispatcher verification, not overwrite")
    if any((out/name).exists() for name in ("selected.pt","prediction.parquet","errors.parquet")):
        raise ValueError("Unreceipted publication artifacts require manual inspection")
    if shutil.disk_usage(out).free < config["disk_reserve_bytes"]:
        raise RuntimeError("Disk reserve reached")
    accesses = []
    with patch.object(pd, "read_parquet", guarded_read(pd.read_parquet, accesses)), exclusive_job(out):
        mbanks, gbanks, tables, surfaces = load_data(plan, job["fold"])
        train, val = tables["train"], tables["validation"]
        anchor = np.minimum(val.mono_a.numpy(), val.mono_b.numpy())
        anchor_score = learned.surface_macro_rmse(val.target.numpy(), anchor, val.surface_ids)
        learned._seed_everything(job["seed"])
        started = perf_counter()
        torch.cuda.reset_peak_memory_stats()
        if job["model"] in light.VARIANTS:
            features = light.fit_feature_state(train, mbanks, seed=job["seed"])
            state = fit_streaming(train, mbanks, features, job["model"], alpha=config["ridge_alpha"],
                batch_points=config["ridge_batch_points"], device="cuda", max_working_bytes=config["ridge_max_working_bytes"])
            fitted = predict_streaming(val, mbanks, features, job["model"], state)
            if plan["preflight"]:
                dense = light.ridge_fit(light.feature_matrix(train, mbanks, features, job["model"]), train, alpha=config["ridge_alpha"])
                oracle = light.ridge_predict(light.feature_matrix(val, mbanks, features, job["model"]), val, dense)
                np.testing.assert_allclose(fitted, oracle, atol=1e-7, rtol=0)
            fitted_score = learned.surface_macro_rmse(val.target.numpy(), fitted, val.surface_ids)
            use_anchor = not fitted_score < anchor_score-config["min_delta"]
            prediction = anchor if use_anchor else fitted
            checkpoint = out/"selected.pt"
            learned._write_checkpoint_atomic(checkpoint, dict(feature_state=features, fitted_state=state, use_anchor=use_anchor))
            loaded = torch.load(checkpoint, weights_only=True, map_location="cpu")
            np.testing.assert_allclose(fitted, predict_streaming(val, mbanks, loaded["feature_state"], job["model"],
                                      loaded["fitted_state"]), atol=1e-10, rtol=0)
            selection = dict(best_epoch=None, candidate="anchor" if use_anchor else "fitted_ridge",
                best_score=anchor_score if use_anchor else fitted_score, normal_error=state["normal_equation_relative_error"],
                estimated_working_bytes=state["estimated_working_bytes"], finished=True)
        else:
            if job["model"] == "morgan_mlp":
                model = MorganHallmarkResidualMLP(2048, 100, 256, dropout=.1, reference_mode="observed_hsa", monotherapy_upper_bound=3)
                bank, forward, predict_fn = mbanks, learned._model_forward, learned.predict_compact
            else:
                if job["model"] == "independent_pooled_gnn":
                    model = CompactIndependentPooledGNN(62, 53, 171, dropout=.1, reference_mode="observed_hsa", monotherapy_upper_bound=3)
                else:
                    exchange, fusion = job["model"].split("__")
                    model = FactorialBiSurfGNN(62, 53, job["width"], exchange=exchange, fusion=fusion)
                bank, forward, predict_fn = gbanks, graph._model_forward, graph.predict_graph_compact
            model.to("cuda")
            learned._seed_everything(job["seed"])
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                lr=config["learning_rate"], weight_decay=config["weight_decay"])
            def predict():
                return predict_fn(model, val, bank, batch_size=config["evaluation_batch_points"], device=torch.device("cuda"))[0]
            np.testing.assert_allclose(predict(), anchor, atol=3.1e-6, rtol=2e-6)
            trained = [False]
            def evaluate(model):
                return learned.surface_macro_rmse(val.target.numpy(), predict(), val.surface_ids) if trained[0] else anchor_score
            def epoch(model, optimizer, epoch_index, remaining_seconds=None):
                trained[0] = True
                model.train()
                numerator, count, elapsed, max_batch, updates = 0.0, 0, 0.0, 0.0, 0
                batches = graph._surface_batch_indices(train, config["batch_surfaces"], shuffle_seed=job["seed"]*1_000_003+epoch_index)
                for indices in batches:
                    torch.cuda.synchronize()
                    tick = perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    output = forward(model, train, bank, indices, torch.device("cuda"))
                    ids = train.surface_ids[indices.numpy()]
                    loss = learned._surface_mean_mse(output.prediction, train.target[indices].to("cuda"), ids)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise RuntimeError("Non-finite NN update")
                    optimizer.step()
                    torch.cuda.synchronize()
                    duration = perf_counter()-tick
                    elapsed += duration
                    max_batch = max(max_batch, duration)
                    n = len(set(ids)); numerator += float(loss.detach())*n; count += n; updates += 1
                    if remaining_seconds is not None and elapsed >= remaining_seconds:
                        break
                return dict(surface_mse=numerator/count, surfaces=count, updates=updates,
                    partial_epoch=updates < len(batches), training_seconds=elapsed, max_batch_seconds=max_batch)
            training = {k:config[k] for k in ("max_epochs", "min_epochs", "patience", "min_delta", "checkpoint_stride")}
            if job["track"] == "isotime":
                training.update(max_epochs=config["isotime_max_epochs"], min_epochs=1,
                    checkpoint_stride=config["isotime_max_epochs"], max_train_seconds=config["isotime_seconds"])
            contract = dict(identity=dict(job=job, plan_hash=plan["manifest_sha256"],
                table_hashes={r:learned._point_table_hash(t) for r,t in tables.items()},
                graph_bank_hash=graph._graph_bank_hash(gbanks), morgan_bank_hash=learned._feature_bank_hash(mbanks)),
                files=plan["sources"]+list(plan["inputs"][str(job["fold"])].values()), training=training)
            result = train_resumable(model, optimizer, out/"epochs", contract, epoch, evaluate)
            if job["track"] == "isotime" and result["training_seconds"] < config["isotime_seconds"]:
                raise RuntimeError("Safety epoch cap reached before isotime budget")
            model.load_state_dict(result["best_model"], strict=True)
            prediction = anchor if result["best_epoch"] == -1 else predict()
            checkpoint = out/"selected.pt"
            learned._write_checkpoint_atomic(checkpoint, dict(state_dict=result["best_model"], best_epoch=result["best_epoch"]))
            loaded = torch.load(checkpoint, weights_only=True, map_location="cpu")
            model.load_state_dict(loaded["state_dict"], strict=True)
            np.testing.assert_allclose(prediction, anchor if loaded["best_epoch"] == -1 else predict(), atol=1e-7, rtol=0)
            selection = {k:result[k] for k in ("best_epoch", "best_score", "completed_epochs", "training_seconds", "history", "finished")}
            selection["parameters_total"] = sum(p.numel() for p in model.parameters())
            selection["parameters_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not np.isfinite(prediction).all():
            raise RuntimeError("Non-finite predictions")
        score = learned.surface_macro_rmse(val.target.numpy(), prediction, val.surface_ids)
        np.testing.assert_allclose(score, selection["best_score"], atol=2e-6, rtol=0)
        point = pd.DataFrame(dict(fold=job["fold"], seed=job["seed"], model=job["model"], source_id=val.source_rows,
            surface_id=val.surface_ids, group_id=val.group_ids, cluster_id=val.cluster_ids,
            assay_block_id=pd.Series(val.surface_ids).map(surfaces.set_index("surface_replicate_id").assay_block_id).to_numpy(),
            target=val.target.numpy(), prediction=prediction))
        if point.isna().any().any():
            raise RuntimeError("Missing point metadata")
        def errors(frame):
            error = frame.assign(sq=(frame.target.astype(float)-frame.prediction.astype(float))**2).groupby(["fold","seed","model","surface_id"]).agg(
                group_id=("group_id","first"), cluster_id=("cluster_id","first"), assay_block_id=("assay_block_id","first"),
                n_points=("source_id","size"), rmse=("sq","mean")).reset_index()
            error["rmse"] = np.sqrt(error.rmse)
            return error
        error = pd.concat([errors(point), errors(point.assign(model="hsa", prediction=anchor))], ignore_index=True)
        artifacts = {}
        for key, frame in (("prediction",point), ("errors",error)):
            dest = out/f"{key}.parquet"
            if dest.exists():
                raise ValueError("Orphan publication file requires review")
            frame.to_parquet(dest, index=False)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(dest))
            artifacts[key] = learned._file_record(dest)
        verify(plan["sources"]+list(plan["inputs"][str(job["fold"])].values()))
        receipt = seal(out/"receipt.json", dict(job=job, plan_hash=plan["manifest_sha256"], finished=selection["finished"],
            **artifacts, checkpoint=learned._file_record(checkpoint), selection=selection,
            elapsed_seconds=perf_counter()-started, cuda_peak_bytes=torch.cuda.max_memory_allocated(),
            table_hashes={r:learned._point_table_hash(t) for r,t in tables.items()},
            test_response_files_read=0, parquet_read_trace=accesses, pid=os.getpid()))
    print(f"Completed {job_id}", flush=True)
    return receipt


def aggregate(plan_path):
    plan_path = Path(plan_path); plan = read_json(plan_path); check_seal(plan)
    expected_paths = {plan_path.parent/"jobs"/j["id"]/"receipt.json" for j in plan["inventory"]["jobs"]}
    if set((plan_path.parent/"jobs").glob("*/receipt.json")) != expected_paths:
        raise ValueError("Missing or extra receipt files in matrix")
    receipts = [read_json(plan_path.parent/"jobs"/j["id"]/"receipt.json") for j in plan["inventory"]["jobs"]]
    verify_receipts(plan["inventory"], receipts, plan["manifest_sha256"])
    by_id = {r["job"]["id"]:r for r in receipts}
    collected, seen = [], {}
    for job in plan["inventory"]["jobs"]+plan["inventory"]["aliases"]:
        receipt = by_id[job.get("source",job["id"])]
        point = pd.read_parquet(receipt["prediction"]["path"])
        if set(point.fold) != {job["fold"]} or set(point.seed) != {job["seed"]} or set(point.model) != {job["model"]}:
            raise ValueError("Point file job metadata mismatch")
        key = job["fold"]
        if key in seen:
            validate_paired_points([seen[key],point])
        else:
            seen[key] = point
        errors = pd.read_parquet(receipt["errors"]["path"]).assign(track=job["track"])
        collected.append(errors)
    ledger = pd.concat(collected, ignore_index=True)
    hsa = ledger[ledger.model == "hsa"]
    if hsa.groupby(["track","fold","seed","surface_id"]).rmse.nunique().max() != 1:
        raise ValueError("HSA differs across workers")
    ledger = pd.concat([ledger[ledger.model != "hsa"], hsa.drop_duplicates(["track","fold","seed","surface_id"])])
    results = {}
    for track in ("main","capacity","isotime"):
        part = ledger[ledger.track == track].drop(columns="track")
        results[track] = dict(factorial={unit:factorial_bootstrap(part, iterations=plan["config"]["bootstrap_iterations"], unit=unit)
                                       for unit in ("cluster_id","group_id")},
            versus_hsa=paired_block_bootstrap(part, iterations=plan["config"]["bootstrap_iterations"]))
        results[track]["attention_diagnostics"] = {fusion:paired_block_bootstrap(
            part[part.model.isin([f"bixt__{fusion}",f"cross_attention__{fusion}"])],
            reference=f"cross_attention__{fusion}", iterations=plan["config"]["bootstrap_iterations"])
            for fusion in ("concat","gate")}
    dest = plan_path.parent/"paired_errors.parquet"
    if dest.exists():
        raise ValueError("Aggregate already exists")
    ledger.to_parquet(dest,index=False)
    return seal(plan_path.parent/"aggregate.json", dict(plan_hash=plan["manifest_sha256"],
        error_ledger=learned._file_record(dest), receipts=[learned._file_record(plan_path.parent/"jobs"/j["id"]/"receipt.json") for j in plan["inventory"]["jobs"]],
        job_count=len(receipts), aliases=len(plan["inventory"]["aliases"]), statistics=results,
        claim_scope="engineering_only" if plan["preflight"] else "conditional_development_results_not_confirmation"))


def launch(plan_path, readiness):
    plan_path=Path(plan_path).resolve(); plan=read_json(plan_path); check_seal(plan)
    ready=read_json(readiness); check_seal(ready)
    if plan["preflight"] or ready["status"] != "passed":
        raise ValueError("Explicit full plan and passing readiness required")
    verify(plan["sources"]+ready["sources"]+ready["evidence"])
    if plan["sources"] != ready["sources"]:
        raise ValueError("Prepared plan is not the source version that passed readiness")
    if shutil.disk_usage(plan_path.parent).free < plan["config"]["estimated_run_bytes"]+plan["config"]["disk_reserve_bytes"]:
        raise RuntimeError("Full-run disk budget unavailable")
    jobs=[dict(id=j["id"], command=[sys.executable,str(Path(__file__).resolve()),"worker","--plan",str(plan_path),"--job-id",j["id"]],
        receipt=str(plan_path.parent/"jobs"/j["id"]/"receipt.json"), artifact_keys=["prediction","errors","checkpoint"])
        for j in plan["inventory"]["jobs"]]
    dispatch(plan_path.parent, jobs, plan["sources"],
             dict(PYTHONPATH=str(ROOT/"src"),PYTHONIOENCODING="utf-8",CUBLAS_WORKSPACE_CONFIG=":4096:8"),
             timeout_seconds=plan["config"]["worker_timeout_seconds"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare","worker","aggregate","launch"))
    parser.add_argument("--output-dir",type=Path); parser.add_argument("--plan",type=Path)
    parser.add_argument("--job-id"); parser.add_argument("--readiness",type=Path)
    args=parser.parse_args()
    if args.action=="prepare": prepare(args.output_dir)
    elif args.action=="worker": worker(args.plan,args.job_id)
    elif args.action=="aggregate": aggregate(args.plan)
    elif args.action=="launch": launch(args.plan,args.readiness)


if __name__=="__main__": main()

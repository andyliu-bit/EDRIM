"""Inference-only input sensitivity; output anchor stays at its observed value."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import t
from threadpoolctl import threadpool_limits
from bisurfgnn import independent_graph_benchmark as graph, learned_benchmark as learned
from bisurfgnn.models.compact_bisurf import CompactBiSurfGNN
from bisurfgnn.models.baselines import _anchored_output

spec = importlib.util.spec_from_file_location("input_path_pilot", Path(__file__).with_name("run_comparator_pilot.py"))
pilot = importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
CONFIG = ROOT / "configs/model_improvement/input_diagnostic_v1.yaml"
OUTPUT = ROOT / "runs/model_improvement/input_diagnostic_v1"


def diagnostic_forward(model, bank, contexts, a, b, cells, doses, mono_a, mono_b,
                       *, decoder_doses=None, decoder_mono_a=None, decoder_mono_b=None,
                       zero_residual=False):
    tokens, mask = bank
    pair, arm_a, arm_b = model.exchange_unique_pairs(tokens, mask, a, b)
    context = contexts[cells.long()]
    pair = model.context_fusion(pair, context)
    delta, _ = model.interaction_decoder(arm_a, arm_b, context, pair,
        doses if decoder_doses is None else decoder_doses,
        mono_a=mono_a if decoder_mono_a is None else decoder_mono_a,
        mono_b=mono_b if decoder_mono_b is None else decoder_mono_b)
    flags = torch.ones(len(a), dtype=torch.bool, device=doses.device)
    prediction = _anchored_output(doses, mono_a, mono_b, flags, flags,
        torch.zeros_like(delta) if zero_residual else delta,
        model.reference_mode, model.monotherapy_upper_bound)
    return prediction, delta


def permute_bank(bank, order):
    return tuple(value[order] for value in bank)


def predict(model, table, bank, contexts, cfg, *, condition="normal", drug_order=None,
            cell_order=None, point_order=None):
    device = torch.device(cfg["device"])
    if condition == "drug_structure":
        bank = permute_bank(bank, drug_order.to(device))
    if condition == "cell_context":
        contexts = contexts[cell_order.to(device)]
    values, deltas = [], []
    with torch.no_grad():
        for ix in learned._batch_indices(len(table), cfg["evaluation_batch_size"], shuffle_seed=None):
            donor = point_order[ix] if point_order is not None else ix
            kwargs = {}
            if condition == "decoder_monotherapy":
                kwargs = dict(decoder_mono_a=table.mono_a[donor].to(device),decoder_mono_b=table.mono_b[donor].to(device))
            elif condition == "decoder_dose":
                kwargs = dict(decoder_doses=table.doses[donor].to(device))
            result, delta = diagnostic_forward(model, bank, contexts, table.drug_a_index[ix].to(device),
                table.drug_b_index[ix].to(device), table.cell_index[ix].to(device), table.doses[ix].to(device),
                table.mono_a[ix].to(device), table.mono_b[ix].to(device),
                zero_residual=condition == "zero_residual", **kwargs)
            values.append(result.prediction.cpu().numpy()); deltas.append(delta.cpu().numpy())
    return np.concatenate(values), np.concatenate(deltas)


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")); start = time.monotonic()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    reference = ROOT / cfg["reference_run"]
    launch = json.loads((reference/"launch.json").read_text()); data = launch["config"]
    binding = json.loads((reference/"data_binding.json").read_text())
    records = [learned._file_record(CONFIG),learned._file_record(Path(__file__))] + launch["sources"] + binding["inputs"]
    records += [learned._file_record(reference/f"bisurf_mono_v2_seed_{seed}_primary/model.pt") for seed in cfg["seeds"]]
    for record in records:
        if learned._file_record(Path(record["path"]))["sha256"] != record["sha256"]:
            raise RuntimeError("Reference/input/source drift before diagnostic")
    learned._write_json_atomic(OUTPUT/"binding.json",dict(config=cfg,records=records,written_before_response_loading=True))
    shutil.copyfile(CONFIG,OUTPUT/"config.yaml")
    accesses=[]; original=pd.read_parquet
    def guard(path,*args,**kwargs):
        resolved=Path(path).resolve()
        if resolved.name.lower() in {"test.parquet","points.parquet"}: raise RuntimeError("Forbidden response source")
        accesses.append(str(resolved)); return original(path,*args,**kwargs)
    try:
        with threadpool_limits(limits=cfg["threads"]), patch.object(pd,"read_parquet",guard):
            torch.set_num_threads(cfg["threads"]); torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            root=(ROOT/data["data_root"]).resolve()
            benchmark=pilot.load_benchmark_config(root/data["benchmark_config"])
            cell_cfg=pilot.load_cellminer_config(root/data["cell_config"])
            split=pilot.load_one_new_scaffold_benchmark(benchmark)
            context=pilot.load_benchmark_cell_context(benchmark,cell_cfg)
            shards=learned.load_learned_point_shards(benchmark,verify_roles=("train","validation"))
            full={role:learned.load_learned_role_frame(shards,split.surfaces,0,role,sample_seed=data["sampling_seed"]) for role in ("train","validation")}
            frames=pilot.sample_supported_frames(full["train"],full["validation"],split.surfaces,data); del full
            audit=pilot.audit_sample(frames,split.surfaces)
            if not all(audit["checks"].values()): raise RuntimeError("Split audit failed")
            learned._write_json_atomic(OUTPUT/"split_audit.json",audit)
            for role,frame in frames.items():
                ids=frame[["source_combodrugseq","prediction_group_id","surface_replicate_id"]]
                ids.to_csv(OUTPUT/f"{role}_ids.csv",index=False)
                if learned._file_record(OUTPUT/f"{role}_ids.csv")["sha256"]!=learned._file_record(reference/f"{role}_ids.csv")["sha256"]: raise RuntimeError("Frozen IDs/order differ")
            both=pd.concat(frames.values()); drugs=sorted(set(both.drug_a_id)|set(both.drug_b_id));cells=sorted(set(both.cell_id))
            banks=graph.build_graph_feature_banks(drugs,cells,context,0,benchmark)
            if banks.context_state_hash!=binding["context_state_hash"]: raise RuntimeError("Cell fitted state drift")
            table=learned.build_compact_point_table(frames["validation"],banks,role="validation",fold=0,monotherapy_upper_bound=3.)
            maps={}
            for seed in cfg["permutation_seeds"]:
                rng=np.random.default_rng(seed)
                maps[seed]=dict(drug_order=torch.tensor(rng.permutation(len(drugs))),cell_order=torch.tensor(rng.permutation(len(cells))),point_order=torch.tensor(rng.permutation(len(table))))
            np.savez(OUTPUT/"permutation_maps.npz",**{f"{s}_{k}":v.numpy() for s,m in maps.items() for k,v in m.items()})
            jobs=[(s,c,p) for s in cfg["seeds"] for c in cfg["conditions"] for p in cfg["permutation_seeds"]]
            pd.DataFrame(jobs,columns=["seed","condition","permutation_seed"]).assign(status="pending").to_csv(OUTPUT/"planned_tasks.csv",index=False)
            rows=[]; completed=0; repeat=0.
            for seed in cfg["seeds"]:
                model=CompactBiSurfGNN(banks.atom_feature_dim,banks.cell_feature_dim,136,graph_layers=2,attention_heads=4,dropout=.1,monotherapy_upper_bound=3.,residual_monotherapy_input=True).to(cfg["device"]).eval()
                model.load_state_dict(torch.load(reference/f"bisurf_mono_v2_seed_{seed}_primary/model.pt",weights_only=True,map_location=cfg["device"]))
                with torch.no_grad():
                    bank=model.encode_drugs(graph.pad_graph_items(banks.drug_graphs).to(cfg["device"]))
                    ctx=model.encode_context(graph.pad_graph_items(banks.cell_graphs).to(cfg["device"]))
                normal,delta=predict(model,table,bank,ctx,cfg)
                saved=pd.read_csv(reference/f"bisurf_mono_v2_seed_{seed}_primary/predictions.csv")
                np.testing.assert_allclose(normal,saved.prediction.to_numpy(),rtol=0,atol=1e-7)
                score=pilot.metric(table.target.numpy(),normal,table.surface_ids)["surface_macro_rmse"]
                if seed==cfg["seeds"][0]:
                    repeated,_=predict(model,table,bank,ctx,cfg);repeat=float(np.abs(normal-repeated).max())
                    if repeat!=0: raise RuntimeError("Repeated normal predictions differ")
                pd.DataFrame(dict(source_row=table.source_rows,surface_id=table.surface_ids,target=table.target.numpy(),prediction=normal)).to_csv(OUTPUT/f"seed_{seed}_normal.csv",index=False)
                scenarios=[("zero_residual",None,{})]+[(c,p,maps[p]) for c in cfg["conditions"] for p in cfg["permutation_seeds"]]
                for condition,p,options in scenarios:
                    if time.monotonic()-start>cfg["max_wall_seconds"]: raise TimeoutError("Frozen diagnostic budget exhausted")
                    predicted,perturbed_delta=predict(model,table,bank,ctx,cfg,condition=condition,**options)
                    metric=pilot.metric(table.target.numpy(),predicted,table.surface_ids)
                    row=dict(seed=seed,condition=condition,permutation_seed=p,baseline_rmse=score,
                        surface_macro_rmse=metric["surface_macro_rmse"],relative_error_increase_percent=100*(metric["surface_macro_rmse"]-score)/score,
                        prediction_change_rmse=float(np.sqrt(np.mean((predicted-normal)**2))),mean_abs_logit_change=float(np.mean(np.abs(perturbed_delta-delta))) if condition!="zero_residual" else float(np.mean(np.abs(delta))))
                    rows.append(row);completed+=1
                    pd.DataFrame(rows).to_csv(OUTPUT/"results.csv",index=False)
                    learned._write_json_atomic(OUTPUT/"progress.json",dict(status="running",completed=completed,total=65,seed=seed,condition=condition))
                    np.savez_compressed(OUTPUT/f"seed_{seed}_{condition}_{p}.npz",prediction=predicted,delta=perturbed_delta)
                    print(json.dumps(row),flush=True)
            frame=pd.DataFrame(rows)
            by_seed=frame.groupby(["condition","seed"],as_index=False).mean(numeric_only=True)
            summary=[]
            for condition,g in by_seed.groupby("condition"):
                x=g.relative_error_increase_percent;radius=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
                summary.append(dict(condition=condition,seeds=len(x),mean_relative_error_increase_percent=x.mean(),sd=x.std(ddof=1),seed_ci95_low=x.mean()-radius,seed_ci95_high=x.mean()+radius,mean_prediction_change_rmse=g.prediction_change_rmse.mean(),mean_abs_logit_change=g.mean_abs_logit_change.mean()))
            summary=pd.DataFrame(summary);summary.to_csv(OUTPUT/"summary.csv",index=False)
            summary.to_csv(ROOT/"reports/tables/input_diagnostic_v1_summary.csv",index=False)
            for record in records:
                if learned._file_record(Path(record["path"]))["sha256"]!=record["sha256"]: raise RuntimeError("Input/source changed during diagnosis")
            learned._write_json_atomic(OUTPUT/"summary.json",dict(status="passed",evaluations=len(rows),baseline_repeat_max_difference=repeat,wall_seconds=time.monotonic()-start,test_response_rows_read=0,scope=cfg["scope"],ci_scope="seed_variation_conditional_on_development_sample_and_fixed_permutation_maps"))
            learned._write_json_atomic(OUTPUT/"progress.json",dict(status="passed",completed=completed,total=65))
            print(summary.to_string(index=False),flush=True)
    except Exception as error:
        learned._write_json_atomic(OUTPUT/"failure.json",dict(status="failed",error=repr(error)))
        learned._write_json_atomic(OUTPUT/"progress.json",dict(status="failed",error=repr(error)))
        raise
    finally:
        learned._write_json_atomic(OUTPUT/"input_access.json",dict(parquet_reads=accesses,test_response_rows_read=0))


if __name__=="__main__": main()

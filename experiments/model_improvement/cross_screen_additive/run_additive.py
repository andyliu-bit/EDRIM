"""CrossScreenAdditive-v1 runner (docs/CROSS_SCREEN_ADDITIVE_V1_PROTOCOL.md sections 2-4).

The only change from CrossScreenMono-v1 is the output map of the BiSurfGNN v2 network:
  bisurf_mono_v2        CompactBiSurfGNN, anchored output 3*sigmoid(logit(r/3) + gate*z)
  bisurf_additive_v2    AdditiveCompactBiSurfGNN, additive output r + gate*z (same network and initialisation)
Both are trained by the frozen AlgorithmSearch-v1 `run_algorithm_search.train_variant`; the additive variant
is registered for this run only (patched VARIANTS / build_model), so no existing file is modified.

Information settings (CrossScreenMono-v1 data path, unchanged):
  cs   CS-covered frames whose monotherapy columns are REPLACED by cross-screen values (input and HSA anchor)
  sp   full frozen sample with same-plate monotherapy (as the saved v2_sp reference rows)

Usage (repository root):
  .venv/Scripts/python.exe experiments/model_improvement/cross_screen_additive/run_additive.py \
      --config experiments/model_improvement/cross_screen_additive/configs/gpu.yaml \
      --output-dir runs/model_improvement/cross_screen_additive_v1/gpu
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
CSM = HERE.parent / "cross_screen_mono"
sys.path.insert(0, str(CSM))
from common import (CS_DIR, MONO_MAX, ROLES, ROOT, covered_cs_frames, frozen_frames, learned,  # noqa: E402
                    load_config, pilot, search, sha256_file, write_ids, append_log)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from bisurfgnn.models.compact_bisurf_additive import AdditiveCompactBiSurfGNN  # noqa: E402
from bisurfgnn.pretrain.nci60 import forbid_test_responses  # noqa: E402

RUN_ROOT = ROOT / "runs/model_improvement/cross_screen_additive_v1"
ANCHORED, ADDITIVE = "bisurf_mono_v2", "bisurf_additive_v2"
FROZEN_GPU_CONFIG = CSM / "configs/gpu.yaml"
# keys that may differ from the CrossScreenMono-v1 GPU configuration; every training/data setting must match
RUN_KEYS = {"protocol", "protocol_document", "jobs", "hsa_cs", "models", "seeds", "repeat_seed",
            "same_plate_controls", "max_wall_seconds", "minimum_free_gpu_bytes"}
COLUMNS = ["information", "run", "model", "seed", "best_epoch", "surface_macro_rmse", "point_rmse", "mae",
           "above_one_rmse", "prediction_min", "prediction_max", "wall_seconds", "status"]
ORIGINAL_BUILD = search.build_model
INIT_CHECKS: list[dict] = []


def code_records() -> dict[str, str]:
    """Files executed by this runner; summarize.py is not imported and is deliberately not bound."""
    paths = sorted(set((ROOT / "src/bisurfgnn").rglob("*.py")) | set((ROOT / "experiments/model_improvement").glob("*.py"))
                   | set((ROOT / "experiments/model_improvement/analysis").glob("*.py")) | {CSM / "common.py"}
                   | {HERE / "run_additive.py"} | set((HERE / "configs").glob("*.yaml")))
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def build_model(banks, variant, seed, cfg):
    """Frozen `run_algorithm_search.build_model`, plus the additive variant.

    The additive model must equal the frozen v2 model in module structure, scalar settings and initial
    parameters; only its output map differs. The RNG is re-seeded last, exactly as in the frozen builder."""
    if variant != ADDITIVE:
        return ORIGINAL_BUILD(banks, variant, seed, cfg)
    reference = ORIGINAL_BUILD(banks, ANCHORED, seed, cfg).cpu()
    learned._seed_everything(seed)
    model = AdditiveCompactBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim, 136, graph_layers=2,
                                     attention_heads=4, dropout=0.1, monotherapy_upper_bound=3.0,
                                     residual_monotherapy_input=True)
    expected, actual = reference.state_dict(), model.state_dict()
    same_state = expected.keys() == actual.keys() and all(torch.equal(expected[k], actual[k]) for k in expected)
    same_modules = repr(reference).split("(", 1)[1] == repr(model).split("(", 1)[1]
    same_settings = all(getattr(reference, k) == getattr(model, k) for k in
                        ("hidden_dim", "reference_mode", "monotherapy_upper_bound", "residual_monotherapy_input"))
    if not (same_state and same_modules and same_settings):
        raise RuntimeError(f"Additive model differs from frozen v2 beyond the output map (seed {seed})")
    INIT_CHECKS.append(dict(seed=seed, tensors=len(actual), initial_state_identical_to_v2=True,
                            module_structure_identical_to_v2=True, settings_identical_to_v2=True))
    del reference
    learned._seed_everything(seed)
    return model.to(torch.device(cfg["device"]))


def check_frozen_settings(cfg: dict) -> list[str]:
    frozen = load_config(FROZEN_GPU_CONFIG)
    keys = (set(cfg) | set(frozen)) - RUN_KEYS
    different = sorted(k for k in keys if cfg.get(k) != frozen.get(k))
    if different:
        raise ValueError(f"Training/data settings differ from CrossScreenMono-v1 GPU: {different}")
    return sorted(keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path, output = args.config.resolve(), args.output_dir.resolve()
    cfg = load_config(config_path)
    if cfg["fold"] != 0 or cfg["claim_scope"] != "engineering_pilot_no_performance_ranking":
        raise ValueError("Development fold-0 pilot configuration required")
    checked_keys = check_frozen_settings(cfg)
    jobs = [(j["information"], j["model"], int(j["seed"]), j["run"], j.get("saved_predictions")) for j in cfg["jobs"]]
    if not all(i in {"cs", "sp"} and m in {ANCHORED, ADDITIVE} and r in {"primary", "repeat", "control"}
               for i, m, _, r, _ in jobs):
        raise ValueError("Unregistered job")
    if not output.is_relative_to(RUN_ROOT) or output == RUN_ROOT:
        raise ValueError("Output must be a fresh subdirectory of runs/model_improvement/cross_screen_additive_v1")
    if not (CS_DIR / "manifest.json").exists() or (CS_DIR / "STOP.json").exists():
        raise RuntimeError("CrossScreenMono-v1 step 1 (CS table) is required")
    if cfg["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable; no silent CPU fallback")
        free, _ = torch.cuda.mem_get_info()
        if free < cfg["minimum_free_gpu_bytes"]:
            raise RuntimeError(f"Only {free / 2**30:.1f} GiB GPU memory free")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); deadline = start + cfg["max_wall_seconds"]
    code = code_records()
    snapshot = output / "source_snapshot"
    for rel in [config_path.relative_to(ROOT).as_posix(), "experiments/model_improvement/cross_screen_additive/run_additive.py",
                "experiments/model_improvement/cross_screen_mono/common.py", "src/bisurfgnn/models/compact_bisurf_additive.py"]:
        dest = snapshot / rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / rel, dest)
    learned._write_json_atomic(output / "launch.json", {
        "config": cfg, "config_sha256": sha256_file(config_path), "frozen_settings_checked_against":
        str(FROZEN_GPU_CONFIG.relative_to(ROOT)), "frozen_settings_keys": checked_keys, "code_sha256": code,
        "jobs": jobs, "cs_table_manifest_sha256": sha256_file(CS_DIR / "manifest.json"),
        "written_before_response_loading": True, "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0) if cfg["device"] == "cuda" else None,
        "free_gpu_bytes_at_launch": torch.cuda.mem_get_info()[0] if cfg["device"] == "cuda" else None})
    pd.DataFrame([dict(information=i, run=r, model=m, seed=s, status="pending") for i, m, s, r, _ in jobs],
                 columns=COLUMNS).to_csv(output / "results.csv", index=False)
    guard = forbid_test_responses()
    try:
        with guard, threadpool_limits(limits=cfg["threads"]), \
                patch.dict(search.VARIANTS, {ADDITIVE: None}), patch.object(search, "build_model", build_model):
            torch.set_num_threads(cfg["threads"])
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            frames, banks, tables, _, binding = frozen_frames(cfg, output / "sp")
            frames_cs, cs_audit = covered_cs_frames(frames)
            tables_cs = {r: learned.build_compact_point_table(frames_cs[r], banks, role=r, fold=0,
                                                              monotherapy_upper_bound=MONO_MAX) for r in ROLES}
            (output / "cs").mkdir()
            id_records = {}
            for role in ROLES:
                h = write_ids(frames_cs[role], output / "cs" / f"{role}_ids.csv")
                if h != sha256_file(CS_DIR / f"covered_{role}_ids.csv"):
                    raise RuntimeError(f"Covered ID file drift: {role}")
                id_records[("cs", role)] = dict(path=str(output / "cs" / f"{role}_ids.csv"), sha256=h)
                id_records[("sp", role)] = dict(path=str(output / "sp" / f"{role}_ids.csv"),
                                                sha256=binding["id_sha256"][role])
            learned._write_json_atomic(output / "data_binding.json", {
                "frozen_id_sha256": binding["id_sha256"], "reference_inputs": binding["reference_inputs"],
                "context_state_hash": banks.context_state_hash, "cs_subset": cs_audit,
                "covered_ids": {f"{k[0]}:{k[1]}": v for k, v in id_records.items()}, "test_response_rows_read": 0,
                "same_plate_monotherapy": "absent from every cs job input (columns overwritten by CS values)"})
            result, primary, checks = [], {}, []
            for info, model, seed, run, saved in jobs:
                job_start = time.monotonic()
                out = output / info / f"{model}_seed_{seed}_{run}"; out.mkdir()
                learned._write_json_atomic(output / "progress.json", {"status": "running", "information": info,
                    "model": model, "seed": seed, "run": run, "completed_jobs": len(result), "total_jobs": len(jobs),
                    "elapsed_seconds": time.monotonic() - start})
                print(f"START {info} {model} seed={seed} {run}", flush=True)
                T = tables_cs if info == "cs" else tables
                refs = [id_records[(info, r)] for r in ROLES]
                with patch.object(search, "check_reference", lambda *_: refs):
                    pred, detail = search.train_variant(T, banks, cfg, seed, None, out, deadline)
                va = T["validation"]
                pred = np.asarray(pred, dtype=np.float64)
                row = dict(information=info, run=run, model=model, seed=seed, best_epoch=detail["best_epoch"],
                           **pilot.metric(va.target.numpy(), pred, va.surface_ids),
                           prediction_min=float(pred.min()), prediction_max=float(pred.max()),
                           wall_seconds=time.monotonic() - job_start, status="passed")
                frame = pd.DataFrame(dict(source_row=va.source_rows, surface_id=va.surface_ids,
                                          target=va.target.numpy(), prediction=pred))
                frame.to_csv(out / "predictions.csv", index=False)
                frame.assign(abs_error=lambda f: abs(f.target - f.prediction)).nlargest(20, "abs_error").to_csv(
                    out / "largest_errors.csv", index=False)
                learned._write_json_atomic(out / "metrics.json", {**row, **{k: v for k, v in detail.items() if k != "history"},
                                                                  "history_length": len(detail.get("history", []))})
                if run == "primary":
                    primary[(info, model, seed)] = pred.copy()
                elif run == "repeat":
                    diff = float(np.max(np.abs(primary[(info, model, seed)] - pred)))
                    checks.append(dict(kind="repeat", information=info, model=model, seed=seed, max_abs_difference=diff))
                    if diff > 1e-7:
                        raise RuntimeError(f"Repeated prediction drift: {model} {seed} {diff}")
                else:
                    saved_frame = pd.read_csv(ROOT / saved)
                    if not (saved_frame.source_row.to_numpy() == va.source_rows).all():
                        raise RuntimeError("Saved control predictions are not in the validation order")
                    diff = float(np.max(np.abs(saved_frame.prediction.to_numpy(np.float64) - pred)))
                    checks.append(dict(kind="reproduction_control", information=info, model=model, seed=seed,
                                       max_abs_difference=diff, saved_predictions=saved,
                                       saved_surface_macro_rmse=pilot.metric(va.target.numpy(), saved_frame.prediction.to_numpy(),
                                                                             va.surface_ids)["surface_macro_rmse"],
                                       rerun_surface_macro_rmse=row["surface_macro_rmse"]))
                    print(f"CONTROL {info} {model} seed={seed} max|diff| vs saved = {diff:.3g}", flush=True)
                    if diff > 1e-7:
                        raise RuntimeError(f"Reproduction control drift: {model} {seed} {diff}")
                result.append(row)
                pd.DataFrame(result, columns=COLUMNS).to_csv(output / "results.csv", index=False)
                print(f"DONE {info} {model} seed={seed} {run} rmse={row['surface_macro_rmse']:.7f}", flush=True)
                if cfg["device"] == "cuda":
                    torch.cuda.empty_cache()
        drift = [r for r, h in code.items() if sha256_file(ROOT / r) != h]
        if drift:
            raise RuntimeError(f"Source drift during the run: {drift}")
        summary = {"status": "passed", "claim_scope": cfg["claim_scope"], "results": result, "checks": checks,
                   "initialisation_checks": INIT_CHECKS, "test_response_rows_read": 0,
                   "wall_seconds": time.monotonic() - start,
                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if cfg["device"] == "cuda" else None}
        learned._write_json_atomic(output / "summary.json", summary)
        learned._write_json_atomic(output / "progress.json", {"status": "passed", "completed_jobs": len(result),
                                                              "total_jobs": len(jobs), "wall_seconds": summary["wall_seconds"]})
        append_log(RUN_ROOT / "STEP_LOG.md", f"RUN {output.name} passed in {summary['wall_seconds']:.0f}s: " + "; ".join(
            f"{r['information']}:{r['model']}:{r['seed']}:{r['run']}={r['surface_macro_rmse']:.6f}" for r in result)
            + "; checks " + json.dumps(checks))
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}, default=str), flush=True)
    except Exception as error:
        learned._write_json_atomic(output / "failure.json", {"status": "failed", "error": repr(error)})
        learned._write_json_atomic(output / "progress.json", {"status": "failed", "error": repr(error)})
        append_log(RUN_ROOT / "STEP_LOG.md", f"RUN {output.name} FAILED: {error!r}")
        raise
    finally:
        learned._write_json_atomic(output / "input_access.json", {"parquet_reads": guard.accesses,
                                                                  "test_response_rows_read": 0})


if __name__ == "__main__":
    main()

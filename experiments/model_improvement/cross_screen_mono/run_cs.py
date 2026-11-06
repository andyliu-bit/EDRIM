"""CrossScreenMono-v1 step 3 runner (docs/CROSS_SCREEN_MONO_V1_PROTOCOL.md sections 3-4).

Methods (no new tuning; frozen AlgorithmSearch-v1 code paths are called unchanged):
  hsa_cs                 min(CS mono a, CS mono b), zero parameters
  xgb_residual_d10_m1    run_algorithm_search.train_xgb (frozen T configuration) on CS-covered frames whose
                         monotherapy columns are REPLACED by CS values (features and HSA residual base)
  bisurf_mono_v2         run_algorithm_search.train_variant (CompactBiSurfGNN v2: hidden 136, 2 layers, 4 heads,
                         dropout 0.1, upper bound 3, residual monotherapy input) on CS tables: CS values are the
                         decoder monotherapy input and the HSA anchor
Same-plate controls (full frozen sample, original information) re-run one saved reference job through this
runner and compare with its saved predictions; they are reproduction checks, not comparison rows.

Usage (repository root):
  .venv/Scripts/python.exe experiments/model_improvement/cross_screen_mono/run_cs.py \
      --config experiments/model_improvement/cross_screen_mono/configs/cpu.yaml \
      --output-dir runs/model_improvement/cross_screen_mono_v1/cpu
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CS_DIR, HERE, MONO_MAX, ROLES, ROOT, RUN_ROOT, append_log, covered_cs_frames,  # noqa: E402
                    frozen_frames, learned, load_config, pilot, search, sha256_file, write_ids)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from bisurfgnn.pretrain.nci60 import forbid_test_responses  # noqa: E402

COLUMNS = ["information", "run", "model", "seed", "best_epoch", "surface_macro_rmse", "point_rmse", "mae",
           "above_one_rmse", "wall_seconds", "status"]


def code_records() -> dict[str, str]:
    paths = sorted(set((ROOT / "src/bisurfgnn").rglob("*.py")) | set((ROOT / "experiments/model_improvement").glob("*.py"))
                   | set((ROOT / "experiments/model_improvement/analysis").glob("*.py")) | set(HERE.rglob("*.py"))
                   | set(HERE.rglob("*.yaml")))
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def jobs_from(cfg: dict) -> list[tuple[str, str, int | None, str]]:
    jobs = [("cs", "hsa_cs", None, "primary")] if cfg.get("hsa_cs") else []
    jobs += [("sp_control", c["model"], int(c["seed"]), "control") for c in cfg.get("same_plate_controls", [])]
    jobs += [("cs", m, int(s), "primary") for m in cfg["models"] for s in cfg["seeds"]]
    jobs += [("cs", m, int(cfg["repeat_seed"]), "repeat") for m in cfg["models"]]
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path, output = args.config.resolve(), args.output_dir.resolve()
    cfg = load_config(config_path)
    if cfg["fold"] != 0 or cfg["claim_scope"] != "engineering_pilot_no_performance_ranking":
        raise ValueError("Development fold-0 pilot configuration required")
    if not all(m in {"xgb_residual_d10_m1", "bisurf_mono_v2"} for m in cfg["models"]):
        raise ValueError("Only the frozen XGB T configuration and BiSurfGNN v2 are registered")
    if not output.is_relative_to(RUN_ROOT) or output == RUN_ROOT:
        raise ValueError("Output must be a fresh subdirectory of runs/model_improvement/cross_screen_mono_v1")
    if not (CS_DIR / "manifest.json").exists() or (CS_DIR / "STOP.json").exists():
        raise RuntimeError("Step 1 must have completed without triggering the coverage stop rule")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable; no silent CPU fallback")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); deadline = start + cfg["max_wall_seconds"]
    jobs = jobs_from(cfg)
    code = code_records()
    snapshot = output / "source_snapshot"
    for rel in [config_path.relative_to(ROOT).as_posix()] + [r for r in code if r.startswith("experiments/model_improvement/cross_screen_mono")
                                                     or r.startswith("src/bisurfgnn/cross_screen")]:
        dest = snapshot / rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / rel, dest)
    learned._write_json_atomic(output / "launch.json", {
        "config": cfg, "config_sha256": sha256_file(config_path), "code_sha256": code, "jobs": jobs,
        "cs_table_manifest_sha256": sha256_file(CS_DIR / "manifest.json"), "written_before_response_loading": True,
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "xgboost": __import__("xgboost").__version__,
        "gpu": torch.cuda.get_device_name(0) if cfg["device"] == "cuda" else None})
    pd.DataFrame([dict(information=i, run=r, model=m, seed=s, status="pending") for i, m, s, r in jobs],
                 columns=COLUMNS).to_csv(output / "results.csv", index=False)
    guard = forbid_test_responses()
    try:
        with guard, threadpool_limits(limits=cfg["threads"]):
            torch.set_num_threads(cfg["threads"])
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            frames, banks, tables, fingerprints, binding = frozen_frames(cfg, output / "sp_control")
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
                id_records[("sp_control", role)] = dict(path=str(output / "sp_control" / f"{role}_ids.csv"),
                                                       sha256=binding["id_sha256"][role])
            tr_cs = frames_cs["train"]
            full_tr = frames["train"]
            feature_sets = dict(
                train_drugs_full=len(set(full_tr.drug_a_id) | set(full_tr.drug_b_id)),
                train_drugs_covered=len(set(tr_cs.drug_a_id) | set(tr_cs.drug_b_id)),
                train_cells_full=int(full_tr.cell_id.nunique()), train_cells_covered=int(tr_cs.cell_id.nunique()),
                pca_fit_sets_identical=(set(full_tr.drug_a_id) | set(full_tr.drug_b_id)) == (set(tr_cs.drug_a_id) | set(tr_cs.drug_b_id))
                and set(full_tr.cell_id) == set(tr_cs.cell_id))
            learned._write_json_atomic(output / "data_binding.json", {
                "frozen_id_sha256": binding["id_sha256"], "reference_inputs": binding["reference_inputs"],
                "context_state_hash": banks.context_state_hash, "cs_subset": cs_audit, "xgb_pca_fit_sets": feature_sets,
                "covered_ids": {f"{k[0]}:{k[1]}": v for k, v in id_records.items()}, "test_response_rows_read": 0,
                "same_plate_monotherapy": "absent from every cs job input (columns overwritten by CS values)"})
            controls = {(c["model"], int(c["seed"])): ROOT / c["saved_predictions"] for c in cfg.get("same_plate_controls", [])}
            result, primary, checks = [], {}, []
            for info, model, seed, run in jobs:
                job_start = time.monotonic()
                out = output / info / f"{model}_seed_{seed}_{run}"; out.mkdir()
                learned._write_json_atomic(output / "progress.json", {"status": "running", "information": info, "model": model,
                    "seed": seed, "run": run, "completed_jobs": len(result), "total_jobs": len(jobs),
                    "elapsed_seconds": time.monotonic() - start})
                print(f"START {info} {model} seed={seed} {run}", flush=True)
                F, T = (frames_cs, tables_cs) if info == "cs" else (frames, tables)
                refs = [id_records[(info, r)] for r in ROLES]
                with patch.object(search, "check_reference", lambda *_: refs):
                    if model == "hsa_cs":
                        pred = np.minimum(T["validation"].mono_a.numpy(), T["validation"].mono_b.numpy()).astype(np.float64)
                        detail = {"best_epoch": None}
                    elif model.startswith("xgb"):
                        pred, detail = search.train_xgb(F, banks, fingerprints, cfg, out, deadline)
                    else:
                        pred, detail = search.train_variant(T, banks, cfg, seed, None, out, deadline)
                va = T["validation"]
                row = dict(information=info, run=run, model=model, seed=seed, best_epoch=detail["best_epoch"],
                           **pilot.metric(va.target.numpy(), pred, va.surface_ids),
                           wall_seconds=time.monotonic() - job_start, status="passed")
                frame = pd.DataFrame(dict(source_row=va.source_rows, surface_id=va.surface_ids,
                                          target=va.target.numpy(), prediction=pred))
                frame.to_csv(out / "predictions.csv", index=False)
                frame.assign(abs_error=lambda f: abs(f.target - f.prediction)).nlargest(20, "abs_error").to_csv(
                    out / "largest_errors.csv", index=False)
                learned._write_json_atomic(out / "metrics.json", {**row, **{k: v for k, v in detail.items() if k != "history"},
                                                                  "history_length": len(detail.get("history", []))})
                if run == "primary":
                    primary[(model, seed)] = np.asarray(pred).copy()
                elif run == "repeat":
                    diff = float(np.max(np.abs(primary[(model, seed)] - pred)))
                    checks.append(dict(kind="repeat", model=model, seed=seed, max_abs_difference=diff))
                    if diff > 1e-7:
                        raise RuntimeError(f"Repeated prediction drift: {model} {seed} {diff}")
                else:
                    saved = pd.read_csv(controls[(model, seed)])
                    if not (saved.source_row.to_numpy() == va.source_rows).all():
                        raise RuntimeError("Saved control predictions are not in frozen validation order")
                    diff = float(np.max(np.abs(saved.prediction.to_numpy(np.float64) - np.asarray(pred, np.float64))))
                    saved_rmse = pilot.metric(va.target.numpy(), saved.prediction.to_numpy(), va.surface_ids)["surface_macro_rmse"]
                    checks.append(dict(kind="same_plate_reproduction", model=model, seed=seed, max_abs_difference=diff,
                                       saved_surface_macro_rmse=saved_rmse, rerun_surface_macro_rmse=row["surface_macro_rmse"],
                                       saved_predictions=str(controls[(model, seed)].relative_to(ROOT))))
                    print(f"CONTROL {model} seed={seed} max|diff| vs saved = {diff:.3g}", flush=True)
                result.append(row)
                pd.DataFrame(result, columns=COLUMNS).to_csv(output / "results.csv", index=False)
                print(f"DONE {info} {model} seed={seed} {run} rmse={row['surface_macro_rmse']:.7f}", flush=True)
                if cfg["device"] == "cuda":
                    torch.cuda.empty_cache()
        drift = [r for r, h in code.items() if sha256_file(ROOT / r) != h]
        if drift:
            raise RuntimeError(f"Source drift during the run: {drift}")
        summary = {"status": "passed", "claim_scope": cfg["claim_scope"], "results": result, "checks": checks,
                   "test_response_rows_read": 0, "wall_seconds": time.monotonic() - start,
                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if cfg["device"] == "cuda" else None}
        learned._write_json_atomic(output / "summary.json", summary)
        learned._write_json_atomic(output / "progress.json", {"status": "passed", "completed_jobs": len(result),
                                                              "total_jobs": len(jobs), "wall_seconds": summary["wall_seconds"]})
        append_log(RUN_ROOT / "STEP_LOG.md", f"STEP3 {output.name} passed in {summary['wall_seconds']:.0f}s: " + "; ".join(
            f"{r['information']}:{r['model']}:{r['seed']}:{r['run']}={r['surface_macro_rmse']:.6f}" for r in result)
            + "; checks " + json.dumps(checks))
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}, default=str), flush=True)
    except Exception as error:
        learned._write_json_atomic(output / "failure.json", {"status": "failed", "error": repr(error)})
        learned._write_json_atomic(output / "progress.json", {"status": "failed", "error": repr(error)})
        append_log(RUN_ROOT / "STEP_LOG.md", f"STEP3 {output.name} FAILED: {error!r}")
        raise
    finally:
        learned._write_json_atomic(output / "input_access.json", {"parquet_reads": guard.accesses,
                                                                  "test_response_rows_read": 0})


if __name__ == "__main__":
    main()

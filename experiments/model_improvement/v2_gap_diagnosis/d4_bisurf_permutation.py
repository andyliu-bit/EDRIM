"""V2 gap diagnosis, step D4: does the gate's BiSurfGNN use cell context / drug structure beyond the anchors?

Inference only, VALIDATION split only, saved gate checkpoints only (no training, nothing written into the gate dir).
For every saved BiSurfGNN run (additive seeds 41-45, anchored 41):
  1. rebuild the model exactly as the gate did (gate_common.build_model), load model.pt and re-predict validation;
     the result must reproduce the saved validation_predictions.csv;
  2. cell permutation: every validation point reads the context graph of another cell (fixed derangement of the
     59 cells); the monotherapy anchors, doses and drugs are untouched;
  3. drug permutation: every drug reads the molecular graph of another drug (fixed derangement of the 101 drugs);
     anchors, doses and cells untouched.
The RMSE increase measures how much the learned correction depends on that channel.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/v2_gap_diagnosis/d4_bisurf_permutation.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public_split_gate"))
sys.path.insert(0, str(HERE))
import gate_common as gate  # noqa: E402  (frozen gate helpers, imported read only)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from gate_common import learned, search  # noqa: E402

OUT_DEFAULT = gate.ROOT / "runs/model_improvement/v2_gap_diagnosis/permutation"
RUNS = {
    "BS_add_s41": ("additive", 41, gate.RUN / "s3_additive_r2/bisurf_additive_seed_41"),
    "BS_add_s42": ("additive", 42, gate.RUN / "s4_bisurf_additive_a/bisurf_additive_seed_42"),
    "BS_add_s43": ("additive", 43, gate.RUN / "s4_bisurf_additive_b/bisurf_additive_seed_43"),
    "BS_add_s44": ("additive", 44, gate.RUN / "s4_bisurf_additive_a/bisurf_additive_seed_44"),
    "BS_add_s45": ("additive", 45, gate.RUN / "s4_bisurf_additive_b/bisurf_additive_seed_45"),
    "BS_anch_s41": ("anchored", 41, gate.RUN / "s3_anchored_r2/bisurf_anchored_seed_41"),
}
PERM_SEED = 20261013


def derangement(n: int, rng) -> np.ndarray:
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def main():
    if not sys.flags.dont_write_bytecode:
        raise SystemExit("run with -B (no bytecode next to the frozen gate code)")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = gate.load_config(gate.HERE / "configs/gate.yaml") if hasattr(gate, "HERE") else \
        gate.load_config(gate.ROOT / "experiments/model_improvement/public_split_gate/configs/gate.yaml")
    t0 = time.time()
    frames = {r: gate.load_points(r) for r in ("train", "validation")}      # split 1/2 tables only
    banks = gate.build_banks(frames)
    va = learned.build_compact_point_table(frames["validation"], banks, role="validation", fold=0, monotherapy_upper_bound=3.0)
    target = va.target.numpy().astype(float) * 100.0
    del frames
    rng = np.random.default_rng(PERM_SEED)
    cell_perm = torch.as_tensor(derangement(len(banks.cell_ids), rng), dtype=va.cell_index.dtype)
    drug_perm = torch.as_tensor(derangement(len(banks.drug_ids), rng), dtype=va.drug_a_index.dtype)
    va_cell = dataclasses.replace(va, cell_index=cell_perm[va.cell_index.long()])
    va_drug = dataclasses.replace(va, drug_a_index=drug_perm[va.drug_a_index.long()],
                                  drug_b_index=drug_perm[va.drug_b_index.long()])
    report = {"step": "D4", "test_rows_read": 0, "validation_points": int(len(target)), "perm_seed": PERM_SEED,
              "setup_seconds": round(time.time() - t0, 1), "runs": {}}
    hsa = np.minimum(va.mono_a.numpy(), va.mono_b.numpy()).astype(float) * 100.0
    for tag, (form, seed, path) in RUNS.items():
        t1 = time.time()
        model = gate.build_model(banks, form, seed, cfg["device"])
        model.load_state_dict(torch.load(path / "model.pt", weights_only=True, map_location="cpu"))
        model.eval()
        saved = pd.read_csv(path / "validation_predictions.csv")
        if not (saved.source_row.to_numpy() == va.source_rows).all():
            raise RuntimeError(f"{tag}: validation order differs from the saved predictions")
        base = search.predict(model, va, banks, cfg).astype(float) * 100.0
        diff = float(np.max(np.abs(base - saved.prediction.to_numpy(float) * 100.0)))
        if diff > 1e-3:
            raise RuntimeError(f"{tag}: re-prediction differs from the saved validation predictions ({diff})")
        cell = search.predict(model, va_cell, banks, cfg).astype(float) * 100.0
        drug = search.predict(model, va_drug, banks, cfg).astype(float) * 100.0
        r = lambda p: float(np.sqrt(np.mean((p - target) ** 2)))  # noqa: E731
        corr = base - hsa
        report["runs"][tag] = dict(
            reproduce_max_abs_diff_percent=diff, rmse=r(base), rmse_cell_permuted=r(cell), rmse_drug_permuted=r(drug),
            rmse_hsa=r(hsa), rel_cell=r(cell) / r(base) - 1.0, rel_drug=r(drug) / r(base) - 1.0,
            correction_sd=float(np.std(corr)),
            correction_change_rms_cell=float(np.sqrt(np.mean((cell - base) ** 2))),
            correction_change_rms_drug=float(np.sqrt(np.mean((drug - base) ** 2))),
            seconds=round(time.time() - t1, 1))
        np.savez_compressed(out / f"{tag}_permuted_predictions.npz", base=base.astype(np.float32),
                            cell_permuted=cell.astype(np.float32), drug_permuted=drug.astype(np.float32))
        print(tag, json.dumps(report["runs"][tag]), flush=True)
        del model
        torch.cuda.empty_cache()
    report["cell_permutation"] = [banks.cell_ids[i] + "->" + banks.cell_ids[int(j)] for i, j in enumerate(cell_perm)][:5]
    with (out / "d4_report.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()

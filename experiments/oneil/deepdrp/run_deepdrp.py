"""DeepDRP (adapted) on O'Neil, launch card docs/DEEPDRP_ADAPTATION_CARD_20261103.md: the NCI adapter
(experiments/model_improvement/deepdrp_adapted/deepdrp.py) on 5x5 grids, index 0 = drug absent, i.e. the 4x4
combination cells plus their same-batch monotherapy. Global learning rate as chosen on NCI split 2, early stopping on
O'Neil validation; cell features = the 78 highest-variance genes of DepMap 24Q4 over the 30 cells, z-scored. Writes
validation and test predictions (ratio scale) through common.write_predictions; the test inputs carry no responses.

  .venv/Scripts/python.exe -I -B experiments/oneil/deepdrp/run_deepdrp.py --seed 41 --global-lr 1e-4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "model_improvement/deepdrp_adapted")]
import common  # noqa: E402

common.install_guard()
import deepdrp as dd  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DEPMAP_NPZ = common.OLD / "runs/audit_revision/oneil_depmap_coverage_v1/expression.npz"
# smoke tests set ONEIL_PRED_ROOT (common.py): the run folder then goes there too
RUNS = Path(os.environ["ONEIL_PRED_ROOT"]) / "runs" if os.environ.get("ONEIL_PRED_ROOT") else common.OUT / "deepdrp_adapted"
G, CONTROL_PERCENT = 5, 100.0                 # O'Neil responses are viability ratios to the untreated control
DEVIATIONS = dict(dd.DEVIATIONS, O1="O'Neil: 5x5 grids (untreated row/column + the 4x4 combination cells), "
                  "same-batch monotherapy, DepMap 24Q4 genes (78 by variance over the 30 cells), control 100%, "
                  "global learning rate as chosen on NCI split 2")


def cell_features(cells):
    z = np.load(DEPMAP_NPZ, allow_pickle=False)
    pos = {m: k for k, m in enumerate(z["model_ids"])}
    x = z["expression"][[pos[c] for c in cells]].astype(np.float64)
    top = np.argsort(-x.var(0), kind="stable")[:dd.N_GENES]
    x = x[:, top]
    return torch.tensor((x - x.mean(0)) / x.std(0), dtype=torch.float32), [str(g) for g in z["genes"][top]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--global-lr", type=float, required=True)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    out = RUNS / f"lr{args.global_lr:g}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    log_file = (out / "log.txt").open("w", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    t0 = time.perf_counter()
    dd.seed_everything(args.seed)
    train, val, test = (common.load_points(r) for r in common.ROLES)
    drugs = sorted(set(train.drug_a_id) | set(train.drug_b_id))
    cells = sorted(set(train.biological_cell_id))
    for name, f in (("validation", val), ("test", test)):
        if not (set(f.drug_a_id) | set(f.drug_b_id)) <= set(drugs) or not set(f.biological_cell_id) <= set(cells):
            raise RuntimeError(f"{name} drugs/cells missing from the training split")
    tokens = dd.drug_tokens(drugs, common.DRUG / "drugs.csv").to(device)
    cellx, genes = cell_features(cells)
    cellx = cellx.to(device)
    lohi = dd.dose_range(train)
    pos_d, pos_c = {d: i for i, d in enumerate(drugs)}, {c: i for i, c in enumerate(cells)}
    Mtr, Mva, Mte = (dd.Matrices(f, pos_d, pos_c, lohi, CONTROL_PERCENT, device, G=G) for f in (train, val, test))
    log(f"train matrices {Mtr.n}, validation {Mva.n}, test {Mte.n}, drugs {len(drugs)}, cells {len(cells)}")
    g = dd.GlobalNet().to(device)
    h_global = dd.train_global(g, tokens, cellx, Mtr, Mva, args.global_lr, args.seed, args.max_epochs, args.patience, log)
    dd.seed_everything(args.seed + 1)
    local = dd.LocalNet(dd.grid_adjacency(G).to(device)).to(device)
    h_local = dd.train_local(g, local, tokens, cellx, Mtr, Mva, args.seed, args.max_epochs, args.patience, log)
    torch.save({"global": g.state_dict(), "local": local.state_dict(), "drugs": drugs, "cells": cells, "genes": genes,
                "dose_range": lohi, "control_percent": CONTROL_PERCENT, "grid": G}, out / "model_selected.pt")
    meta = dict(launch_card="docs/DEEPDRP_ADAPTATION_CARD_20261103.md", seed=args.seed, global_lr=args.global_lr)
    pred = {}
    for role, f, M in (("validation", val, Mva), ("test", test, Mte)):
        pred[role] = M.frame_order(dd.predict_final(g, local, tokens, cellx, M))          # ratio scale
        common.write_predictions("deepdrp", args.seed, role, f.assign(prediction=pred[role]), meta=meta)
    gv = Mva.frame_order(dd.predict_global(g, tokens, cellx, Mva))
    result = dict(method="DeepDRP (adapted), O'Neil", seed=args.seed, global_lr=args.global_lr, test_responses_read=0,
                  validation=dd.metrics(100.0 * pred["validation"], val),
                  validation_surface_macro_percent=common.surface_macro_percent(val, pred["validation"]),
                  validation_global_only=dd.metrics(100.0 * gv, val),
                  global_stage={k: v for k, v in h_global.items() if k != "history"},
                  local_stage={k: v for k, v in h_local.items() if k != "history"},
                  genes=genes, dose_range_log10=lohi, deviations=DEVIATIONS,
                  wall_seconds=round(time.perf_counter() - t0, 1))
    (out / "history.json").write_text(json.dumps({"global": h_global["history"], "local": h_local["history"]},
                                                 indent=1), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    log(f"validation combination RMSE% {result['validation']['combination_rmse_percent']:.4f}, surface macro "
        f"{result['validation_surface_macro_percent']:.4f} (global only "
        f"{result['validation_global_only']['combination_rmse_percent']:.4f}, lookup Bliss "
        f"{result['validation']['lookup_bliss_rmse_percent']:.4f}) wall {result['wall_seconds']}s")
    log_file.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:                                       # a CUDA error otherwise hangs the process at teardown
        import os
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

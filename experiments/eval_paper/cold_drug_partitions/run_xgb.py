"""Eval-paper supplement G: gate XGBoost residual (cap 20000 trees) on one new-drug partition.

Runs the D11 runner experiments/model_improvement/cold_drug/run_cold_xgb.py main() unchanged, with three module
attributes swapped before the call: the output root (runs/eval_paper/cold_drug_partitions/p<partition>), the frames
(this partition's split.json through cold_split.assign on gate_common.load_points), and the tree cap (20000, the D11
"D11 supplement" value). Tree count is chosen on the partition's new-drug validation role only; the test role is
predicted and saved, never scored here (score.py scores).
Output: p<partition>/xgb_seed_<s>/ (smoke: p<partition>/smoke_xgb_seed_<s>/, tiny subsets, 100 trees).

  .venv/Scripts/python.exe -I -B experiments/eval_paper/cold_drug_partitions/run_xgb.py --partition 20261018 --seed 41 [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/model_improvement/cold_drug"))
sys.path.insert(0, str(ROOT / "experiments/model_improvement/input_parity"))
import common as c  # noqa: E402
import run_parity as rp  # noqa: E402
import cold_split  # noqa: E402
import run_cold_xgb  # noqa: E402

import pandas as pd  # noqa: E402

MAX_TREES = 20000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", type=int, choices=c.NEW_SEEDS, required=True)
    ap.add_argument("--seed", type=int, choices=c.XGB_SEEDS, required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    split = json.loads(c.split_path(a.partition).read_text(encoding="utf-8"))
    held = set(split["test_drugs"]) | set(split["val_drugs"])

    def load_frames():
        both = pd.concat([rp.gate.load_points("train"), rp.gate.load_points("validation")], ignore_index=True)
        role = cold_split.assign(both, set(split["test_drugs"]), set(split["val_drugs"]))
        frames = {r: both[role == r].reset_index(drop=True) for r in cold_split.ROLES}
        del both
        tr = frames["train"]
        assert not (set(tr.drug_a_id) | set(tr.drug_b_id)) & held, "held-out drug in training"
        assert {r: len(f) for r, f in frames.items()} == {r: v["points"] for r, v in split["counts"].items()}
        return frames

    base_config = rp.effective_config

    def config():
        cfg = base_config()
        cfg["xgboost"]["max_trees"] = MAX_TREES
        return cfg

    # run_cold_xgb.main reads these module attributes at call time; swapping them reuses its code path verbatim
    cold_split.OUT = c.OUT / f"p{a.partition}"
    cold_split.load_frames = load_frames
    rp.effective_config = config
    sys.argv = [sys.argv[0], "--seed", str(a.seed)] + (["--smoke"] if a.smoke else [])
    run_cold_xgb.main()


if __name__ == "__main__":
    main()

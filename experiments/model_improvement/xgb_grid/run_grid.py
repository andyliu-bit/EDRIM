"""D20 (docs/XGB_GRID_CARD_20261026.md): validation-only grid for XGBoost + identity (the D8 model).

One run = one config x one seed. Features, swap augmentation, weights, tree cap (20000, GPU) and the tree-count selection
(every 50 trees by validation surface-macro RMSE) are D8's: run_xgb_identity_final.train_select is called unchanged; only
max_depth, eta and min_child_weight are replaced. Split 1 trains, split 2 selects and is reported. Split 3 is never read:
only run_gate.prepare (gate.load_points refuses any role but train/validation) is used, and none of the base runner's
test functions is called. Config 'base' = D8; every non-smoke run with D8's values is compared with D8's
selection_summary.json for its seed (metrics.json: d8_check).

  .venv/Scripts/python.exe -I -B experiments/model_improvement/xgb_grid/run_grid.py --config d8_e0.1_m10 --seed 41
  .venv/Scripts/python.exe -I -B experiments/model_improvement/xgb_grid/run_grid.py --config base --seed 41 --out-root <dir>
  .venv/Scripts/python.exe -I -B experiments/model_improvement/xgb_grid/run_grid.py --config d12_e0.1_m10 --seed 41 --smoke --out-root <dir>
(--smoke: tiny split-1/2 subsets, 100 trees, CPU, as the base runner's smoke.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "identity_control"))
import run_xgb_identity_final as fin  # noqa: E402

from threadpoolctl import threadpool_limits  # noqa: E402

gate, run_gate, rxi, learned, pilot = fin.gate, fin.run_gate, fin.rxi, fin.learned, fin.pilot
OUT = fin.V2 / "xgb_grid"
D8_SUMMARY = fin.OUT / "selection_summary.json"
BASE = dict(max_depth=10, eta=0.03, min_child_weight=1)
GRID = {f"d{d}_e{e}_m{m}": dict(max_depth=d, eta=e, min_child_weight=m)
        for d in (6, 8, 10, 12) for e in (0.03, 0.1) for m in (1, 10)}
CONFIGS = {"base": BASE, **GRID}
# Peak commit per GPU run in GiB: d10 (= D8) 16.4 and d12 32.6 measured by the D20 verifier, d6 5.6 measured, d8 guessed.
PEAK_GIB = {6: 6, 8: 10, 10: 16.4, 12: 32.6}
MARGIN_GIB = 3


def free_commit_gib():
    """Commit headroom (GlobalMemoryStatusEx.ullAvailPageFile) in GiB; pagefile growth is not counted."""
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("phys", "avail_phys", "commit_limit", "avail_commit", "virt",
                                              "avail_virt", "avail_ext")]
    s = Status(length=ctypes.sizeof(Status))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        raise ctypes.WinError()
    return s.avail_commit / 2**30


def wait_for_commit(depth):
    """Start only with room for this depth's peak: running out of commit would crash this run or the user's programs."""
    need, waited = PEAK_GIB[depth] + MARGIN_GIB, 0
    while (have := free_commit_gib()) < need:
        if waited == 0:
            print(f"WAIT free commit {have:.1f} GiB < {need:.1f} GiB needed for max_depth={depth}; polling every 60 s",
                  flush=True)
        time.sleep(60)
        waited += 1
    if waited:
        print(f"RESUME after {waited} min, free commit {have:.1f} GiB", flush=True)


def train_with(values, *args):
    """fin.train_select(*args) with max_depth/eta/min_child_weight set to `values`; selection unchanged."""
    import xgboost as xgb
    original = xgb.train

    def train(params, *a, **k):  # train_select hardcodes these three, so they are swapped in at its xgb.train call
        if {key: params[key] for key in BASE} != BASE:
            raise RuntimeError(f"Base runner parameters changed ({params}); the D20 grid no longer applies")
        params.update(values)  # in place, so the params train_select returns record the values actually used
        return original(params, *a, **k)

    xgb.train = train
    try:
        return fin.train_select(*args)
    finally:
        xgb.train = original


def d8_check(m):
    ref = next(s for s in json.loads(D8_SUMMARY.read_text(encoding="utf-8"))["seeds"] if s["seed"] == m["seed"])
    keys = ("best_trees", "validation_surface_macro_rmse", "validation_pooled_rmse_percent")
    diff = {k: m[k] - ref[k] for k in keys}
    return {"reference": {k: ref[k] for k in keys}, "difference": diff,
            "match": diff["best_trees"] == 0 and abs(diff["validation_pooled_rmse_percent"]) < 1e-6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--seed", type=int, choices=fin.SEEDS, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", type=Path, default=OUT)
    a = ap.parse_args()
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B`")
    if a.smoke and a.out_root.resolve() == OUT.resolve():
        raise SystemExit("--smoke needs its own --out-root")
    sys.stdout.reconfigure(encoding="utf-8")
    out = a.out_root / a.config / f"seed_{a.seed}"
    if out.exists():
        raise SystemExit(f"Already run: {out}")
    if not a.smoke:
        wait_for_commit(CONFIGS[a.config]["max_depth"])
    start = time.monotonic()
    cfg = gate.load_config(run_gate.CONFIG)
    device, max_trees = ("cpu", 100) if a.smoke else ("cuda", fin.MAX_TREES)
    values = CONFIGS[a.config]
    with threadpool_limits(limits=cfg["xgboost"]["threads"]):
        frames, banks, _ = run_gate.prepare(cfg)
        if set(frames) != {"train", "validation"}:
            raise RuntimeError(f"Unexpected frames {sorted(frames)}")
        if a.smoke:
            frames = run_gate.tiny_frames(frames, cfg["timing"]["subset_seed"] + 2)
        build = rxi.identity_inputs(frames, banks, gate.morgan(banks), cfg)
        print(f"START D20 config={a.config} {values} seed={a.seed} device={device} max_trees={max_trees} "
              f"train_points={len(frames['train'])} validation_points={len(frames['validation'])}", flush=True)
        booster, _, best_k, history, _, m = train_with(values, frames, build, cfg, a.seed, device, max_trees)
        del booster
    result = {"task": "D20", "card": "docs/XGB_GRID_CARD_20261026.md", "config": a.config, "grid_values": values,
              "smoke": a.smoke, "split_3_read": False, **m, "total_wall_seconds": time.monotonic() - start}
    if values == BASE and not a.smoke:
        result["d8_check"] = d8_check(m)
    out.mkdir(parents=True)  # created only now, so a crashed run leaves nothing behind; raises rather than overwrite
    pilot.write_history(out, history, config=a.config, seed=a.seed)
    learned._write_json_atomic(out / "metrics.json", result)
    print(f"SELECTED config={a.config} seed={a.seed} best_trees={best_k} at_cap={m['at_cap']} "
          f"val_surface_macro={m['validation_surface_macro_rmse']:.6f} val_pooled={m['validation_pooled_rmse_percent']:.4f} "
          f"({result['total_wall_seconds']:.0f}s)", flush=True)
    if "d8_check" in result:
        print("D8_CHECK " + json.dumps(result["d8_check"]), flush=True)


if __name__ == "__main__":
    main()

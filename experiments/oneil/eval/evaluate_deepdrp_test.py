"""docs/DEEPDRP_ADAPTATION_CARD_20261103.md: one-shot O'Neil test scoring of DeepDRP (adapted PyTorch re-implementation,
5 seeds) — descriptive only, reported as "主结果后补跑", following evaluate_combofm_test.py (copied, names changed).

The joint scorer (evaluate_oneil.py) already used its marker; DeepDRP gets its own marker and reuses the joint scorer's
decode path, bootstrap draws (2000, seed 20261009, sorted pairs) and the NCI Bliss definition, so its intervals are
directly comparable with the main table. No verdicts.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/eval/evaluate_deepdrp_test.py --role validation
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/eval/evaluate_deepdrp_test.py --role test   # once
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import common  # noqa: E402
import evaluate_oneil as ev  # noqa: E402

MARKER = ev.EVAL / "ONEIL_DEEPDRP_TEST_EVALUATED.marker"
STEM = {"validation": "deepdrp_validation_dryrun", "test": "deepdrp_test_results"}
RUNS = ([("hsa", None)] + [(m, s) for m in ("deepdrp", "ours_e500", "deepsynba_5x5", "xgb_id")
                                               for s in common.SEEDS])
PAIRS = [("ours_e500", "deepdrp"), ("deepdrp", "hsa"), ("deepdrp", "deepsynba_5x5"), ("deepdrp", "xgb_id")]


def load(role, points):
    out = {}
    for m, s in RUNS:
        p = points[common.KEY].merge(common.read_predictions(m, s, role)[common.KEY + ["prediction"]], on=common.KEY,
                                     how="left", validate="one_to_one")["prediction"].to_numpy(float)
        if not np.isfinite(p).all():
            raise SystemExit(f"STOP: {m}/{s}/{role}: predictions missing or non-finite")
        out[(m, s)] = p
    return out


def score(points, preds):
    y = points["response"].to_numpy(float)
    codes, pairs = pd.factorize(points["pair_id"], sort=True)
    n = np.bincount(codes, minlength=len(pairs)).astype(float)
    W = ev.bootstrap_weights(len(pairs))
    rmse, macro, boot = {}, {}, {}
    for (m, s), p in preds.items():
        rmse.setdefault(m, []).append(common.pooled_rmse_percent(y, p))
        macro.setdefault(m, []).append(common.surface_macro_percent(points, p))
        sse = np.bincount(codes, weights=((p - y) * 100.0) ** 2, minlength=len(pairs))
        boot.setdefault(m, []).append(np.sqrt((W @ sse) / (W @ n)))
    mean = {m: float(np.mean(v)) for m, v in rmse.items()}
    boot = {m: np.mean(v, axis=0) for m, v in boot.items()}
    contrasts = {f"{a} vs {b}": dict(r=mean[a] / mean[b] - 1.0, ci95=ev.interval(boot[a] / boot[b] - 1.0))
                 for a, b in PAIRS}
    contrasts["ours_e500 vs deepdrp"]["ours_seeds_below"] = int(sum(v < mean["deepdrp"] for v in rmse["ours_e500"]))
    # Bliss (NCI synergy_check definition, as ev.bliss_part, restricted to these methods)
    surf, keys = pd.factorize(points["matrix_key"])
    i, j = points["row_i"].to_numpy() - 1, points["col_j"].to_numpy() - 1

    def grid(values):
        g = np.full((len(keys), 4, 4), np.nan)
        g[surf, i, j] = np.asarray(values, float)
        return g

    ga, gb = grid(points["mono_a_percent"]), grid(points["mono_b_percent"])
    ref = ev.references(ga[:, :, 0], gb[:, 0, :])
    observed = ev.matrix_scores(ref, grid(y * 100.0))["Bliss"]
    pcodes, _ = pd.factorize(points.drop_duplicates("matrix_key")["pair_id"], sort=True)
    Wb = ev.bootstrap_weights(len(pairs))[:, pcodes]
    pear, pboot = {}, {}
    for (m, s), p in preds.items():
        d = ev.matrix_scores(ref, grid(p * 100.0))["Bliss"]
        pear.setdefault(m, []).append(float(np.corrcoef(d, observed)[0, 1]))
        pboot.setdefault(m, []).append(ev.w_pearson_fixed(d, observed, Wb))
    bliss = {m: float(np.mean(v)) for m, v in pear.items()}
    bliss_contrast = dict(difference=bliss["ours_e500"] - bliss["deepdrp"],
                          ci95=ev.interval(np.mean(pboot["ours_e500"], axis=0) - np.mean(pboot["deepdrp"], axis=0)))
    return dict(surfaces=len(keys), pairs=len(pairs), points=len(points),
                rmse_percent={m: dict(mean=mean[m], seeds=v, surface_macro_mean=float(np.mean(macro[m])))
                              for m, v in rmse.items()},
                contrasts=contrasts, bliss_pearson=bliss, bliss_ours_e500_minus_deepdrp=bliss_contrast)


def write(result, role, exclusive):
    ev.EVAL.mkdir(parents=True, exist_ok=True)
    with (ev.EVAL / f"{STEM[role]}.json").open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, ensure_ascii=False)


def main():
    role = argparse.ArgumentParser()
    role.add_argument("--role", required=True, choices=("validation", "test"))
    role = role.parse_args().role
    t0 = time.perf_counter()
    if role == "validation":
        common.install_guard()
        points = common.load_points("validation")
        result = dict(role=role, nature="dry run", **score(points, load(role, points)))
        write(result, role, exclusive=False)
    else:
        if "ONEIL_PRED_ROOT" in os.environ:
            raise SystemExit("STOP: unset ONEIL_PRED_ROOT")
        if MARKER.exists() or (ev.EVAL / f"{STEM[role]}.json").exists():
            raise SystemExit(f"STOP: {MARKER} or the output exists; DeepDRP's test is scored only once")
        sys.addaudithook(ev._test_guard)
        sys.dont_write_bytecode = True
        load(role, common.load_points("test"))                 # every needed test prediction present before the marker
        sys.path.insert(0, str(common.OLD / "experiments/audit_revision"))
        import evaluate_oneil_confirmation as old            # import only; nothing decoded before the marker
        with MARKER.open("x", encoding="utf-8") as handle:
            json.dump(dict(created_local=time.strftime("%Y-%m-%dT%H:%M:%S%z")), handle)
        points, decode = ev.decode_test(old.build_surfaces)
        result = dict(role=role, nature="DeepDRP (adapted) one-shot test scoring after the joint scoring (descriptive)",
                      marker=str(MARKER), decode=decode, **score(points, load(role, points)))
        write(result, role, exclusive=True)
    result["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps({k: result[k] for k in ("role", "surfaces", "pairs", "contrasts", "bliss_pearson",
                                             "bliss_ours_e500_minus_deepdrp")}
                     | {"rmse_mean": {m: round(v["mean"], 3) for m, v in result["rmse_percent"].items()},
                        "surface_macro": {m: round(v["surface_macro_mean"], 3) for m, v in result["rmse_percent"].items()}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

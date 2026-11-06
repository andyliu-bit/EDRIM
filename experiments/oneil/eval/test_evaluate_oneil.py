"""Checks for the O'Neil joint scorer. Nothing here decodes a real test response or calls the real build_surfaces.

  dryrun-preds  write synthetic validation predictions for every method/seed under ONEIL_PRED_ROOT (required, must not
                be the real prediction tree): HSA = min of the anchors, each other method = truth + Gaussian noise with a
                method-specific scale, so the dry run has a known ordering (ours_e500 < ours < xgb_id < 5x5 < 4x4 < ddprism).
  unit --tmp D  test-mode control flow on a fresh prediction tree D/preds (marker and outputs land in D/eval) with a
                fake build_surfaces built from points_test_inputs.parquet plus fake responses; the real
                evaluate_oneil_confirmation module is replaced by one that raises.

  ONEIL_PRED_ROOT=<scratch>/preds PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/eval/test_evaluate_oneil.py dryrun-preds
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/eval/test_evaluate_oneil.py unit --tmp <scratch>/unit
"""
import argparse
import os
import sys
import types
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOISE = {"xgb_id": 0.08, "ours": 0.06, "ours_e500": 0.055, "deepsynba_4x4": 0.10, "deepsynba_5x5": 0.09, "ddprism": 0.11}


def write_synthetic(common, role, truth, methods=None, start=0):
    """truth: ratio-scale responses aligned with load_points(role); hsa is exact, the rest truth + noise."""
    points = common.load_points(role)
    hsa = np.minimum(points["mono_a_ratio_raw"], points["mono_b_ratio_raw"]).to_numpy()
    for m in methods or ("hsa",) + tuple(NOISE):
        for k, seed in enumerate([None] if m == "hsa" else common.SEEDS):
            rng = np.random.default_rng(start + 100 * list(common.METHODS).index(m) + k)
            pred = hsa if m == "hsa" else truth + rng.normal(0.0, NOISE[m], len(truth))
            common.write_predictions(m, seed, role, points[common.KEY].assign(prediction=pred),
                                     dict(synthetic=True, selected_epoch=-1))


def dryrun_preds():
    root = os.environ.get("ONEIL_PRED_ROOT")
    if not root or Path(root).resolve() == (HERE.parents[2] / "runs/oneil_v1/preds").resolve():
        raise SystemExit("set ONEIL_PRED_ROOT to a scratch folder")
    sys.path.insert(0, str(HERE.parent))
    import common
    common.install_guard()
    write_synthetic(common, "validation", common.load_points("validation")["response"].to_numpy())
    print("synthetic validation predictions written under", common.PRED)


def fake_builder(inputs, responses, flip_every=2, outside=2, drop_one=True, corrupt=False):
    """Old build_surfaces output shape: {"confirmation": [surface, ...]}, counters; drug order flipped on some surfaces."""
    calls = []
    frame = inputs.assign(fake_response=responses)
    surfaces = []
    for k, (key, g) in enumerate(frame.groupby("matrix_key", sort=False)):
        flip = k % flip_every == 0
        pts = []
        for r in g.itertuples(index=False):
            da, db = (format(Decimal(repr(r.dose_a_M)) * Decimal("1e6"), "f"), format(Decimal(repr(r.dose_b_M)) * Decimal("1e6"), "f"))
            p = dict(drug_a=r.drug_a_id, drug_b=r.drug_b_id, dose_a_uM=da, dose_b_uM=db, mono_a=r.mono_a_ratio_raw,
                     mono_b=r.mono_b_ratio_raw, response=float(r.fake_response), reasons=[])
            if flip:
                p.update(drug_a=r.drug_b_id, drug_b=r.drug_a_id, dose_a_uM=db, dose_b_uM=da,
                         mono_a=r.mono_b_ratio_raw, mono_b=r.mono_a_ratio_raw)
            pts.append(p)
        first = g.iloc[0]
        surfaces.append(dict(batch=first.batch, cell=first.deepsynba_cell, drugs=[first.drug_a_id, first.drug_b_id],
                             role="confirmation", points=pts, primary_eligible=True))
    for k in range(outside):                       # surfaces in cells without DepMap expression
        surfaces.append(dict(surfaces[k], cell=f"FAKE-CELL-{k}"))
    if drop_one:                                   # one non-finite response -> whole surface dropped
        surfaces[1]["points"][3] = dict(surfaces[1]["points"][3], response=None, reasons=["no_finite_combination_replicates"])
    if corrupt:
        surfaces[5]["points"][0] = dict(surfaces[5]["points"][0], mono_a=surfaces[5]["points"][0]["mono_a"] * (1 + 1e-9))

    def build_surfaces(decode_roles):
        assert decode_roles == ("confirmation",)
        calls.append(decode_roles)
        return {"confirmation": surfaces}, {"fake": True}
    return build_surfaces, calls


def expect_stop(fn, *args):
    try:
        fn(*args)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} did not stop")


def unit(tmp):
    tmp = Path(tmp).resolve()
    if tmp.exists() and any(tmp.iterdir()):
        raise SystemExit(f"{tmp} must be empty or absent")
    os.environ["ONEIL_PRED_ROOT"] = str(tmp / "preds")
    sys.path[:0] = [str(HERE), str(HERE.parent)]
    sys.modules["evaluate_oneil_confirmation"] = types.SimpleNamespace(
        build_surfaces=lambda *a: (_ for _ in ()).throw(AssertionError("real build_surfaces called")))
    import common
    import evaluate_oneil as ev
    import select_deepsynba_version as sel
    assert ev.MARKER == tmp / "eval/ONEIL_TEST_EVALUATED.marker" and common.PRED == tmp / "preds", ev.MARKER
    inputs = common.load_points("test")
    rng = np.random.default_rng(7)
    hsa = np.minimum(inputs["mono_a_ratio_raw"], inputs["mono_b_ratio_raw"]).to_numpy()
    fake_y = hsa + rng.normal(0.0, 0.10, len(inputs))      # fake test responses, in memory only
    write_synthetic(common, "test", fake_y)
    val_y = common.load_points("validation")["response"].to_numpy()
    good, calls = fake_builder(inputs, fake_y)

    msg = expect_stop(ev.run_test, good)                    # 1. DeepSynBa validation predictions missing
    assert not ev.MARKER.exists() and not calls, msg
    print("ok 1 missing DeepSynBa validation files -> refused, no marker:", msg.splitlines()[0])
    write_synthetic(common, "validation", val_y, methods=("deepsynba_4x4", "deepsynba_5x5"))
    msg = expect_stop(ev.run_test, good)                    # 2. primary json missing
    assert not ev.MARKER.exists() and not calls and "deepsynba_primary.json" in msg, msg
    print("ok 2 missing deepsynba_primary.json -> refused, no marker")
    rec = sel.write_primary()
    assert rec["primary"] == "deepsynba_5x5", rec["primary"]
    assert "exists" in expect_stop(sel.write_primary)
    print("ok 3 primary written once (5x5, as the synthetic noise implies); second write refused")
    for m, s in (("ddprism", 43), ("ours_e500", 45), ("ours", 41)):
        victim = common.pred_path(m, s, "test")
        victim.rename(victim.with_suffix(".hidden"))       # 4. one test prediction file missing
        msg = expect_stop(ev.run_test, good)
        assert not ev.MARKER.exists() and not calls and f"{m}/seed_{s}" in msg.replace("\\", "/"), msg
        victim.with_suffix(".hidden").rename(victim)
    print("ok 4 missing test prediction file (ddprism, ours_e500, ours) -> refused, no marker")
    bad, bad_calls = fake_builder(inputs, fake_y, corrupt=True)
    try:                                                    # 5. anchor mismatch aborts after the marker
        ev.run_test(bad)
        raise RuntimeError("corrupted anchor accepted")
    except AssertionError as e:
        assert "mono_a_dec" in str(e) and ev.MARKER.exists() and len(bad_calls) == 1, e
        assert not (ev.EVAL / "oneil_test_results.json").exists()
    ev.MARKER.unlink()                                      # temp marker of this unit test only
    print("ok 5 corrupted anchor (1e-9 relative) -> assertion after the marker, no outputs")
    res = ev.run_test(good)                                 # 6. clean run
    d = res["decode"]
    assert ev.MARKER.exists() and len(calls) == 1 and (ev.EVAL / "oneil_test_results.md").exists()
    assert d["outside_30_cells"] == 2 and d["dropped_surfaces_response_not_finite"] == 1 and not d["pairs_lost_entirely"], d
    assert res["surfaces"] == inputs["matrix_key"].nunique() - 1 and res["points"] == len(inputs) - 16
    assert all(v["passed"] for v in res["verdicts"].values()), res["verdicts"]
    assert abs(res["methods"]["ours"]["mean_rmse_percent"] - 6.0) < 0.1, res["methods"]["ours"]
    assert abs(res["methods"]["ours_e500"]["mean_rmse_percent"] - 5.5) < 0.1, res["methods"]["ours_e500"]
    cs = res["card_setting_60_epochs"]                      # verdicts / contrasts per Ours: e500 headline, 60 epochs block
    assert res["ours_method"] == "ours_e500" and cs["ours_method"] == "ours", (res["ours_method"], cs["ours_method"])
    for b, c in res["contrasts"].items():
        m = res["methods"]
        assert abs(c["relative_difference"] - (m["ours_e500"]["mean_rmse_percent"] / m[b]["mean_rmse_percent"] - 1)) < 1e-12, b
    for b, c in cs["contrasts"].items():
        m = res["methods"]
        assert abs(c["relative_difference"] - (m["ours"]["mean_rmse_percent"] / m[b]["mean_rmse_percent"] - 1)) < 1e-12, b
    assert "ours" in res["contrasts"] and "ours_e500" not in cs["contrasts"] and "ours" not in cs["contrasts"]
    assert res["contrasts"]["ours"]["ci95"][1] < 0 and "ours" not in {v["comparator"] for v in res["verdicts"].values()}
    assert all(v["passed"] for v in cs["verdicts"].values()), cs["verdicts"]
    bl = res["bliss"]
    assert bl["contrasts"][rec["primary"]]["ours_method"] == "ours_e500" and cs["bliss_contrasts"] == bl["card_setting_contrasts"]
    assert abs(cs["bliss_contrasts"]["xgb_id"]["difference"] - (bl["methods"]["ours"]["mean_pearson"]
                                                             - bl["methods"]["xgb_id"]["mean_pearson"])) < 1e-12
    md = (ev.EVAL / "oneil_test_results.md").read_text(encoding="utf-8")
    assert "Ours（500 轮＋早停，D13）" in md and "卡原设定（60 轮）" in md and "Ours（卡原设定 60 轮） 各种子" in md
    print("ok 6 clean run: marker + outputs written; 2 outside-cell surfaces skipped, 1 surface dropped;",
          {k: round(v["mean_rmse_percent"], 3) for k, v in res["methods"].items()},
          {k: (round(v["relative_difference"], 4), [round(x, 4) for x in v["ci95"]], v["passed"]) for k, v in res["verdicts"].items()},
          "card 60 epochs:", {k: v["passed"] for k, v in cs["verdicts"].items()},
          "e500 vs 60:", round(res["contrasts"]["ours"]["relative_difference"], 4), [round(x, 4) for x in res["contrasts"]["ours"]["ci95"]])
    msg = expect_stop(ev.run_test, good)                    # 7. second run
    assert len(calls) == 1 and "only once" in msg, msg
    print("ok 7 second run refused before decoding:", msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dryrun-preds", "unit"))
    parser.add_argument("--tmp")
    args = parser.parse_args()
    dryrun_preds() if args.mode == "dryrun-preds" else unit(args.tmp)

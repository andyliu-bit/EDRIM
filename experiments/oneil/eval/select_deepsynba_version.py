"""Card E v2 (DEVIATIONS D13): fix the DeepSynBa primary comparator before the joint test scoring.

Primary = the version (deepsynba_4x4 / deepsynba_5x5) with the lower 5-seed mean validation surface-macro RMSE (%).
Each seed's value is recomputed from its saved validation predictions (the selected epoch) with
common.surface_macro_percent, so it does not depend on how a builder names fields in its meta JSON; the meta file
must exist and its scalar fields are copied into the record. Ties keep 4x4 (the authors' own O'Neil setting).
Refuses if any seed is missing and never overwrites runs/oneil_v1/eval/deepsynba_primary.json.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/eval/select_deepsynba_version.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

VERSIONS = ("deepsynba_4x4", "deepsynba_5x5")
EVAL = common.PRED.parent / "eval"          # runs/oneil_v1/eval; follows ONEIL_PRED_ROOT in smoke tests
PRIMARY_JSON = EVAL / "deepsynba_primary.json"


def choose():
    points = common.load_points("validation")
    versions = {}
    for method in VERSIONS:
        seeds = {}
        for seed in common.SEEDS:
            path = common.pred_path(method, seed, "validation")
            meta_path = path.with_name("validation_meta.json")
            if not (path.exists() and meta_path.exists()):
                raise SystemExit(f"STOP: {method} seed {seed}: validation predictions or meta missing in {path.parent}")
            pred = points[common.KEY].merge(common.read_predictions(method, seed, "validation")[common.KEY + ["prediction"]],
                                            on=common.KEY, how="left", validate="one_to_one")["prediction"].to_numpy(float)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            seeds[str(seed)] = dict(val_surface_macro_percent=common.surface_macro_percent(points, pred),
                                    meta_scalars={k: v for k, v in meta.items() if not isinstance(v, (list, dict))})
        versions[method] = dict(seeds=seeds, mean_val_surface_macro_percent=float(
            np.mean([s["val_surface_macro_percent"] for s in seeds.values()])))
    m4, m5 = (versions[v]["mean_val_surface_macro_percent"] for v in VERSIONS)
    primary = "deepsynba_5x5" if m5 < m4 else "deepsynba_4x4"
    return dict(primary=primary, other=[v for v in VERSIONS if v != primary][0], versions=versions,
                rule="lower 5-seed mean validation surface-macro RMSE (%), 16 combination cells; ties keep 4x4",
                pred_root=str(common.PRED))


def write_primary():
    if PRIMARY_JSON.exists():
        raise SystemExit(f"STOP: {PRIMARY_JSON} exists; the primary version is written once")
    record = dict(choose(), written_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    EVAL.mkdir(parents=True, exist_ok=True)
    with PRIMARY_JSON.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, ensure_ascii=False)
    return record


def main():
    common.install_guard()
    record = write_primary()
    print(json.dumps({v: record["versions"][v]["mean_val_surface_macro_percent"] for v in VERSIONS}),
          "primary:", record["primary"], "->", PRIMARY_JSON, flush=True)


if __name__ == "__main__":
    main()

"""D10 C2: score comboKR 2.0 (public split, iteration count chosen on split 2 by combokr_select.py) on the test set
once (marker COMBOKR_TEST_EVALUATED.marker), against the saved BiSurfGNN v2, XGBoost+identity, DeepSynBa v2 and
lookup HSA with the same drug-pair bootstrap as D7/D8. comboKR is deterministic: one run.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/comparator_test/combokr_test.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

import numpy as np  # noqa: E402

PUB = common.gate.ROOT / "runs/model_improvement/combokr2_original_v1/public_split"
OUT, MARKER = common.V2 / "combokr_test", common.V2 / "COMBOKR_TEST_EVALUATED.marker"


def main():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with python -I -B")
    if MARKER.exists():
        raise SystemExit("comboKR test set already scored")
    sel = json.loads((PUB / "validation/selection.json").read_text(encoding="utf-8"))
    k = sel["selected_iters"]
    z = np.load(PUB / f"test/pred_iters{k}.npz")
    frame, matrices = common.load_test()
    keys = frame.matrix_key.to_numpy()[::9]
    if not np.array_equal(z["matrix_key"], keys.astype(str)):
        raise RuntimeError("comboKR test predictions are not aligned with the evaluable test frame")
    pred = z["pred"].astype(float).reshape(-1)          # (matrices, 9) row-major i, j = frame order
    if not np.isfinite(pred).all():
        raise RuntimeError(f"{int((~np.isfinite(pred)).sum())} non-finite comboKR predictions")
    r = common.score_once("combokr2", {0: pred}, frame, matrices, OUT, MARKER,
                          {"selected_iters": k, "validation": sel["per_iteration"][str(k)] if str(k) in sel["per_iteration"]
                           else sel["per_iteration"][k], "registered": "DEVIATIONS.md D10"})
    print(json.dumps({key: r[key] for key in ("point", "comparisons", "evaluable")}, indent=1))


if __name__ == "__main__":
    main()

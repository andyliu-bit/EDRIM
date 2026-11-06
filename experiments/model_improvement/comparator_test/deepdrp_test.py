"""DeepDRP (adapted), seeds 41-45 at the global learning rate chosen on split 2 (docs/DEEPDRP_ADAPTATION_CARD_20261103.md).
Each seed's saved model predicts the test frame (mean of both drug orders); the test set is then scored once
(marker DEEPDRP_TEST_EVALUATED.marker).

  .venv/Scripts/python.exe -I -B experiments/model_improvement/comparator_test/deepdrp_test.py --global-lr 1e-4
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "deepdrp_adapted"))
import common  # noqa: E402
import deepdrp as dd  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT, MARKER = common.V2 / "deepdrp_test", common.V2 / "DEEPDRP_TEST_EVALUATED.marker"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-lr", type=float, required=True)
    args = ap.parse_args()
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with python -I -B")
    if MARKER.exists():
        raise SystemExit("DeepDRP test set already scored")
    runs = {s: dd.RUNS / f"lr{args.global_lr:g}_seed{s}" for s in common.SEEDS}
    missing = [s for s, r in runs.items() if not (r / "model_selected.pt").exists()]
    if missing:
        raise SystemExit(f"seeds without a selected model: {missing}")
    device = torch.device("cuda")
    frame, matrices = common.load_test()
    keys = np.repeat([k for k, _, _ in matrices], 9)
    if not np.array_equal(frame.matrix_key.to_numpy(str), keys.astype(str)):
        raise RuntimeError("Test frame is not grouped in blocks of 9 rows per matrix")
    codes = np.repeat(np.arange(len(matrices)), 9)
    preds, val = {}, {}
    for s, r in runs.items():
        g, local, state = dd.load_model(r / "model_selected.pt", device)
        cellx, genes = dd.cell_features(state["cells"])
        if genes != state["genes"]:
            raise RuntimeError("Cell features differ from training")
        M = dd.Matrices(frame, {d: i for i, d in enumerate(state["drugs"])}, {c: i for i, c in enumerate(state["cells"])},
                        tuple(state["dose_range"]), state["control_percent"], device, codes=codes)
        preds[s] = 100.0 * M.frame_order(dd.predict_final(g, local, dd.drug_tokens(state["drugs"]).to(device),
                                                          cellx.to(device), M))
        val[s] = json.loads((r / "metrics.json").read_text(encoding="utf-8"))["validation"]["combination_rmse_percent"]
        np.savez_compressed(r / "test_predictions.npz", prediction=preds[s], point_ids=frame.source_combodrugseq.to_numpy())
    res = common.score_once("deepdrp_adapted", preds, frame, matrices, OUT, MARKER,
                            {"validation_combination_rmse_percent": val, "global_lr": args.global_lr,
                             "launch_card": "docs/DEEPDRP_ADAPTATION_CARD_20261103.md"})
    print(json.dumps({k: res[k] for k in ("point", "comparisons", "seed_sd", "validation_combination_rmse_percent")},
                     indent=1))


if __name__ == "__main__":
    main()

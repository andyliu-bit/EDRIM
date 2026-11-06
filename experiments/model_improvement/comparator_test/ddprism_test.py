"""D10 C1: DD-PRiSM measured-mono adapter, seeds 41-45, each selected on split 2 by launch_validation.py. The saved
selected state predicts the test frame in eval mode (no BatchNorm refresh, adapter deviation D4); the test set is
then scored once (marker DDPRISM_TEST_EVALUATED.marker).

  .venv/Scripts/python.exe -I -B experiments/model_improvement/comparator_test/ddprism_test.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ddprism_measured_mono"))
import common  # noqa: E402
import ddprism_mm as dm  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

RUNS = common.V2 / "ddprism_measured_mono"
OUT, MARKER = common.V2 / "ddprism_test", common.V2 / "DDPRISM_TEST_EVALUATED.marker"


def main():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with python -I -B")
    if MARKER.exists():
        raise SystemExit("DD-PRiSM test set already scored")
    for s in common.SEEDS:
        if not (RUNS / f"seed_{s}/model_selected.pt").exists():
            raise SystemExit(f"seed {s} has no selected model yet")
    cfg = json.loads((HERE.parent / "ddprism_measured_mono/formal_config.json").read_text(encoding="utf-8"))
    device = dm.model_device(cfg)
    tables = dm.load_tables()
    drugs, cells = dm.entity_index(tables)
    gene_sets, _, _ = dm.cell_gene_features(cells)
    _, Combo = dm.author_classes(len(gene_sets))
    frame, matrices = common.load_test()
    if not (set(frame.drug_a_id) | set(frame.drug_b_id)) <= set(drugs) or not set(frame.biological_cell_id) <= set(cells):
        raise RuntimeError("Test entities missing from the DD-PRiSM index")
    data = dm.CombinationData(frame, drugs, cells, device, swap=False)
    preds, val = {}, {}
    for s in common.SEEDS:
        state = torch.load(RUNS / f"seed_{s}/model_selected.pt", map_location=device, weights_only=True)
        combo = Combo().to(device)
        combo.load_state_dict(state["combo"])
        combo.eval()
        att = state["attention"].to(device)
        out = torch.empty_like(data.y)
        with torch.no_grad():
            for lo in range(0, data.n, 8192):
                idx = torch.arange(lo, min(lo + 8192, data.n), device=device)
                out[idx] = combo(data.features(att, idx))[2]
        preds[s] = 100.0 * out.cpu().numpy().ravel().astype(float)
        val[s] = json.loads((RUNS / f"seed_{s}/metrics.json").read_text(encoding="utf-8"))["validation"]["combination_rmse_percent"]
        np.savez_compressed(RUNS / f"seed_{s}/test_predictions.npz", prediction=preds[s],
                            point_ids=frame.source_combodrugseq.to_numpy())
    r = common.score_once("ddprism_measured_mono", preds, frame, matrices, OUT, MARKER,
                          {"validation_combination_rmse_percent": val, "registered": "DEVIATIONS.md D10"})
    print(json.dumps({k: r[k] for k in ("point", "comparisons", "seed_sd", "validation_combination_rmse_percent")}, indent=1))


if __name__ == "__main__":
    main()

"""Secondary endpoints of DeepDRP (adapted) on the NCI test split, pre-specified in docs/DEEPDRP_ADAPTATION_CARD_20261103.md:
Bliss synergy-score Pearson and top-5% hit rate from the saved test predictions (synergy_check definitions, the
matrix's own monotherapy edges as reference, as in score_t_batch T4). EDRIM's saved test predictions go through the
same code as a check against the reported 0.808 / 59.5%; the EDRIM - DeepDRP Pearson difference gets the drug-pair
bootstrap of synergy_check (2,000 draws, seed 20261013). Runs once (refuses if its output exists).

  .venv/Scripts/python.exe -I -B experiments/model_improvement/comparator_test/deepdrp_synergy_test.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "synergy_check"), str(HERE.parent / "deepdrp_adapted")]
import common  # noqa: E402
import deepdrp as dd  # noqa: E402
import synergy_check as sc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = common.V2 / "deepdrp_test" / "synergy_secondary.json"


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; computed once")
    frame, matrices = common.load_test()
    n, ids = len(matrices), frame.source_combodrugseq.to_numpy()
    truth = np.stack([m for _, m, _ in matrices])
    ref = sc.references(truth[:, 1:, 0], truth[:, 0, 1:])
    observed = sc.matrix_scores(ref, frame.response_percent.to_numpy(float).reshape(n, 3, 3))["Bliss"]
    edrim = np.load(common.V2 / "test_eval/test_combination_predictions_bisurf_v2.npz", allow_pickle=False)
    derived = {}
    for s in common.SEEDS:
        z = np.load(dd.RUNS / f"lr0.0003_seed{s}/test_predictions.npz", allow_pickle=False)
        if not np.array_equal(z["point_ids"], ids):
            raise RuntimeError("DeepDRP test predictions misaligned")
        derived[("deepdrp", s)] = sc.matrix_scores(ref, z["prediction"].reshape(n, 3, 3))["Bliss"]
        derived[("edrim", s)] = sc.matrix_scores(ref, edrim[f"seed_{s}"].reshape(n, 3, 3))["Bliss"]
    k = sc.k_frac(n, 0.05)
    pearson = {key: float(np.corrcoef(d, observed)[0, 1]) for key, d in derived.items()}
    hit = {key: sc.precision_at_k(d, observed, k) for key, d in derived.items()}
    codes, pairs = pd.factorize(frame.pair_id.to_numpy()[::9], sort=True)
    W = sc.bootstrap_weights(codes, len(pairs))[:, codes]                  # pair counts -> per-matrix weights
    boot = {m: np.mean([sc.w_pearson_fixed(derived[(m, s)], observed, W) for s in common.SEEDS], axis=0)
            for m in ("edrim", "deepdrp")}
    diff = boot["edrim"] - boot["deepdrp"]
    result = dict(
        matrices=n, pairs=len(pairs), top5_k=k, bootstrap=dict(unit="test drug pair", draws=sc.BOOT_N, seed=sc.BOOT_SEED),
        bliss_pearson={m: dict(mean=float(np.mean([pearson[(m, s)] for s in common.SEEDS])),
                               seeds=[pearson[(m, s)] for s in common.SEEDS]) for m in ("edrim", "deepdrp")},
        top5_hit_rate={m: dict(mean=float(np.mean([hit[(m, s)] for s in common.SEEDS])),
                               seeds=[hit[(m, s)] for s in common.SEEDS]) for m in ("edrim", "deepdrp")},
        edrim_minus_deepdrp_pearson=dict(difference=float(np.mean([pearson[("edrim", s)] - pearson[("deepdrp", s)]
                                                                   for s in common.SEEDS])),
                                         ci95=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

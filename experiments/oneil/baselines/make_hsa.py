"""O'Neil HSA baseline (method "hsa", card E v2 method 1; DEVIATIONS D13): min of the two interpolated monotherapy
anchors (mono_response_a/b, ratio scale), no training, deterministic. Writes validation and test predictions; test
uses the response-free input table only. Refuses to run if either prediction file already exists.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/baselines/make_hsa.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

common.install_guard()
import numpy as np  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    existing = [p for p in (common.pred_path("hsa", None, r) for r in ("validation", "test")) if p.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite: {existing}")
    for role in ("validation", "test"):
        frame = common.load_points(role)
        frame["prediction"] = np.minimum(frame.mono_response_a, frame.mono_response_b).to_numpy(float)
        meta = {"method": "hsa", "seed": None, "role": role, "rule": "min(mono_response_a, mono_response_b)",
                "points": len(frame), "surfaces": int(frame.matrix_key.nunique()),
                "created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "card": "ONEIL_COMPARISON_CARD_20261020 v2"}
        if role == "validation":  # self-check against the stage-1 lookup column; description only
            assert np.allclose(100 * frame.prediction, frame.hsa_lookup_percent, rtol=0, atol=1e-9)
            meta["validation_surface_macro_percent"] = common.surface_macro_percent(frame, frame.prediction.to_numpy())
            meta["validation_pooled_rmse_percent"] = common.pooled_rmse_percent(frame.response, frame.prediction)
        path = common.write_predictions("hsa", None, role, frame, meta)
        print(f"{role}: {len(frame)} points -> {path}"
              + (f" val_surface_macro={meta['validation_surface_macro_percent']:.4f}% "
                 f"val_pooled={meta['validation_pooled_rmse_percent']:.4f}%" if role == "validation" else ""), flush=True)


if __name__ == "__main__":
    main()

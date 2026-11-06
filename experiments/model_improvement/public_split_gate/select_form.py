"""PublicSplitGate-v1 step S3: choose the BiSurfGNN output form from the two seed-41 validation runs.

Rule (launch card section 5, fixed before results): the form with the lower validation pooled combination RMSE
(percent); if the two differ by less than 0.5% (relative to the anchored run), keep the anchored default.
Writes runs/model_improvement/public_split_gate_v1/s3_selection.json exclusively.
"""
from __future__ import annotations

import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[3] / "runs/model_improvement/public_split_gate_v1"


def main():
    runs = {form: RUN / f"s3_{form}_r2" for form in ("anchored", "additive")}
    metrics = {}
    for form, run in runs.items():
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "passed":
            raise RuntimeError(f"S3 {form} run did not pass its end-of-run checks")
        metrics[form] = json.loads((run / f"bisurf_{form}_seed_41/metrics.json").read_text(encoding="utf-8"))
    a = metrics["anchored"]["validation_pooled_rmse_percent"]
    b = metrics["additive"]["validation_pooled_rmse_percent"]
    relative = (b - a) / a
    chosen = "anchored" if abs(relative) < 0.005 else ("additive" if b < a else "anchored")
    record = {"rule": "lower validation pooled combination RMSE; |relative difference| < 0.5% keeps anchored",
              "validation_pooled_rmse_percent": {"anchored": a, "additive": b},
              "validation_surface_macro_rmse": {f: metrics[f]["validation_surface_macro_rmse"] for f in metrics},
              "best_epoch": {f: metrics[f]["best_epoch"] for f in metrics},
              "stopped_by": {f: metrics[f]["stopped_by"] for f in metrics},
              "additive_relative_to_anchored": relative, "chosen_form": chosen}
    with (RUN / "s3_selection.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()

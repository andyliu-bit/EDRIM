"""Descriptive diagnostics on already-used, frozen development rows only."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def diagnose(output):
    reference = ROOT / "runs/model_improvement/comparator_gpu_v2"
    ids = pd.read_csv(reference / "validation_ids.csv")
    source = ROOT.parent / "bisurfgnn/data/processed/one_new_scaffold_benchmark_unclipped/learned_point_shards/fold_0/validation.parquet"
    frame = pd.read_parquet(source, filters=[("source_combodrugseq", "in", ids.source_combodrugseq.tolist())],
        columns=["source_combodrugseq", "cell_id", "drug_a_id", "drug_b_id", "dose_a_model", "dose_b_model",
                 "mono_response_a", "mono_response_b", "response"])
    frame = ids.merge(frame, on="source_combodrugseq", validate="one_to_one")
    paths = [reference / f"bisurf_mono_v2_seed_{s}_primary/predictions.csv" for s in range(41,46)]
    predictions = [pd.read_csv(p) for p in paths]
    for pred in predictions:
        assert pred.source_row.tolist() == ids.source_combodrugseq.tolist()
        assert pred.surface_id.tolist() == ids.surface_replicate_id.tolist()
        np.testing.assert_allclose(pred.target, frame.response, atol=1e-6)
    frame["prediction"] = np.stack([p.prediction.to_numpy() for p in predictions]).mean(0)
    frame["reference"] = frame[["mono_response_a", "mono_response_b"]].min(axis=1)
    frame["squared_error"] = (frame.prediction-frame.response)**2
    frame["absolute_error"] = abs(frame.prediction-frame.response)
    frame["anchor_group"] = pd.cut(frame.reference, [-np.inf,.01,.2,.8,np.inf],
        labels=["anchor_le_0.01", "anchor_0.01_0.2", "anchor_0.2_0.8", "anchor_gt_0.8"])
    frame["dose_group"] = pd.cut(frame[["dose_a_model","dose_b_model"]].max(axis=1),
        [-np.inf,3.,6.,np.inf], labels=["dose_le_3", "dose_3_6", "dose_gt_6"])
    rows=[]
    for dimension in ("anchor_group", "dose_group"):
        for group, part in frame.groupby(dimension, observed=True):
            rows.append(dict(dimension=dimension, group=str(group), points=len(part),
                surfaces=part.surface_replicate_id.nunique(), point_rmse=float(np.sqrt(part.squared_error.mean())),
                mae=float(part.absolute_error.mean()), reference_mean=float(part.reference.mean())))
    groups=pd.DataFrame(rows); groups.to_csv(output/"diagnostic_groups.csv", index=False)
    frame.nlargest(25,"absolute_error").to_csv(output/"diagnostic_largest_errors.csv",index=False)
    selected=groups[(groups.dimension=="dose_group") & (groups.points>=100)]
    ratio=float(selected.point_rmse.max()/max(selected.point_rmse.min(),1e-12)) if len(selected)>=2 else None
    low=frame[frame.reference<=.01]
    total_error=float(frame.squared_error.sum())
    result=dict(scope="already_observed_fold0_development_descriptive_only", rows=len(frame),
        low_anchor_points=len(low), low_anchor_squared_error_share=float(low.squared_error.sum()/total_error),
        low_anchor_mean_local_logit_derivative=float((low.reference*(1-low.reference/3)).mean()),
        dose_rmse_ratio=ratio, dose_experiment_eligible=ratio is not None and ratio>=1.5,
        dose_rule="at_least_two_fixed_dose_bins_with_100_points_and_rmse_ratio_ge_1.5",
        interpretation="dose grouping is descriptive and confounded; it does not prove a dose mechanism",
        test_response_rows_read=0)
    (output/"diagnostic_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    groups.to_csv(ROOT/"reports/tables/remaining_suite_diagnostics.csv",index=False)
    return result


if __name__=="__main__":
    output=ROOT/"runs/model_improvement/remaining_suite_v1"
    output.mkdir(parents=True,exist_ok=True)
    print(json.dumps(diagnose(output)),flush=True)

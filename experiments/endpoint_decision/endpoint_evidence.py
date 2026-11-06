"""Endpoint decision evidence for the M1 open item.

`NCI_ALMANAC_M1_REPORT.md` predeclared three endpoint candidates and one rule:
"Do not choose the target using test performance." This script produces the
descriptive evidence for that decision on a development cohort that is never a
test fold in any of the five One-new-scaffold splits. Since the 2026-09-11 audit,
an explicit physical development-only table is mandatory and its group IDs are
checked before response columns are read. Historical runs loaded the full table
first and must not be described as zero physical test-response access.

It writes nothing into the frozen data or run directories; it prints a report
and optionally dumps a JSON summary.

Usage:
    python experiments/endpoint_decision/endpoint_evidence.py --development-points DEV.parquet [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from bisurfgnn.learned_benchmark import _file_record

ROOT = Path(__file__).resolve().parents[2]
POINTS = ROOT / "data" / "processed" / "nci_almanac" / "points.parquet"
GROUPS = ROOT / "data" / "processed" / "one_new_scaffold_benchmark" / "group_assignments.parquet"

POINT_COLUMNS = [
    "prediction_group_id",
    "surface_replicate_id",
    "primary_metric_mask",
    "response_mask",
    "percent_growth_raw",
    "percent_growth_no_tz_raw",
    "response",
    "response_clipped_high",
    "conc_index_a",
    "conc_index_b",
    "mono_response_a",
    "mono_response_b",
    "mono_response_a_raw_ratio",
    "mono_response_b_raw_ratio",
    "mono_mask_a",
    "mono_mask_b",
]


def development_groups() -> set[str]:
    """Groups that are never a test fold anywhere, and are used at least once."""
    role_columns = [f"role_fold_{k}" for k in range(5)]
    frame = pq.read_table(GROUPS, columns=["prediction_group_id"] + role_columns).to_pandas()
    roles = frame[role_columns].values
    never_test = ~(roles == "test").any(axis=1)
    ever_used = np.isin(roles, ["train", "validation"]).any(axis=1)
    return set(frame.loc[never_test & ever_used, "prediction_group_id"])


def read_development_points(path: Path, allowed_groups: set[str]) -> tuple[pd.DataFrame, dict]:
    """Fail before decoding responses if this file contains a non-development ID."""
    path = path.resolve()
    if path == POINTS.resolve():
        raise ValueError("Full response table is forbidden; supply a physical development-only table")
    ids = pq.read_table(path, columns=["prediction_group_id"]).to_pandas()["prediction_group_id"]
    if ids.empty or ids.isna().any() or not set(ids).issubset(allowed_groups):
        raise ValueError("Development input contains unauthorized or empty group inventory")
    before = _file_record(path)
    table = pq.read_table(path, columns=POINT_COLUMNS).to_pandas()
    if not table.prediction_group_id.equals(ids) or before != _file_record(path):
        raise ValueError("Development input changed during guarded read")
    return table, before


def surface_macro_rmse(frame: pd.DataFrame, target: str, prediction: str) -> float:
    squared = (frame[target] - frame[prediction]) ** 2
    per_surface = squared.groupby(frame["surface_replicate_id"]).mean()
    return float(np.sqrt(per_surface).mean())


def aligned_replicate_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two aligned frames holding the first and second replicate of each grid point."""
    replicate_counts = frame.groupby("prediction_group_id")["surface_replicate_id"].nunique()
    repeated = set(replicate_counts[replicate_counts > 1].index)
    subset = frame[frame["prediction_group_id"].isin(repeated)].copy()
    subset["grid"] = (
        subset["conc_index_a"].astype(str) + "_" + subset["conc_index_b"].astype(str)
    )
    subset = subset.sort_values(["prediction_group_id", "grid", "surface_replicate_id"])
    subset["rank"] = subset.groupby(["prediction_group_id", "grid"]).cumcount()
    first = subset[subset["rank"] == 0].set_index(["prediction_group_id", "grid"])
    second = subset[subset["rank"] == 1].set_index(["prediction_group_id", "grid"])
    shared = first.index.intersection(second.index)
    return first.loc[shared], second.loc[shared]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--development-points", type=Path, required=True)
    args = parser.parse_args()
    if args.json is not None and args.json.exists():
        raise SystemExit("Refusing to overwrite an existing endpoint report")

    dev = development_groups()
    table, source_record = read_development_points(args.development_points, dev)
    report: dict[str, object] = {
        "scope": "development_cohort_never_test_in_any_fold",
        "test_response_rows_read": 0,
        "source_points_parquet_rows": int(len(table)),
        "development_groups": len(dev),
        "source": source_record,
        "availability_scope": "physical_development_view_only_not_full_release",
        "access_contract_version": "development_ids_before_response_decode_v2",
    }

    # Availability is compared on all rows of this DEVELOPMENT view, not the release.
    tz_present = table["percent_growth_raw"].notna()
    notz_present = table["percent_growth_no_tz_raw"].notna()
    report["availability"] = {
        "percent_growth_raw_non_null": int(tz_present.sum()),
        "percent_growth_no_tz_raw_non_null": int(notz_present.sum()),
        "rows_only_time_zero_endpoint_has_label": int((tz_present & ~notz_present).sum()),
        "response_mask_equals_no_tz_availability": bool(
            (table["response_mask"] == notz_present).all()
        ),
    }

    points = table[
        table["prediction_group_id"].isin(dev)
        & table["primary_metric_mask"]
        & table["response_mask"]
        & table["mono_mask_a"]
        & table["mono_mask_b"]
    ].copy()
    points["target_clipped"] = points["response"]
    points["target_no_tz"] = points["percent_growth_no_tz_raw"] / 100.0
    points["target_time_zero"] = points["percent_growth_raw"] / 100.0
    points["hsa_clipped"] = np.minimum(points["mono_response_a"], points["mono_response_b"])
    points["hsa_raw"] = np.minimum(
        points["mono_response_a_raw_ratio"], points["mono_response_b_raw_ratio"]
    )
    report["development_primary_points"] = int(len(points))

    no_tz = points["target_no_tz"]
    time_zero = points["target_time_zero"]
    report["distribution"] = {
        "no_tz_min": float(no_tz.min()),
        "no_tz_max": float(no_tz.max()),
        "no_tz_fraction_above_one": float((no_tz > 1).mean()),
        "no_tz_fraction_below_zero": float((no_tz < 0).mean()),
        "time_zero_min": float(time_zero.min()),
        "time_zero_max": float(time_zero.max()),
        "time_zero_fraction_above_one": float((time_zero > 1).mean()),
        "time_zero_fraction_below_zero": float((time_zero < 0).mean()),
    }

    # How far above one do the clipped points actually sit?
    report["excess_above_one"] = {
        f"fraction_above_{threshold}": float((no_tz > threshold).mean())
        for threshold in (1.0, 1.05, 1.1, 1.2, 1.5)
    }
    dose_level = points["conc_index_a"] + points["conc_index_b"]
    report["clip_rate_by_dose_level"] = {
        int(level): float(value)
        for level, value in (no_tz > 1).groupby(dose_level).mean().items()
    }

    # Resolution in the pharmacologically decisive strong-effect region.
    strong = no_tz < 0.2
    report["strong_effect_resolution"] = {
        "fraction_of_points": float(strong.mean()),
        "no_tz_sd": float(no_tz[strong].std()),
        "time_zero_sd": float(time_zero[strong].std()),
    }

    # Clipping creates an exact-1 point mass shared by target and HSA reference.
    both_one = (points["target_clipped"] == 1.0) & (points["hsa_clipped"] == 1.0)
    report["clipping_and_reference"] = {
        "fraction_target_and_hsa_both_exactly_one": float(both_one.mean()),
        "hsa_macro_rmse_clipped_all_points": surface_macro_rmse(
            points, "target_clipped", "hsa_clipped"
        ),
        "hsa_macro_rmse_clipped_excluding_both_one": surface_macro_rmse(
            points[~both_one], "target_clipped", "hsa_clipped"
        ),
        "hsa_macro_rmse_unclipped_no_tz": surface_macro_rmse(points, "target_no_tz", "hsa_raw"),
    }

    first, second = aligned_replicate_pairs(points)
    report["replicate_pairs"] = int(len(first))
    noise: dict[str, dict[str, float]] = {}
    for label, column in (
        ("no_tz_unclipped", "target_no_tz"),
        ("time_zero", "target_time_zero"),
        ("no_tz_clipped_current", "target_clipped"),
    ):
        left = first[column].values
        right = second[column].values
        difference = left - right
        per_observation = float(difference.std() / np.sqrt(2))
        span = float(np.percentile(left, 99) - np.percentile(left, 1))
        noise[label] = {
            "sd_of_replicate_difference": float(difference.std()),
            "per_observation_noise_sd": per_observation,
            "central_98_percent_range": span,
            "noise_to_range_ratio": per_observation / span,
        }
    report["replicate_noise"] = noise

    clipped_both = first["response_clipped_high"].values & second["response_clipped_high"].values
    clipped_difference = first["target_clipped"].values - second["target_clipped"].values
    report["clipping_destroys_replicate_signal"] = {
        "fraction_of_pairs_with_both_clipped": float(clipped_both.mean()),
        "sd_of_difference_when_both_clipped": float(clipped_difference[clipped_both].std()),
        "sd_of_difference_when_neither_clipped": float(clipped_difference[~clipped_both].std()),
    }

    print(json.dumps(report, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

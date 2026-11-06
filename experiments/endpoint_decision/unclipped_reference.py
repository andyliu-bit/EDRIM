"""Zero-parameter references under the D1 unclipped endpoint, validation only.

`ENDPOINT_DECISION.md` recommends D1: drop the upper clip and keep the
no-time-zero ratio as the primary target. This script recomputes the product and
HSA zero-parameter references under both the current clipped target and the D1
unclipped target, so the 5% gate position can be compared before deciding
whether to rerun the learned 5x3 runs.

The scored rows come from validation shards only; raw anchors/targets are read
from the SAME fold's D1 train/validation shards, never the full response table.
Historical versions loaded a full raw lookup and their zero-read wording was
incorrect as a physical access claim. The endpoint was selected on a
development cohort, so recomputing the test references before D1 is frozen would
turn a test result into part of the endpoint decision. Test references are
recomputed only after D1 is executed.

Usage:
    python experiments/endpoint_decision/unclipped_reference.py [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from bisurfgnn.benchmark_config import load_benchmark_config
from bisurfgnn.learned_benchmark import (
    FrozenLearnedPointShards, _file_record, _verify_recorded_file,
    load_learned_point_shards,
)

ROOT = Path(__file__).resolve().parents[2]
POINTS = ROOT / "data" / "processed" / "nci_almanac" / "points.parquet"
SCAFFOLD_SHARDS = ROOT / "data" / "processed" / "one_new_scaffold_benchmark" / "learned_point_shards"
WARM_SHARDS = ROOT / "data" / "processed" / "warm_pairout_development"

GATE = 0.05


def raw_lookup(fold: int, shards: FrozenLearnedPointShards) -> pd.DataFrame:
    """Read raw ratios from D1 physical development roles, not mixed-role data."""
    if shards.manifest.get("endpoint") != "unclipped_PERCENTGROWTHNOTZ_over_100":
        raise ValueError("Raw lookup requires the D1 unclipped endpoint")
    parts = []
    for role in ("train", "validation"):
        path, record = shards.role_path(fold, role), shards.role_record(fold, role)
        if path.name != f"{role}.parquet":
            raise ValueError("Raw lookup requires a physical development role file")
        _verify_recorded_file(path, record, "D1 development raw lookup")
        table = pq.read_table(path, columns=[
            "source_combodrugseq", "response", "mono_response_a", "mono_response_b",
        ]).to_pandas()
        if len(table) != record["rows"]:
            raise ValueError("D1 raw lookup row count changed")
        parts.append(table)
    combined = pd.concat(parts, ignore_index=True)
    if combined.source_combodrugseq.duplicated().any():
        raise ValueError("D1 train/validation raw source rows overlap")
    return combined.rename(columns={
        "response": "target_unclipped", "mono_response_a": "mono_response_a_raw_ratio",
        "mono_response_b": "mono_response_b_raw_ratio",
    }).set_index("source_combodrugseq")


def surface_macro_rmse(frame: pd.DataFrame, target: str, prediction: str) -> float:
    squared = (frame[target] - frame[prediction]) ** 2
    per_surface = squared.groupby(frame["surface_replicate_id"]).mean()
    return float(np.sqrt(per_surface).mean())


def evaluate(shard: Path, raw: pd.DataFrame) -> dict[str, float]:
    if shard.name != "validation.parquet":
        raise ValueError("Reference evaluation only accepts validation.parquet")
    frame = pq.read_table(shard).to_pandas()
    frame = frame[
        frame["primary_metric_mask"]
        & frame["response_mask"]
        & frame["mono_mask_a"]
        & frame["mono_mask_b"]
    ].copy()
    joined = frame.join(raw, on="source_combodrugseq", how="left", validate="one_to_one")
    missing = int(joined["target_unclipped"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} rows in {shard} could not be joined to raw responses")

    joined["hsa_clipped"] = np.minimum(joined["mono_response_a"], joined["mono_response_b"])
    joined["product_clipped"] = joined["mono_response_a"] * joined["mono_response_b"]
    joined["hsa_unclipped"] = np.minimum(
        joined["mono_response_a_raw_ratio"], joined["mono_response_b_raw_ratio"]
    )
    joined["product_unclipped"] = (
        joined["mono_response_a_raw_ratio"] * joined["mono_response_b_raw_ratio"]
    )

    hsa_clipped = surface_macro_rmse(joined, "response", "hsa_clipped")
    hsa_unclipped = surface_macro_rmse(joined, "target_unclipped", "hsa_unclipped")
    return {
        "surfaces": int(joined["surface_replicate_id"].nunique()),
        "points": int(len(joined)),
        "product_clipped": surface_macro_rmse(joined, "response", "product_clipped"),
        "hsa_clipped": hsa_clipped,
        "gate_clipped": hsa_clipped * (1.0 - GATE),
        "product_unclipped": surface_macro_rmse(joined, "target_unclipped", "product_unclipped"),
        "hsa_unclipped": hsa_unclipped,
        "gate_unclipped": hsa_unclipped * (1.0 - GATE),
        "fraction_clipped_high": float(joined["response_clipped_high"].mean()),
    }


def summarise(per_fold: list[dict[str, float]], key: str) -> dict[str, float]:
    values = np.array([fold[key] for fold in per_fold])
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.json is not None and args.json.exists():
        raise SystemExit("Refusing to overwrite an existing reference report")

    shards = load_learned_point_shards(
        load_benchmark_config(ROOT / "configs/benchmark_one_new_scaffold_unclipped.yaml"),
        verify_roles=("train", "validation"),
    )
    raw_by_fold = {fold: raw_lookup(fold, shards) for fold in range(5)}
    report: dict[str, object] = {
        "scope": "validation_scoring_with_same_fold_train_validation_D1_lookup",
        "response_roles_loaded": ["train", "validation"],
        "full_response_table_opened": False,
        "access_contract_version": "physical_development_lookup_v2",
        "test_shards_read": 0,
        "reason": (
            "The endpoint was selected on a development cohort; recomputing test "
            "references before D1 is frozen would make a test result part of the "
            "endpoint decision."
        ),
        "gate_definition": "relative improvement of 5% against the HSA reference",
    }

    for name, folder, filename in (
        ("one_new_scaffold", SCAFFOLD_SHARDS, "validation.parquet"),
        ("warm_pairout", WARM_SHARDS, "validation.parquet"),
    ):
        per_fold = []
        for fold in range(5):
            shard = folder / f"fold_{fold}" / filename
            result = evaluate(shard, raw_by_fold[fold])
            result["fold"] = fold
            per_fold.append(result)
            print(
                f"{name} fold {fold}: "
                f"HSA clipped {result['hsa_clipped']:.6f} -> unclipped {result['hsa_unclipped']:.6f} | "
                f"gate {result['gate_clipped']:.6f} -> {result['gate_unclipped']:.6f} | "
                f"clip rate {result['fraction_clipped_high']:.4f}"
            )
        report[name] = {
            "folds": per_fold,
            "hsa_clipped": summarise(per_fold, "hsa_clipped"),
            "hsa_unclipped": summarise(per_fold, "hsa_unclipped"),
            "product_clipped": summarise(per_fold, "product_clipped"),
            "product_unclipped": summarise(per_fold, "product_unclipped"),
            "gate_clipped": summarise(per_fold, "gate_clipped"),
            "gate_unclipped": summarise(per_fold, "gate_unclipped"),
        }
        print()

    report["input_records"] = [
        _file_record(shards.role_path(fold, role))
        for fold in range(5) for role in ("train", "validation")
    ]

    print(json.dumps({k: v for k, v in report.items() if isinstance(v, dict)}, indent=2)[:0])
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

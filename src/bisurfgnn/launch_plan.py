"""Frozen full-matrix inventory and strict receipt aggregation checks."""
from pathlib import Path

import pandas as pd

from .models.factorial_bisurf import VARIANTS as GRAPH
from .lightweight_controls import VARIANTS as RIDGE
from .learned_benchmark import _canonical_hash, _file_record

WIDTHS = dict(pooled__concat=140, pooled__gate=136, bixt__concat=140, bixt__gate=136,
              cross_attention__concat=136, cross_attention__gate=136)
MAIN = (*GRAPH, "morgan_mlp", "independent_pooled_gnn", *RIDGE)


def make_inventory(config):
    if config["folds"] != list(range(5)) or config["seeds"] != [41, 42, 43]:
        raise ValueError("Formal inventory is fixed at 5 folds x 3 seeds")
    jobs, aliases = [], []
    for fold in config["folds"]:
        for seed in config["seeds"]:
            for track, models in (("main", MAIN), ("capacity", GRAPH), ("isotime", GRAPH)):
                for model in models:
                    identity = dict(track=track, model=model, fold=fold, seed=seed)
                    identity["id"] = f"{track}--{model}--f{fold}--s{seed}"
                    identity["width"] = WIDTHS[model] if track == "capacity" else 136
                    if track == "capacity" and identity["width"] == 136:
                        aliases.append(dict(**identity, source=f"main--{model}--f{fold}--s{seed}"))
                    else:
                        jobs.append(identity)
    return dict(jobs=jobs, aliases=aliases)


def verify_receipts(inventory, receipts, plan_hash):
    expected = {job["id"]: job for job in inventory["jobs"]}
    if len(receipts) != len(expected) or len({r["job"]["id"] for r in receipts}) != len(receipts):
        raise ValueError("Incomplete or duplicate job receipt inventory")
    for receipt in receipts:
        data = {k: v for k, v in receipt.items() if k != "manifest_sha256"}
        if _canonical_hash(data) != receipt.get("manifest_sha256"):
            raise ValueError("Receipt self-hash mismatch")
        if receipt["job"]["id"] not in expected or receipt["job"] != expected[receipt["job"]["id"]]:
            raise ValueError("Job metadata differs from frozen matrix")
        if receipt["plan_hash"] != plan_hash or receipt["test_response_files_read"] != 0:
            raise ValueError("Plan/contact contract mismatch")
        if not receipt["finished"]:
            raise ValueError("Unfinished job cannot enter aggregation")
        for key in ("prediction", "errors", "checkpoint"):
            if _file_record(Path(receipt[key]["path"])) != receipt[key]:
                raise ValueError("Receipt artifact changed")
    return True


def validate_paired_points(frames):
    expected = None
    columns = ["source_id", "surface_id", "group_id", "cluster_id", "assay_block_id", "target"]
    for frame in frames:
        if frame.source_id.duplicated().any() or frame[columns].isna().any().any():
            raise ValueError("Duplicate/missing point metadata")
        actual = frame[columns].sort_values("source_id").reset_index(drop=True)
        if expected is None:
            expected = actual
        else:
            try:
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
            except AssertionError as exc:
                raise ValueError("Model/seed point inventories or targets differ") from exc

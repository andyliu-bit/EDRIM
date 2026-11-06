"""Shared helpers for CrossScreenMono-v1 (docs/CROSS_SCREEN_MONO_V1_PROTOCOL.md).

Frozen-sample loading is the AlgorithmSearch-v1 path (`diagnose_fit_vs_generalization.frozen_tables`, which
drives `run_comparator_pilot.sample_supported_frames` and the same feature banks); this module only adds
hash checks, the CellMiner cell mapping and the cross-screen (CS) table I/O.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs/model_improvement/cross_screen_mono_v1"
CS_DIR = RUN_ROOT / "cs_table"
NCI60_DIR = ROOT / "data/interim/nci60_pretrain_v1"
ID_COLUMNS = ["source_combodrugseq", "prediction_group_id", "surface_replicate_id"]
ROLES = ("train", "validation")
MONO_MAX = 3.0
FROZEN_ID_SHA256 = {"train": "5752b4675ef81a9ccd021545a8d1072b965125bbfca7d67684a8e019b5b5d81e",
                    "validation": "eb3b88ca488f038e3792ed5cd08d547023648137bbd22be39da65449d117ca5e"}
NCI60_INPUT_SHA256 = {"nci60_points.parquet": "a0f8052c66223d6e62660ed2e5fa4572fa58270b7b14854031c4b33cc0b27e10",
                      "nci60_compounds.parquet": "41983cd94f292ced71990b91ceee37c3ca9d4fa39c07f3f616ecb1e21a5a6929"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


fitgen = load_module("csm_fitgen", ROOT / "experiments/model_improvement/analysis/diagnose_fit_vs_generalization.py")
search, pilot, learned, graph = fitgen.search, fitgen.pilot, fitgen.learned, fitgen.graph
summ = load_module("csm_summary", ROOT / "experiments/model_improvement/analysis/summarize_algorithm_search.py")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n({time.strftime('%Y-%m-%dT%H:%M:%S%z')}) {text.rstrip()}\n")


def write_ids(frame: pd.DataFrame, path: Path) -> str:
    """Same writer as run_comparator_pilot.execute (ID columns, no index)."""
    frame[ID_COLUMNS].to_csv(path, index=False)
    return sha256_file(path)


def verify_reference_inputs(cfg: dict, banks) -> list[dict]:
    """Equivalent of run_algorithm_search.check_reference without its output-directory layout: every frozen
    input of the reference run is re-hashed and the training-fitted expression state must be unchanged."""
    reference = ROOT / cfg["reference_run"]
    binding = json.loads((reference / "data_binding.json").read_text())
    if banks.context_state_hash != binding["context_state_hash"]:
        raise RuntimeError("Training-fitted expression state drift")
    for record in binding["inputs"]:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"Frozen data or preprocessing changed: {record['path']}")
    for role in ROLES:
        if sha256_file(reference / f"{role}_ids.csv") != FROZEN_ID_SHA256[role]:
            raise RuntimeError(f"Reference ID file changed: {role}")
    return binding["inputs"]


def frozen_frames(cfg: dict, id_dir: Path):
    """Frozen 18000/9000 development frames, banks, same-plate tables and Morgan-512 bank.

    Writes ``{role}_ids.csv`` into ``id_dir`` and asserts the frozen SHA-256 (byte comparison)."""
    frames, banks, tables, fingerprints = fitgen.frozen_tables(cfg)
    id_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for role in ROLES:
        hashes[role] = write_ids(frames[role], id_dir / f"{role}_ids.csv")
        if hashes[role] != FROZEN_ID_SHA256[role]:
            raise RuntimeError(f"Point ID/order drift relative to the frozen sample: {role} {hashes[role]}")
    inputs = verify_reference_inputs(cfg, banks)
    return frames, banks, tables, fingerprints, dict(id_sha256=hashes, reference_inputs=inputs)


def cell_mapping(cfg: dict) -> dict[str, str]:
    """Frozen CellMiner mapping (ALMANAC source cell id -> biological cell id), as in NCI60-Pretrain-v1 S2."""
    from bisurfgnn.benchmark_config import load_benchmark_config
    from bisurfgnn.data.benchmark_context import load_benchmark_cell_context
    from bisurfgnn.data.cellminer import load_cellminer_config
    data_root = (ROOT / cfg["data_root"]).resolve()
    ctx = load_benchmark_cell_context(load_benchmark_config(data_root / cfg["benchmark_config"]),
                                      load_cellminer_config(data_root / cfg["cell_config"]))
    mapped = ctx.mapping[ctx.mapping.status.astype(str) == "mapped"]
    return dict(zip(mapped.source_cell_id, mapped.biological_cell_id))


def load_cs_points(role: str) -> pd.DataFrame:
    """CS values per point (step 1 output), hash-checked against the step-1 manifest."""
    manifest = json.loads((CS_DIR / "manifest.json").read_text(encoding="utf-8"))
    path = CS_DIR / f"cs_points_{role}.csv"
    if sha256_file(path) != manifest["outputs"][path.name]:
        raise RuntimeError(f"CS table changed after step 1: {path.name}")
    return pd.read_csv(path)


def covered_cs_frames(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Restrict to CS-covered points and REPLACE the same-plate monotherapy columns by CS values.

    The returned frames carry no same-plate monotherapy at all (the columns are overwritten). Values are
    clipped into [0, MONO_MAX] for the CompactPointTable contract; the number clipped is reported."""
    out, audit = {}, {}
    for role in ROLES:
        frame, cs = frames[role], load_cs_points(role)
        if not (cs.source_combodrugseq.to_numpy() == frame.source_combodrugseq.to_numpy()).all():
            raise RuntimeError(f"CS table order differs from the frozen frame: {role}")
        keep = cs.covered.to_numpy(bool)
        sub = frame.loc[keep].copy().reset_index(drop=True)
        clipped = 0
        for arm in ("a", "b"):
            values = cs.loc[keep, f"cs_mono_{arm}"].to_numpy(np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError("Covered point without a CS value")
            clipped += int(((values < 0) | (values > MONO_MAX)).sum())
            sub[f"mono_response_{arm}"] = np.clip(values, 0.0, MONO_MAX)
        ids = pd.read_csv(CS_DIR / f"covered_{role}_ids.csv")
        if not (ids.source_combodrugseq.to_numpy() == sub.source_combodrugseq.to_numpy()).all():
            raise RuntimeError(f"Covered subset differs from the step-1 ID file: {role}")
        out[role] = sub
        audit[role] = dict(points=len(sub), surfaces=int(sub.surface_replicate_id.nunique()),
                           values_clipped_into_0_3=clipped)
    return out, audit

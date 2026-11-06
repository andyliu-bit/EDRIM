"""Shared helpers for PublicSplitGate-v1 (docs/PUBLIC_SPLIT_GATE_V1_PROTOCOL.md; implementation details in
runs/model_improvement/public_split_gate_v1/FROZEN_RECORD.md).

Turns the S1 point tables into the frames, graph banks and compact point tables consumed by the frozen
AlgorithmSearch-v1 code paths, and builds BiSurfGNN exactly as `run_algorithm_search.build_model`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from bisurfgnn import independent_graph_benchmark as graph  # noqa: E402
from bisurfgnn import learned_benchmark as learned  # noqa: E402
from bisurfgnn.models.compact_bisurf import CompactBiSurfGNN  # noqa: E402
from bisurfgnn.models.compact_bisurf_additive import AdditiveCompactBiSurfGNN  # noqa: E402

DATA = ROOT.parent / "bisurfgnn"
RUN = ROOT / "runs/model_improvement/public_split_gate_v1"
S1 = RUN / "s1"
CARD = ROOT / "docs/PUBLIC_SPLIT_GATE_V1_PROTOCOL.md"
ROLE_OFFSET = {"train": 0, "validation": 10_000_000, "test": 20_000_000}
MODEL_KWARGS = dict(graph_layers=2, attention_heads=4, dropout=0.1, monotherapy_upper_bound=3.0,
                    residual_monotherapy_input=True)
FORMS = {"anchored": CompactBiSurfGNN, "additive": AdditiveCompactBiSurfGNN}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


search = load_module("gate_search", ROOT / "experiments/model_improvement/run_algorithm_search.py")
pilot = search.pilot


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def check_bindings() -> dict:
    """Card unchanged since its freeze record and protocol.json; S1 outputs unchanged since S1."""
    card = sha256(CARD)
    freeze = json.loads((RUN / "CARD_FREEZE.json").read_text(encoding="utf-8"))
    protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
    if not card == freeze["sha256"] == protocol["card_sha256"]:
        raise RuntimeError("Launch card changed after it was frozen")
    manifest = json.loads((S1 / "manifest.json").read_text(encoding="utf-8"))
    drift = [name for name, h in manifest.items() if sha256(S1 / name) != h]
    if drift:
        raise RuntimeError(f"S1 outputs changed: {drift}")
    return dict(card_sha256=card, s1_manifest_sha256=sha256(S1 / "manifest.json"))


def load_points(role: str) -> pd.DataFrame:
    if role not in ("train", "validation"):
        raise ValueError("Development code reads only the train and validation point tables")
    frame = pd.read_parquet(S1 / f"points_{role}.parquet")
    frame["source_combodrugseq"] = ROLE_OFFSET[role] + np.arange(len(frame), dtype=np.int64)
    frame["surface_replicate_id"] = frame.matrix_key
    frame["prediction_group_id"] = frame.matrix_key
    frame["heldout_cluster_id"] = frame.pair_id
    frame["response_clipped_high"] = False
    return frame


def build_banks(frames: dict[str, pd.DataFrame]):
    from bisurfgnn.benchmark_config import load_benchmark_config
    from bisurfgnn.data.benchmark_context import load_benchmark_cell_context
    from bisurfgnn.data.cellminer import load_cellminer_config
    benchmark = load_benchmark_config(DATA / "configs/benchmark_bisurf_unclipped.yaml")
    context = load_benchmark_cell_context(benchmark, load_cellminer_config(DATA / "configs/cell_context.yaml"))
    fit_cells = set(context.folds[0].state.fit_cell_ids)
    frame = pd.concat(frames.values())
    used_cells = set(frame.biological_cell_id)
    if not used_cells <= fit_cells or len(fit_cells) != 59:
        raise RuntimeError("Fold-0 cell context does not cover exactly the 59 mapped cells")
    banks = graph.build_graph_feature_banks(sorted(set(frame.drug_a_id) | set(frame.drug_b_id)),
                                            sorted(set(frame.cell_id)), context, 0, benchmark)
    return banks


def build_tables(frames: dict[str, pd.DataFrame], banks) -> dict:
    return {r: learned.build_compact_point_table(f, banks, role=r, fold=0, monotherapy_upper_bound=3.0)
            for r, f in frames.items()}


def morgan(banks) -> torch.Tensor:
    return pilot.morgan512(DATA, banks)


def build_model(banks, form: str, seed: int, device: str):
    """Identical to run_algorithm_search.build_model('bisurf_mono_v2'); the additive form shares the network."""
    learned._seed_everything(seed)
    model = FORMS[form](banks.atom_feature_dim, banks.cell_feature_dim, 136, **MODEL_KWARGS)
    learned._seed_everything(seed)
    reference = search.build_model(banks, "bisurf_mono_v2", seed, {"device": "cpu"})
    expected, actual = reference.state_dict(), model.state_dict()
    if expected.keys() != actual.keys() or not all(torch.equal(expected[k], actual[k]) for k in expected):
        raise RuntimeError(f"{form} model initialisation differs from the frozen v2 builder (seed {seed})")
    if repr(reference).split("(", 1)[1] != repr(model).split("(", 1)[1]:
        raise RuntimeError("Module structure differs from the frozen v2 builder")
    del reference
    learned._seed_everything(seed)
    return model.to(torch.device(device))


def pooled_rmse_percent(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(100.0 * np.sqrt(np.mean((np.asarray(prediction, float) - np.asarray(target, float)) ** 2)))


def surface_macro(table, prediction) -> float:
    return pilot.metric(table.target.numpy(), prediction, table.surface_ids)["surface_macro_rmse"]

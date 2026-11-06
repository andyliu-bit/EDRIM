"""DeepSynBa's precomputed drug and cell inputs, aligned to BiSurfGNN graph banks (BiSurfGNN v2, phase 1).

Two author-precomputed files of the DeepSynBa data release (github.com/hikuru/DeepSynBa, commit 43c16961), read
only from the main checkout:

* ``NCI-ALMANAC_drug_molformer_embeddings.pt``: dict drug name -> float32 tensor (1, 768), MoLFormer embedding;
* ``NCI-60_landmark_gex.csv``: column ``genes`` (976 landmark symbols) plus 60 NCI-60 cell columns, author values.

Mapping (docs/INPUT_PARITY_FEASIBILITY_20261013.md section 2.2-2.3):

* drug: bank canonical ID -> S1 ``drug_mapping_used.csv`` ``deepsynba_drug`` -> author ``normalize_drug_name``
  (drops one trailing space, e.g. ``Pemetrexed Disodium ``) -> embedding key;
* cell: bank source cell ID -> S1 ``cell_mapping_used.csv`` ``deepsynba_cell`` -> author ``convert_cell_name``
  (e.g. ``HL-60(TB)`` -> ``HL_60``) -> expression column. MDA-MB-468 is refused: the author rule rewrites it to
  ``MDA_N``, a different cell line, and the gate excludes it.

Preprocessing: MoLFormer is standardised per dimension with the mean and SD (ddof 0) of the training-role drugs
only; expression keeps the author values and NaN is set to 0, as DeepSynBa's forward does (the only NaNs are 82
genes of SF-539). Both mappings are checked to be one-to-one and against the point tables' own author-name columns.

The ``.pt`` container is read with ``torch.load(weights_only=True)`` and only in an isolated interpreter
(``python -I``). License of both files: CC BY-NC-SA 4.0 (DeepSynBa README); local use only, do not distribute the
files or tensors derived from them.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

EMBEDDING_FILE = "NCI-ALMANAC_drug_molformer_embeddings.pt"
EXPRESSION_FILE = "NCI-60_landmark_gex.csv"
EXPECTED_SHA256 = {
    EMBEDDING_FILE: "6fd9fde5e7abb5d849bef0944deab2494668b5ded2aef50e9ddb9454972af106",
    EXPRESSION_FILE: "d35dcc3c4ae7b40c3671327e0c53b7641dff48c8bc1bb57266a79065c20f687b",
}
DRUG_MAPPING_FILE = "drug_mapping_used.csv"
CELL_MAPPING_FILE = "cell_mapping_used.csv"
DRUG_DIM = 768
CELL_DIM = 976
EXCLUDED_CELL_NAMES = frozenset({"MDA-MB-468"})
EXCLUDED_EXPRESSION_COLUMNS = frozenset({"MDA_N"})
FRAME_COLUMNS = ("drug_a_id", "drug_b_id", "cell_id", "deepsynba_drug_a", "deepsynba_drug_b", "deepsynba_cell")
PROTECTED_PATH_TOKENS = ("public_split_gate_v1/s5", "test_evaluated", "test_predictions", "final_predictions",
                         "test.parquet", "points_test", "jaaks", "intensity", "az_sanger", "reserve")


class ExternalFeatureError(RuntimeError):
    """Raised when the external inputs cannot be aligned exactly and safely."""


# --- verbatim copies of the author's name normalisation (DeepSynBa dataset.py L66-71 and L76-80) -------------
def convert_cell_name(cell):
    cell = cell.replace('-', '_').replace('/ATCC', '')
    cell = cell.replace(' ', '').replace('(TB)', '').replace('MDA_MB_468', 'MDA_N').replace('NCI/', 'NCI_')
    cell = cell.replace('RXF393', 'RXF_393').replace('T_47D', 'T47D')
    return cell


def normalize_drug_name(drug_name):
    if drug_name[-1] == ' ':
        drug_name = drug_name[:-1]
    return drug_name


def _guard(path: Path) -> Path:
    text = Path(path).resolve().as_posix().lower()
    if any(token in text for token in PROTECTED_PATH_TOKENS):
        raise ExternalFeatureError(f"Refusing to open a protected path: {text}")
    return Path(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _guard(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(str(array.dtype).encode("ascii") + str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _clean(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


@dataclass(frozen=True)
class ExternalFeatureBanks:
    """External rows in bank order: ``drug[i]`` belongs to ``drug_ids[i]``, ``cell[j]`` to ``cell_ids[j]``."""

    drug_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    drug: Tensor
    cell: Tensor
    drug_keys: tuple[str, ...]
    cell_columns: tuple[str, ...]
    genes: tuple[str, ...]
    record: dict[str, Any]

    def __post_init__(self) -> None:
        if self.drug.shape != (len(self.drug_ids), DRUG_DIM) or self.drug.dtype != torch.float32:
            raise ExternalFeatureError("Drug feature table must be float32 [drugs, 768] in bank order")
        if self.cell.shape != (len(self.cell_ids), CELL_DIM) or self.cell.dtype != torch.float32:
            raise ExternalFeatureError("Cell feature table must be float32 [cells, 976] in bank order")
        if not bool(torch.isfinite(self.drug).all()) or not bool(torch.isfinite(self.cell).all()):
            raise ExternalFeatureError("External feature tables must be finite")
        if len(self.drug_keys) != len(self.drug_ids) or len(self.cell_columns) != len(self.cell_ids):
            raise ExternalFeatureError("Feature provenance is incomplete")
        if len(set(self.drug_keys)) != len(self.drug_keys) or len(set(self.cell_columns)) != len(self.cell_columns):
            raise ExternalFeatureError("External features must map one-to-one onto bank rows")


def _drug_mapping(path: Path) -> dict[str, str]:
    table = pd.read_csv(_guard(path), keep_default_na=False, dtype=str)
    table = table[table["mapped"].astype(str).eq("True")]
    ids = table["canonical_drug_id"].map(_clean)
    if ids.duplicated().any() or table["deepsynba_drug"].duplicated().any():
        raise ExternalFeatureError("S1 drug mapping is not one-to-one")
    return dict(zip(ids, table["deepsynba_drug"]))           # raw author names, trailing spaces kept


def _cell_mapping(path: Path) -> dict[str, str]:
    table = pd.read_csv(_guard(path), keep_default_na=False, dtype=str)
    table = table[table["status"].astype(str).eq("mapped")]
    ids = table["cell_id"].map(_clean)
    if ids.duplicated().any() or table["deepsynba_cell"].duplicated().any():
        raise ExternalFeatureError("S1 cell mapping is not one-to-one")
    mapping = dict(zip(ids, table["deepsynba_cell"]))
    if EXCLUDED_CELL_NAMES & set(mapping.values()):
        raise ExternalFeatureError("MDA-MB-468 must not be mapped (the author rule aliases it to MDA_N)")
    return mapping


def _check_frames(frames: Mapping[str, pd.DataFrame], drug_ids: Sequence[str], cell_ids: Sequence[str],
                  drug_names: Mapping[str, str], cell_names: Mapping[str, str]) -> dict[str, Any]:
    """Every frame ID is a bank row, and the point tables' own author names agree with the S1 mapping files."""
    if "train" not in frames or not set(frames) <= {"train", "validation"}:
        raise ExternalFeatureError("Feature alignment uses only the train (required) and validation frames")
    bank_drugs, bank_cells = set(drug_ids), set(cell_ids)
    coverage = {}
    for role, frame in frames.items():
        missing = sorted(set(FRAME_COLUMNS) - set(frame.columns))
        if missing:
            raise ExternalFeatureError(f"{role} frame lacks columns {missing}")
        pairs = pd.concat([
            frame[["drug_a_id", "deepsynba_drug_a"]].set_axis(["drug_id", "name"], axis=1),
            frame[["drug_b_id", "deepsynba_drug_b"]].set_axis(["drug_id", "name"], axis=1),
        ]).drop_duplicates()
        cells = frame[["cell_id", "deepsynba_cell"]].drop_duplicates().set_axis(["cell_id", "name"], axis=1)
        if EXCLUDED_CELL_NAMES & set(cells["name"]):
            raise ExternalFeatureError(f"{role} frame contains MDA-MB-468, which the gate excludes")
        for kind, table, key, names, bank in (("drug", pairs, "drug_id", drug_names, bank_drugs),
                                               ("cell", cells, "cell_id", cell_names, bank_cells)):
            ids = table[key].map(_clean)
            if not set(ids) <= bank:
                raise ExternalFeatureError(f"{role} {kind} IDs outside the bank: {sorted(set(ids) - bank)[:5]}")
            disagree = [(i, n) for i, n in zip(ids, table["name"]) if names.get(i) != n]
            if disagree:
                raise ExternalFeatureError(f"{role} {kind} names disagree with the S1 mapping: {disagree[:5]}")
            if ids.duplicated().any():
                raise ExternalFeatureError(f"{role} {kind} IDs carry more than one author name")
        # every bank row receives a feature row below (or loading fails), so these IDs are all covered
        coverage[role] = dict(points=int(len(frame)), drugs=int(pairs["drug_id"].nunique()),
                              cells=int(cells["cell_id"].nunique()))
    return coverage


def load_deepsynba_features(
    drug_ids: Sequence[str],
    cell_ids: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    *,
    feature_dir: Path,
    mapping_dir: Path,
) -> ExternalFeatureBanks:
    """Load and align both external inputs to the bank order; fail closed on any mismatch."""

    if not sys.flags.isolated:
        raise ExternalFeatureError("The MoLFormer file is a torch/pickle container: run with `python -I`")
    drug_ids = tuple(_clean(v) for v in drug_ids)
    cell_ids = tuple(_clean(v) for v in cell_ids)
    if len(set(drug_ids)) != len(drug_ids) or len(set(cell_ids)) != len(cell_ids) or not drug_ids or not cell_ids:
        raise ExternalFeatureError("Bank IDs must be non-empty and unique")
    feature_dir, mapping_dir = Path(feature_dir), Path(mapping_dir)
    files = {}
    for name in (EMBEDDING_FILE, EXPRESSION_FILE):
        path = feature_dir / name
        digest = _sha256_file(path)
        if digest != EXPECTED_SHA256[name]:
            raise ExternalFeatureError(f"{name} differs from the recorded author file ({digest})")
        files[name] = dict(path=str(path.resolve()), bytes=int(path.stat().st_size), sha256=digest)
    for name in (DRUG_MAPPING_FILE, CELL_MAPPING_FILE):
        path = mapping_dir / name
        files[name] = dict(path=str(path.resolve()), bytes=int(path.stat().st_size), sha256=_sha256_file(path))

    drug_names = _drug_mapping(mapping_dir / DRUG_MAPPING_FILE)
    cell_names = _cell_mapping(mapping_dir / CELL_MAPPING_FILE)
    coverage = _check_frames(frames, drug_ids, cell_ids, drug_names, cell_names)

    # drugs: bank ID -> author name -> embedding key
    unmapped = [d for d in drug_ids if d not in drug_names]
    if unmapped:
        raise ExternalFeatureError(f"Bank drugs without an S1 author name: {unmapped[:5]}")
    embeddings = torch.load(_guard(feature_dir / EMBEDDING_FILE), map_location="cpu", weights_only=True)
    if not isinstance(embeddings, dict) or not all(isinstance(k, str) for k in embeddings):
        raise ExternalFeatureError("Unexpected MoLFormer container")
    drug_keys = tuple(normalize_drug_name(drug_names[d]) for d in drug_ids)
    absent = [k for k in drug_keys if k not in embeddings]
    if absent or len(set(drug_keys)) != len(drug_keys):
        raise ExternalFeatureError(f"Embedding keys missing or not one-to-one: {absent[:5]}")
    raw_rows = []
    for key in drug_keys:
        value = embeddings[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != (1, DRUG_DIM) or value.dtype != torch.float32:
            raise ExternalFeatureError(f"Embedding {key!r} is not a float32 (1, 768) tensor")
        raw_rows.append(value.detach().reshape(-1).numpy().astype(np.float64))
    raw = np.stack(raw_rows)
    if not np.isfinite(raw).all():
        raise ExternalFeatureError("MoLFormer embeddings contain non-finite values")
    train = frames["train"]
    train_drugs = {_clean(v) for v in pd.concat([train["drug_a_id"], train["drug_b_id"]]).unique()}
    fit_rows = [i for i, d in enumerate(drug_ids) if d in train_drugs]
    mean = raw[fit_rows].mean(axis=0)
    sd = raw[fit_rows].std(axis=0, ddof=0)
    if not (sd > 0).all():
        raise ExternalFeatureError("A MoLFormer dimension has zero SD over the training drugs")
    drug = torch.as_tensor(((raw - mean) / sd).astype(np.float32))

    # cells: bank source ID -> author name -> expression column
    unmapped = [c for c in cell_ids if c not in cell_names]
    if unmapped:
        raise ExternalFeatureError(f"Bank cells without an S1 author name: {unmapped[:5]}")
    expression = pd.read_csv(_guard(feature_dir / EXPRESSION_FILE))
    if expression.columns[0] != "genes" or expression.shape != (CELL_DIM, 61):
        raise ExternalFeatureError(f"Unexpected expression layout {expression.shape}")
    genes = tuple(expression["genes"].astype(str))
    if len(set(genes)) != CELL_DIM:
        raise ExternalFeatureError("Expression gene symbols are not unique")
    cell_columns = tuple(convert_cell_name(cell_names[c]) for c in cell_ids)
    absent = [c for c in cell_columns if c not in expression.columns[1:]]
    if absent or len(set(cell_columns)) != len(cell_columns) or EXCLUDED_EXPRESSION_COLUMNS & set(cell_columns):
        raise ExternalFeatureError(f"Expression columns missing, aliased or not one-to-one: {absent[:5]}")
    values = expression.loc[:, list(cell_columns)].to_numpy(np.float64).T          # [cells, genes]
    nan = np.isnan(values)
    if np.isinf(values).any():
        raise ExternalFeatureError("Expression contains infinite values")
    nan_by_column = {cell_columns[i]: int(n) for i, n in enumerate(nan.sum(axis=1)) if n}
    cell = torch.as_tensor(np.where(nan, 0.0, values).astype(np.float32))

    renamed = {cell_names[c]: col for c, col in zip(cell_ids, cell_columns) if cell_names[c] != col}
    record = dict(
        source="DeepSynBa data release (commit 43c16961), author-precomputed; CC BY-NC-SA 4.0, local use only",
        files=files,
        drugs=len(drug_ids), cells=len(cell_ids), coverage=coverage,
        drug_keys=dict(zip(drug_ids, drug_keys)), cell_columns=dict(zip(cell_ids, cell_columns)),
        drug_names_with_trailing_space=sorted(drug_names[d] for d in drug_ids if drug_names[d] != drug_names[d].rstrip()),
        cell_names_rewritten_by_author_rule=len(renamed),
        unused_embedding_keys=sorted(set(embeddings) - set(drug_keys)),
        unused_expression_columns=sorted(set(expression.columns[1:]) - set(cell_columns)),
        molformer_standardisation=dict(fit_role="train", fit_drugs=len(fit_rows), ddof=0,
                                       sd_min=float(sd.min()), sd_max=float(sd.max()),
                                       mean_sha256=_sha256_array(mean), sd_sha256=_sha256_array(sd)),
        expression_nan_set_to_zero=nan_by_column,
        drug_table_sha256=_sha256_array(drug.numpy()), cell_table_sha256=_sha256_array(cell.numpy()),
    )
    return ExternalFeatureBanks(drug_ids=drug_ids, cell_ids=cell_ids, drug=drug, cell=cell, drug_keys=drug_keys,
                                cell_columns=cell_columns, genes=genes, record=record)


__all__ = [
    "CELL_DIM",
    "DRUG_DIM",
    "EXPECTED_SHA256",
    "ExternalFeatureBanks",
    "ExternalFeatureError",
    "convert_cell_name",
    "load_deepsynba_features",
    "normalize_drug_name",
]

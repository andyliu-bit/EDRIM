from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from ..benchmark_config import BenchmarkConfig


CHEMICAL_SPLIT_VERSION = "one-new-scaffold-split-v1"
CLUSTER_ID_PREFIX = "cc_"
VALID_ROLES = frozenset({"train", "validation", "test", "excluded"})


class ChemicalSplitError(ValueError):
    """Raised when a chemical split or its frozen artifacts violate the contract."""


@dataclass(frozen=True)
class ChemicalClusterResult:
    assignments: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class PreparedChemicalSplit:
    manifest: dict[str, Any]
    clusters: pd.DataFrame
    groups: pd.DataFrame
    surfaces: pd.DataFrame

    @property
    def split_hash(self) -> str:
        return str(self.manifest["split_hash"])


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank_left = self.rank[root_left]
        rank_right = self.rank[root_right]
        if rank_left < rank_right:
            root_left, root_right = root_right, root_left
            rank_left, rank_right = rank_right, rank_left
        self.parent[root_right] = root_left
        if rank_left == rank_right:
            self.rank[root_left] += 1


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ChemicalSplitError(f"{label} is missing required columns: {missing}")


def _stable_hex(*parts: object) -> str:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        recorded_path = (
            str(resolved.relative_to(relative_to.resolve())) if relative_to else str(resolved)
        )
    except ValueError:
        recorded_path = str(resolved)
    return {
        "path": recorded_path,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False, lineterminator="\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary_name, index=False)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _normalize_drugs(drugs: pd.DataFrame) -> pd.DataFrame:
    required = {
        "drug_id",
        "canonical_smiles",
        "identity_group",
        "murcko_scaffold",
        "structure_key",
    }
    _require_columns(drugs, required, "drug table")
    normalized = drugs.loc[:, sorted(required)].copy()
    for column in required:
        normalized[column] = normalized[column].map(_clean_text)
    if normalized["drug_id"].duplicated().any():
        duplicates = sorted(normalized.loc[normalized["drug_id"].duplicated(), "drug_id"])
        raise ChemicalSplitError(f"drug_id must be unique; duplicates: {duplicates[:10]}")
    for column in ("drug_id", "canonical_smiles", "identity_group", "structure_key"):
        empty = normalized.loc[normalized[column] == "", "drug_id"].tolist()
        if empty:
            raise ChemicalSplitError(f"{column} must be non-empty for every drug: {empty[:10]}")
    normalized = normalized.sort_values("drug_id", kind="mergesort").reset_index(drop=True)
    return normalized


def build_chemical_clusters(
    drugs: pd.DataFrame,
    *,
    similarity_threshold: float = 0.70,
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2_048,
) -> ChemicalClusterResult:
    """Build leakage groups using identity, non-empty Murcko, and Morgan similarity.

    The Morgan fingerprint is computed on the complete canonical molecule. Exact
    identity groups, exact non-empty Murcko scaffolds, and pairwise similarities
    greater than or equal to the threshold are unioned before connected components
    are assigned. Empty Murcko values intentionally contribute no exact-match edge.
    """

    if not 0 < similarity_threshold < 1:
        raise ChemicalSplitError("similarity_threshold must be in (0, 1)")
    if fingerprint_radius != 2 or fingerprint_bits != 2_048:
        raise ChemicalSplitError(
            "OneNewScaffoldAnchored-v1 requires whole-molecule Morgan radius=2/2048"
        )
    normalized = _normalize_drugs(drugs)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=fingerprint_radius, fpSize=fingerprint_bits
    )
    fingerprints: dict[str, Any] = {}
    fingerprint_hashes: dict[str, str] = {}
    for row in normalized.itertuples(index=False):
        molecule = Chem.MolFromSmiles(row.canonical_smiles)
        if molecule is None:
            raise ChemicalSplitError(f"Invalid canonical SMILES for {row.drug_id}")
        fingerprint = generator.GetFingerprint(molecule)
        fingerprints[row.drug_id] = fingerprint
        fingerprint_hashes[row.drug_id] = hashlib.sha256(
            DataStructs.BitVectToBinaryText(fingerprint)
        ).hexdigest()

    drug_ids = normalized["drug_id"].tolist()
    disjoint = _DisjointSet(drug_ids)
    identity_edges = 0
    murcko_edges = 0
    similarity_edges = 0

    for _, members in normalized.groupby("identity_group", sort=True):
        ids = members["drug_id"].tolist()
        for other in ids[1:]:
            disjoint.union(ids[0], other)
            identity_edges += 1
    nonempty_murcko = normalized[normalized["murcko_scaffold"] != ""]
    for _, members in nonempty_murcko.groupby("murcko_scaffold", sort=True):
        ids = members["drug_id"].tolist()
        for other in ids[1:]:
            disjoint.union(ids[0], other)
            murcko_edges += 1
    for right_index, right_id in enumerate(drug_ids):
        left_ids = drug_ids[:right_index]
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[right_id], [fingerprints[left_id] for left_id in left_ids]
        )
        for left_id, similarity in zip(left_ids, similarities, strict=True):
            if float(similarity) + 1e-12 >= similarity_threshold:
                disjoint.union(left_id, right_id)
                similarity_edges += 1

    members_by_root: dict[str, list[str]] = {}
    for drug_id in drug_ids:
        members_by_root.setdefault(disjoint.find(drug_id), []).append(drug_id)
    cluster_by_drug: dict[str, str] = {}
    for members in members_by_root.values():
        stable_members = sorted(members)
        cluster_id = CLUSTER_ID_PREFIX + _stable_hex(
            CHEMICAL_SPLIT_VERSION, *stable_members
        )[:20]
        for drug_id in stable_members:
            cluster_by_drug[drug_id] = cluster_id

    assignments = normalized.copy()
    assignments["cluster_id"] = assignments["drug_id"].map(cluster_by_drug)
    assignments["fingerprint_sha256"] = assignments["drug_id"].map(fingerprint_hashes)
    columns = [
        "drug_id",
        "identity_group",
        "structure_key",
        "murcko_scaffold",
        "canonical_smiles",
        "fingerprint_sha256",
        "cluster_id",
    ]
    assignments = assignments.loc[:, columns].sort_values("drug_id").reset_index(drop=True)

    max_cross_cluster_similarity = 0.0
    for right_index, right_id in enumerate(drug_ids):
        for left_id in drug_ids[:right_index]:
            if cluster_by_drug[left_id] == cluster_by_drug[right_id]:
                continue
            similarity = float(
                DataStructs.TanimotoSimilarity(fingerprints[left_id], fingerprints[right_id])
            )
            max_cross_cluster_similarity = max(max_cross_cluster_similarity, similarity)
    identity_cluster_counts = assignments.groupby("identity_group")["cluster_id"].nunique()
    murcko_cluster_counts = assignments.loc[
        assignments["murcko_scaffold"] != ""
    ].groupby("murcko_scaffold")["cluster_id"].nunique()
    checks = {
        "drug_ids_are_unique": not assignments["drug_id"].duplicated().any(),
        "identity_groups_do_not_cross_clusters": bool((identity_cluster_counts == 1).all()),
        "nonempty_murcko_scaffolds_do_not_cross_clusters": bool(
            (murcko_cluster_counts == 1).all()
        ),
        "cross_cluster_similarity_is_below_threshold": (
            max_cross_cluster_similarity + 1e-12 < similarity_threshold
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ChemicalSplitError(f"Chemical clustering audit failed: {failed}")
    audit = {
        "status": "passed",
        "checks": checks,
        "drug_count": len(assignments),
        "identity_count": int(assignments["identity_group"].nunique()),
        "nonempty_murcko_count": int(
            assignments.loc[assignments["murcko_scaffold"] != "", "murcko_scaffold"].nunique()
        ),
        "empty_murcko_drug_count": int((assignments["murcko_scaffold"] == "").sum()),
        "cluster_count": int(assignments["cluster_id"].nunique()),
        "identity_union_edges": identity_edges,
        "murcko_union_edges": murcko_edges,
        "similarity_union_edges": similarity_edges,
        "max_cross_cluster_tanimoto": max_cross_cluster_similarity,
        "fingerprint": {
            "kind": "Morgan",
            "molecule_scope": "whole_canonical_molecule",
            "radius": fingerprint_radius,
            "bits": fingerprint_bits,
            "similarity": "Tanimoto",
            "threshold_inclusive": similarity_threshold,
        },
        "rdkit_version": rdBase.rdkitVersion,
    }
    return ChemicalClusterResult(assignments=assignments, audit=audit)


def designated_endpoint(
    prediction_group_id: str,
    drug_a_id: str,
    drug_b_id: str,
    *,
    protocol: str = "OneNewScaffoldAnchored-v1",
) -> tuple[str, str]:
    """Return a stable (heldout, partner) orientation independent of A/B order."""

    group_id = _clean_text(prediction_group_id)
    left = _clean_text(drug_a_id)
    right = _clean_text(drug_b_id)
    if not group_id or not left or not right:
        raise ChemicalSplitError("prediction_group_id and both drug IDs must be non-empty")
    if left == right:
        raise ChemicalSplitError(f"Self-pair is not valid for {group_id}")
    canonical = sorted((left, right))
    bit = int(_stable_hex(protocol, group_id, canonical[0], canonical[1]), 16) & 1
    return canonical[bit], canonical[1 - bit]


def _normalize_cluster_assignments(clusters: pd.DataFrame) -> pd.DataFrame:
    required = {"drug_id", "cluster_id", "identity_group", "structure_key"}
    _require_columns(clusters, required, "cluster assignments")
    normalized = clusters.copy()
    for column in required:
        normalized[column] = normalized[column].map(_clean_text)
    if normalized["drug_id"].duplicated().any():
        raise ChemicalSplitError("cluster assignments contain duplicate drug IDs")
    if (normalized[list(required)] == "").any(axis=None):
        raise ChemicalSplitError("cluster assignment key fields must be non-empty")
    if (normalized.groupby("identity_group")["cluster_id"].nunique() != 1).any():
        raise ChemicalSplitError("An identity_group crosses chemical clusters")
    if (normalized.groupby("structure_key")["cluster_id"].nunique() != 1).any():
        raise ChemicalSplitError("A structure_key crosses chemical clusters")
    return normalized.sort_values("drug_id", kind="mergesort").reset_index(drop=True)


def _normalize_prediction_groups(groups: pd.DataFrame) -> pd.DataFrame:
    required = {"prediction_group_id", "drug_a_id", "drug_b_id", "cell_id"}
    _require_columns(groups, required, "prediction groups")
    normalized = groups.copy()
    for column in required:
        normalized[column] = normalized[column].map(_clean_text)
    if (normalized[list(required)] == "").any(axis=None):
        raise ChemicalSplitError("Prediction-group key fields must be non-empty")
    if normalized["prediction_group_id"].duplicated().any():
        raise ChemicalSplitError("prediction_group_id must be unique in the group ledger")
    if (normalized["drug_a_id"] == normalized["drug_b_id"]).any():
        bad = normalized.loc[
            normalized["drug_a_id"] == normalized["drug_b_id"], "prediction_group_id"
        ].tolist()
        raise ChemicalSplitError(f"Self-pairs are not valid prediction groups: {bad[:10]}")
    return normalized.sort_values("prediction_group_id", kind="mergesort").reset_index(drop=True)


def _assign_cluster_bins(
    oriented_groups: pd.DataFrame,
    cluster_ids: Sequence[str],
    *,
    folds: int,
    split_seed: int,
) -> dict[str, int]:
    if folds != 5:
        raise ChemicalSplitError("OneNewScaffoldAnchored-v1 is frozen to five folds")
    if len(set(cluster_ids)) < folds:
        raise ChemicalSplitError("At least five chemical clusters are required")
    weights = oriented_groups["heldout_cluster_id"].value_counts().to_dict()
    ordered = sorted(
        set(cluster_ids),
        key=lambda cluster_id: (
            -int(weights.get(cluster_id, 0)),
            _stable_hex("lpt-order", split_seed, cluster_id),
            cluster_id,
        ),
    )
    loads = [0] * folds
    cluster_counts = [0] * folds
    assignment: dict[str, int] = {}
    for cluster_id in ordered:
        selected = min(
            range(folds),
            key=lambda fold: (
                loads[fold],
                cluster_counts[fold],
                _stable_hex("lpt-bin", split_seed, cluster_id, fold),
                fold,
            ),
        )
        assignment[cluster_id] = selected
        loads[selected] += int(weights.get(cluster_id, 0))
        cluster_counts[selected] += 1
    return assignment


def assign_one_new_scaffold_folds(
    groups: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    *,
    folds: int = 5,
    validation_offset: int = 1,
    split_seed: int = 41,
    protocol: str = "OneNewScaffoldAnchored-v1",
) -> pd.DataFrame:
    """Orient unique groups and assign strict train/validation/test roles."""

    if protocol != "OneNewScaffoldAnchored-v1":
        raise ChemicalSplitError("Unsupported chemical split protocol")
    if folds != 5 or not 0 < validation_offset < folds:
        raise ChemicalSplitError("The protocol requires five folds and another validation bin")
    normalized_groups = _normalize_prediction_groups(groups)
    clusters = _normalize_cluster_assignments(cluster_assignments)
    cluster_by_drug = clusters.set_index("drug_id").to_dict("index")
    unknown = sorted(
        (set(normalized_groups["drug_a_id"]) | set(normalized_groups["drug_b_id"]))
        - set(cluster_by_drug)
    )
    if unknown:
        raise ChemicalSplitError(f"Prediction groups reference unclustered drugs: {unknown[:10]}")

    oriented_records: list[dict[str, Any]] = []
    for record in normalized_groups.to_dict("records"):
        heldout, partner = designated_endpoint(
            record["prediction_group_id"],
            record["drug_a_id"],
            record["drug_b_id"],
            protocol=protocol,
        )
        heldout_metadata = cluster_by_drug[heldout]
        partner_metadata = cluster_by_drug[partner]
        record.update(
            {
                "heldout_drug_id": heldout,
                "partner_drug_id": partner,
                "heldout_cluster_id": heldout_metadata["cluster_id"],
                "partner_cluster_id": partner_metadata["cluster_id"],
                "heldout_identity_group": heldout_metadata["identity_group"],
                "partner_identity_group": partner_metadata["identity_group"],
                "heldout_structure_key": heldout_metadata["structure_key"],
                "partner_structure_key": partner_metadata["structure_key"],
            }
        )
        oriented_records.append(record)
    oriented = pd.DataFrame(oriented_records)
    distinct = oriented[
        oriented["heldout_cluster_id"] != oriented["partner_cluster_id"]
    ]
    cluster_bins = _assign_cluster_bins(
        distinct,
        clusters["cluster_id"].unique().tolist(),
        folds=folds,
        split_seed=split_seed,
    )
    oriented["heldout_cluster_bin"] = oriented["heldout_cluster_id"].map(cluster_bins)
    oriented["partner_cluster_bin"] = oriented["partner_cluster_id"].map(cluster_bins)
    # ``designated`` is the public protocol term; ``heldout`` is retained as an
    # explicit leakage-audit synonym for downstream readers.
    for suffix in ("drug_id", "cluster_id", "identity_group", "structure_key"):
        oriented[f"designated_{suffix}"] = oriented[f"heldout_{suffix}"]
    oriented["designated_cluster_bin"] = oriented["heldout_cluster_bin"]

    for fold in range(folds):
        test_bin = fold
        validation_bin = (fold + validation_offset) % folds
        reserved = {test_bin, validation_bin}
        preliminary_train = (
            ~oriented["heldout_cluster_bin"].isin(reserved)
            & ~oriented["partner_cluster_bin"].isin(reserved)
        )
        train_groups = oriented[preliminary_train]
        train_drugs = set(train_groups["drug_a_id"]) | set(train_groups["drug_b_id"])
        train_identities = {
            str(cluster_by_drug[drug_id]["identity_group"]) for drug_id in train_drugs
        }
        train_cells = set(train_groups["cell_id"])
        train_drug_cell_pairs = {
            (row.drug_a_id, row.cell_id)
            for row in train_groups.itertuples(index=False)
        } | {
            (row.drug_b_id, row.cell_id)
            for row in train_groups.itertuples(index=False)
        }

        validation_candidate = (
            (oriented["heldout_cluster_bin"] == validation_bin)
            & ~oriented["partner_cluster_bin"].isin(reserved)
        )
        test_candidate = (
            (oriented["heldout_cluster_bin"] == test_bin)
            & ~oriented["partner_cluster_bin"].isin(reserved)
        )
        role: list[str] = []
        reason: list[str] = []
        for row_index, row in oriented.iterrows():
            if bool(preliminary_train.loc[row_index]):
                role.append("train")
                reason.append("")
                continue
            candidate_role = (
                "validation"
                if bool(validation_candidate.loc[row_index])
                else "test"
                if bool(test_candidate.loc[row_index])
                else ""
            )
            if not candidate_role:
                role.append("excluded")
                reason.append("cluster_configuration_not_one_new")
                continue
            failures: list[str] = []
            if row["partner_drug_id"] not in train_drugs:
                failures.append("partner_drug_not_seen_in_train")
            if row["partner_identity_group"] not in train_identities:
                failures.append("partner_identity_not_seen_in_train")
            if row["cell_id"] not in train_cells:
                failures.append("cell_not_seen_in_train")
            if (row["partner_drug_id"], row["cell_id"]) not in train_drug_cell_pairs:
                failures.append("partner_drug_cell_not_seen_in_train")
            if failures:
                role.append("excluded")
                reason.append(";".join(failures))
            else:
                role.append(candidate_role)
                reason.append("")
        oriented[f"role_{fold}"] = role
        oriented[f"exclusion_reason_{fold}"] = reason
        oriented[f"role_fold_{fold}"] = role
        oriented[f"exclusion_reason_fold_{fold}"] = reason

    oriented = oriented.sort_values("prediction_group_id", kind="mergesort").reset_index(
        drop=True
    )
    audit_split_assignments(oriented, clusters, folds=folds, require_nonempty=True)
    return oriented


def audit_split_assignments(
    groups: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    folds: int = 5,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    normalized_clusters = _normalize_cluster_assignments(clusters)
    drug_meta = normalized_clusters.set_index("drug_id").to_dict("index")
    test_sets: list[set[str]] = []
    fold_reports: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    for fold in range(folds):
        role_column = f"role_{fold}"
        _require_columns(groups, {role_column}, "group split ledger")
        roles = set(groups[role_column].map(_clean_text))
        checks[f"fold_{fold}_roles_are_valid"] = roles <= VALID_ROLES
        partition = {
            role: groups[groups[role_column] == role] for role in ("train", "validation", "test")
        }
        if require_nonempty:
            checks[f"fold_{fold}_train_is_nonempty"] = not partition["train"].empty
            checks[f"fold_{fold}_validation_is_nonempty"] = not partition["validation"].empty
            checks[f"fold_{fold}_test_is_nonempty"] = not partition["test"].empty
        train = partition["train"]
        train_drugs = set(train["drug_a_id"]) | set(train["drug_b_id"])
        train_clusters = {drug_meta[drug_id]["cluster_id"] for drug_id in train_drugs}
        train_identities = {drug_meta[drug_id]["identity_group"] for drug_id in train_drugs}
        train_structures = {drug_meta[drug_id]["structure_key"] for drug_id in train_drugs}
        train_cells = set(train["cell_id"])
        train_drug_cell_pairs = {
            (row.drug_a_id, row.cell_id) for row in train.itertuples(index=False)
        } | {
            (row.drug_b_id, row.cell_id) for row in train.itertuples(index=False)
        }
        validation_drugs = set(partition["validation"]["drug_a_id"]) | set(
            partition["validation"]["drug_b_id"]
        )
        validation_clusters = {
            drug_meta[drug_id]["cluster_id"] for drug_id in validation_drugs
        }
        test = partition["test"]
        test_sets.append(set(test["prediction_group_id"]))
        for role in ("validation", "test"):
            selected = partition[role]
            checks[f"fold_{fold}_{role}_partners_seen_as_exact_drugs"] = set(
                selected["partner_drug_id"]
            ) <= train_drugs
            checks[f"fold_{fold}_{role}_partner_identities_seen"] = set(
                selected["partner_identity_group"]
            ) <= train_identities
            checks[f"fold_{fold}_{role}_cells_seen"] = set(selected["cell_id"]) <= train_cells
            checks[f"fold_{fold}_{role}_partner_drug_cell_pairs_seen"] = all(
                (row.partner_drug_id, row.cell_id) in train_drug_cell_pairs
                for row in selected.itertuples(index=False)
            )
            checks[f"fold_{fold}_{role}_heldout_clusters_absent_train"] = set(
                selected["heldout_cluster_id"]
            ).isdisjoint(train_clusters)
            checks[f"fold_{fold}_{role}_heldout_identities_absent_train"] = set(
                selected["heldout_identity_group"]
            ).isdisjoint(train_identities)
            checks[f"fold_{fold}_{role}_heldout_structures_absent_train"] = set(
                selected["heldout_structure_key"]
            ).isdisjoint(train_structures)
        checks[f"fold_{fold}_test_heldout_clusters_absent_validation"] = set(
            test["heldout_cluster_id"]
        ).isdisjoint(validation_clusters)
        fold_reports.append(
            {
                "fold": fold,
                "train_groups": len(train),
                "validation_groups": len(partition["validation"]),
                "test_groups": len(test),
                "excluded_groups": int((groups[role_column] == "excluded").sum()),
                "test_heldout_clusters": int(test["heldout_cluster_id"].nunique()),
            }
        )
    pairwise_disjoint = all(
        test_sets[left].isdisjoint(test_sets[right])
        for left in range(folds)
        for right in range(left + 1, folds)
    )
    checks["test_prediction_groups_are_pairwise_disjoint"] = pairwise_disjoint
    checks["each_group_is_test_at_most_once"] = bool(
        pd.Series(
            [group_id for test_set in test_sets for group_id in test_set], dtype="object"
        ).value_counts().le(1).all()
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "status": "passed" if not failed else "failed",
        "checks": checks,
        "folds": fold_reports,
        "group_count": len(groups),
        "test_group_union": len(set().union(*test_sets)),
    }
    if failed:
        raise ChemicalSplitError(f"One-new-scaffold split audit failed: {failed}")
    return report


def _read_drug_structure_mapping(path: Path) -> dict[str, str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ChemicalSplitError(f"Cannot read drug structure mapping: {path}") from error
    if not isinstance(payload, dict):
        raise ChemicalSplitError("drug_id_to_structure_key.json must contain an object")
    return {_clean_text(key): _clean_text(value) for key, value in payload.items()}


def _source_paths(config: BenchmarkConfig) -> dict[str, Path]:
    return {
        "drugs": config.nci_processed_dir / "drugs.csv",
        "surface_index": config.nci_processed_dir / "surface_index.parquet",
        "points": config.nci_processed_dir / "points.parquet",
        "cell_mapping": config.cell_processed_dir / "cell_mapping.csv",
        "drug_structure_mapping": config.nci_processed_dir
        / "drug_id_to_structure_key.json",
        "structure_manifest": config.nci_processed_dir / "structure_manifest.json",
        "prepare_config": config.nci_processed_dir / "prepare_config.json",
    }


def _verify_required_files(paths: Mapping[str, Path]) -> None:
    missing = {name: str(path) for name, path in paths.items() if not path.is_file()}
    if missing:
        raise ChemicalSplitError(f"Required prepared inputs are missing: {missing}")


def _eligible_surface_cohort(
    surfaces: pd.DataFrame,
    clusters: pd.DataFrame,
    cell_mapping: pd.DataFrame,
    structure_mapping: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    required_surface = {
        "surface_replicate_id",
        "prediction_group_id",
        "drug_a_id",
        "drug_b_id",
        "cell_id",
        "point_count",
        "mono_complete_points",
        "mono_complete_fraction",
        "primary_points",
    }
    _require_columns(surfaces, required_surface, "surface index")
    _require_columns(cell_mapping, {"source_cell_id", "status"}, "cell mapping")
    cohort = surfaces.copy()
    for column in (
        "surface_replicate_id",
        "prediction_group_id",
        "drug_a_id",
        "drug_b_id",
        "cell_id",
    ):
        cohort[column] = cohort[column].map(_clean_text)
    empty_keys = cohort.loc[
        (cohort[[
            "surface_replicate_id",
            "prediction_group_id",
            "drug_a_id",
            "drug_b_id",
            "cell_id",
        ]] == "").any(axis=1),
        "surface_replicate_id",
    ].tolist()
    if empty_keys:
        raise ChemicalSplitError(f"Surface index contains empty key fields: {empty_keys[:10]}")
    if cohort["surface_replicate_id"].duplicated().any():
        raise ChemicalSplitError("surface_replicate_id must be unique")
    if (cohort[list(required_surface & set(cohort.columns))].isna()).any(axis=None):
        raise ChemicalSplitError("Surface index contains null values in required fields")
    cluster_meta = clusters.set_index("drug_id")
    expected_structure = cluster_meta["structure_key"].to_dict()
    covered_drugs = {
        drug_id
        for drug_id, structure_key in structure_mapping.items()
        if drug_id in expected_structure and structure_key == expected_structure[drug_id]
    }
    mapped_cells = set(
        cell_mapping.loc[
            cell_mapping["status"].map(_clean_text) == "mapped", "source_cell_id"
        ].map(_clean_text)
    )
    stages: list[tuple[str, pd.Series]] = [
        (
            "structure_covered",
            cohort["drug_a_id"].isin(covered_drugs)
            & cohort["drug_b_id"].isin(covered_drugs),
        ),
        ("mapped_cell", cohort["cell_id"].isin(mapped_cells)),
        ("has_primary_points", pd.to_numeric(cohort["primary_points"], errors="coerce") > 0),
        (
            "complete_observed_monotherapy",
            pd.to_numeric(cohort["mono_complete_fraction"], errors="coerce").eq(1.0)
            & pd.to_numeric(cohort["mono_complete_points"], errors="coerce").eq(
                pd.to_numeric(cohort["point_count"], errors="coerce")
            ),
        ),
    ]
    counts = {"raw_surface_replicates": len(cohort)}
    selected = pd.Series(True, index=cohort.index)
    for stage, condition in stages:
        selected &= condition.fillna(False)
        counts[stage] = int(selected.sum())
    counts["surface_candidates"] = int(selected.sum())
    group_all_repeats_eligible = selected.groupby(cohort["prediction_group_id"]).all()
    eligible_group_ids = set(group_all_repeats_eligible[group_all_repeats_eligible].index)
    eligible = cohort[cohort["prediction_group_id"].isin(eligible_group_ids)].copy()
    counts["all_repeats_eligible_prediction_groups"] = len(eligible_group_ids)
    counts["all_repeats_eligible_surface_replicates"] = len(eligible)
    counts["eligible_surface_replicates"] = len(eligible)
    counts["eligible_prediction_groups"] = len(eligible_group_ids)
    if eligible.empty:
        raise ChemicalSplitError("No surfaces satisfy the anchored benchmark contract")
    return eligible, counts


def _aggregate_prediction_groups(eligible: pd.DataFrame) -> pd.DataFrame:
    consistency_columns = ["drug_a_id", "drug_b_id", "cell_id"]
    unique_counts = eligible.groupby("prediction_group_id")[consistency_columns].nunique()
    inconsistent = unique_counts.index[(unique_counts != 1).any(axis=1)].tolist()
    if inconsistent:
        raise ChemicalSplitError(
            "prediction_group_id changes drug pair or cell across repeats: "
            f"{inconsistent[:10]}"
        )
    grouped = (
        eligible.groupby("prediction_group_id", as_index=False, sort=True)
        .agg(
            drug_a_id=("drug_a_id", "first"),
            drug_b_id=("drug_b_id", "first"),
            cell_id=("cell_id", "first"),
            surface_replicate_count=("surface_replicate_id", "size"),
            point_count=("point_count", "sum"),
            primary_points=("primary_points", "sum"),
        )
        .sort_values("prediction_group_id", kind="mergesort")
        .reset_index(drop=True)
    )
    return grouped


def prepare_one_new_scaffold_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Prepare, audit, hash, and freeze the five-fold chemical benchmark ledgers."""

    paths = _source_paths(config)
    _verify_required_files(paths)
    drugs = pd.read_csv(paths["drugs"], keep_default_na=False)
    clustering = build_chemical_clusters(
        drugs,
        similarity_threshold=config.similarity_threshold,
        fingerprint_radius=config.fingerprint_radius,
        fingerprint_bits=config.fingerprint_bits,
    )
    structure_mapping = _read_drug_structure_mapping(paths["drug_structure_mapping"])
    expected_mapping = clustering.assignments.set_index("drug_id")["structure_key"].to_dict()
    mismatched_mapping = sorted(
        drug_id
        for drug_id, structure_key in expected_mapping.items()
        if structure_mapping.get(drug_id) != structure_key
    )
    if mismatched_mapping:
        raise ChemicalSplitError(
            "Prepared structure mapping disagrees with drugs.csv: "
            f"{mismatched_mapping[:10]}"
        )
    surfaces = pd.read_parquet(paths["surface_index"])
    cell_mapping = pd.read_csv(paths["cell_mapping"], keep_default_na=False)
    eligible, attrition = _eligible_surface_cohort(
        surfaces, clustering.assignments, cell_mapping, structure_mapping
    )
    group_base = _aggregate_prediction_groups(eligible)
    groups = assign_one_new_scaffold_folds(
        group_base,
        clustering.assignments,
        folds=config.folds,
        validation_offset=config.validation_offset,
        split_seed=config.split_seed,
        protocol=config.protocol,
    )
    split_audit = audit_split_assignments(
        groups, clustering.assignments, folds=config.folds, require_nonempty=True
    )
    attrition["distinct_cluster_prediction_groups"] = int(
        (groups["heldout_cluster_id"] != groups["partner_cluster_id"]).sum()
    )
    role_columns = [
        column
        for fold in range(config.folds)
        for column in (f"role_{fold}", f"exclusion_reason_{fold}")
    ]
    surface_join_columns = [
        "prediction_group_id",
        "heldout_drug_id",
        "partner_drug_id",
        "heldout_cluster_id",
        "partner_cluster_id",
        "heldout_cluster_bin",
        "partner_cluster_bin",
        "designated_drug_id",
        "designated_cluster_id",
        "designated_identity_group",
        "designated_structure_key",
        "designated_cluster_bin",
        *role_columns,
        *[column for fold in range(config.folds) for column in (
            f"role_fold_{fold}", f"exclusion_reason_fold_{fold}"
        )],
    ]
    surface_ledger = eligible.merge(
        groups.loc[:, surface_join_columns],
        on="prediction_group_id",
        how="left",
        validate="many_to_one",
    ).sort_values("surface_replicate_id", kind="mergesort").reset_index(drop=True)
    if surface_ledger[role_columns].isna().any(axis=None):
        raise ChemicalSplitError("A surface repeat is missing its atomic group assignment")
    for fold in range(config.folds):
        role_column = f"role_{fold}"
        expected_roles = groups.set_index("prediction_group_id")[role_column]
        actual_roles = surface_ledger["prediction_group_id"].map(expected_roles)
        if not actual_roles.equals(surface_ledger[role_column]):
            raise ChemicalSplitError(f"Fold {fold} surface roles are not group-atomic")

    output = config.output_dir
    cluster_path = output / "cluster_assignments.csv"
    group_path = output / "group_assignments.parquet"
    surface_path = output / "surface_assignments.parquet"
    _write_csv_atomic(cluster_path, clustering.assignments)
    _write_parquet_atomic(group_path, groups)
    _write_parquet_atomic(surface_path, surface_ledger)
    artifacts = {
        "cluster_assignments": {
            **_file_record(cluster_path, relative_to=output),
            "rows": len(clustering.assignments),
        },
        "group_assignments": {
            **_file_record(group_path, relative_to=output),
            "rows": len(groups),
        },
        "surface_assignments": {
            **_file_record(surface_path, relative_to=output),
            "rows": len(surface_ledger),
        },
    }
    sources = {name: _file_record(path) for name, path in paths.items()}
    split_payload = {
        "split_version": CHEMICAL_SPLIT_VERSION,
        "protocol": config.protocol,
        "information_setting": config.information_setting,
        "endpoint": config.endpoint,
        "endpoint_decision_status": config.endpoint_decision_status,
        "folds": config.folds,
        "validation_offset": config.validation_offset,
        "split_seed": config.split_seed,
        "fingerprint": clustering.audit["fingerprint"],
        "rdkit_version": clustering.audit["rdkit_version"],
        "sources": sources,
        "artifacts": artifacts,
    }
    split_hash = hashlib.sha256(_canonical_json_bytes(split_payload)).hexdigest()
    manifest: dict[str, Any] = {
        **split_payload,
        "split_hash": split_hash,
        "attrition": attrition,
        "chemical_clustering_audit": clustering.audit,
        "split_audit": split_audit,
        "status": "passed",
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    _write_json_atomic(output / "benchmark_manifest.json", manifest)
    return load_one_new_scaffold_benchmark(config).manifest


def _resolve_generated_path(output: Path, record: Mapping[str, Any]) -> Path:
    candidate = Path(str(record.get("path", "")))
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ChemicalSplitError("Generated artifact path must be a safe relative path")
    return (output / candidate).resolve()


def _verify_file_record(path: Path, record: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise ChemicalSplitError(f"Frozen artifact is missing: {path}")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise ChemicalSplitError(f"Frozen artifact byte size changed: {path}")
    if _clean_text(record.get("sha256")) != _sha256_file(path):
        raise ChemicalSplitError(f"Frozen artifact hash changed: {path}")


def load_one_new_scaffold_benchmark(config: BenchmarkConfig) -> PreparedChemicalSplit:
    """Load a frozen split only after verifying manifest, source, and ledger hashes."""

    manifest_path = config.output_dir / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise ChemicalSplitError(f"Benchmark manifest is missing: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as error:
        raise ChemicalSplitError("Benchmark manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ChemicalSplitError("Benchmark manifest must contain an object")
    recorded_manifest_hash = _clean_text(manifest.get("manifest_sha256"))
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(
        _canonical_json_bytes(unhashed_manifest)
    ).hexdigest()
    if recorded_manifest_hash != actual_manifest_hash:
        raise ChemicalSplitError("Benchmark manifest content hash changed")
    expected_context = {
        "split_version": CHEMICAL_SPLIT_VERSION,
        "protocol": config.protocol,
        "information_setting": config.information_setting,
        "endpoint": config.endpoint,
        "endpoint_decision_status": config.endpoint_decision_status,
        "folds": config.folds,
        "validation_offset": config.validation_offset,
        "split_seed": config.split_seed,
        "rdkit_version": rdBase.rdkitVersion,
    }
    mismatched = sorted(
        key for key, expected in expected_context.items() if manifest.get(key) != expected
    )
    fingerprint = manifest.get("fingerprint", {})
    expected_fingerprint = {
        "kind": "Morgan",
        "molecule_scope": "whole_canonical_molecule",
        "radius": config.fingerprint_radius,
        "bits": config.fingerprint_bits,
        "similarity": "Tanimoto",
        "threshold_inclusive": config.similarity_threshold,
    }
    if fingerprint != expected_fingerprint:
        mismatched.append("fingerprint")
    if mismatched:
        raise ChemicalSplitError(f"Benchmark manifest context mismatch: {mismatched}")
    source_paths = _source_paths(config)
    recorded_sources = manifest.get("sources", {})
    if set(recorded_sources) != set(source_paths):
        raise ChemicalSplitError("Benchmark source inventory changed")
    for name, path in source_paths.items():
        record = recorded_sources[name]
        if _clean_text(record.get("path")) != str(path.resolve()):
            raise ChemicalSplitError(f"Benchmark source path changed: {name}")
        _verify_file_record(path, record)
    artifacts = manifest.get("artifacts", {})
    expected_artifacts = {
        "cluster_assignments",
        "group_assignments",
        "surface_assignments",
    }
    if set(artifacts) != expected_artifacts:
        raise ChemicalSplitError("Benchmark generated-artifact inventory changed")
    artifact_paths: dict[str, Path] = {}
    for name, record in artifacts.items():
        path = _resolve_generated_path(config.output_dir, record)
        _verify_file_record(path, record)
        artifact_paths[name] = path
    clusters = pd.read_csv(artifact_paths["cluster_assignments"], keep_default_na=False)
    groups = pd.read_parquet(artifact_paths["group_assignments"])
    surfaces = pd.read_parquet(artifact_paths["surface_assignments"])
    for name, frame in (
        ("cluster_assignments", clusters),
        ("group_assignments", groups),
        ("surface_assignments", surfaces),
    ):
        if len(frame) != int(artifacts[name].get("rows", -1)):
            raise ChemicalSplitError(f"{name} row count differs from its manifest")
    _normalize_cluster_assignments(clusters)
    _normalize_prediction_groups(groups)
    audit_split_assignments(groups, clusters, folds=config.folds, require_nonempty=True)
    if surfaces["surface_replicate_id"].duplicated().any():
        raise ChemicalSplitError("Frozen surface ledger has duplicate replicate IDs")
    group_roles = groups.set_index("prediction_group_id")
    unknown_groups = sorted(set(surfaces["prediction_group_id"]) - set(group_roles.index))
    if unknown_groups:
        raise ChemicalSplitError(f"Surface ledger references unknown groups: {unknown_groups[:10]}")
    for fold in range(config.folds):
        role_column = f"role_{fold}"
        alias_column = f"role_fold_{fold}"
        expected = surfaces["prediction_group_id"].map(group_roles[role_column])
        if not expected.equals(surfaces[role_column]):
            raise ChemicalSplitError(f"Fold {fold} surface assignments are not group-atomic")
        if alias_column not in groups or alias_column not in surfaces:
            raise ChemicalSplitError(f"Fold {fold} role alias is missing")
        if not groups[role_column].equals(groups[alias_column]):
            raise ChemicalSplitError(f"Fold {fold} group role aliases disagree")
        if not surfaces[role_column].equals(surfaces[alias_column]):
            raise ChemicalSplitError(f"Fold {fold} surface role aliases disagree")
    split_payload_keys = {
        "split_version",
        "protocol",
        "information_setting",
        "endpoint",
        "endpoint_decision_status",
        "folds",
        "validation_offset",
        "split_seed",
        "fingerprint",
        "rdkit_version",
        "sources",
        "artifacts",
    }
    split_payload = {key: manifest[key] for key in split_payload_keys}
    if manifest.get("split_hash") != hashlib.sha256(
        _canonical_json_bytes(split_payload)
    ).hexdigest():
        raise ChemicalSplitError("Frozen split hash changed")
    if manifest.get("status") != "passed":
        raise ChemicalSplitError("Frozen benchmark status is not passed")
    return PreparedChemicalSplit(
        manifest=manifest,
        clusters=clusters,
        groups=groups,
        surfaces=surfaces,
    )


def load_benchmark_manifest(config: BenchmarkConfig) -> dict[str, Any]:
    """Verify the complete frozen benchmark and return its manifest."""

    return load_one_new_scaffold_benchmark(config).manifest


__all__ = [
    "CHEMICAL_SPLIT_VERSION",
    "ChemicalClusterResult",
    "ChemicalSplitError",
    "PreparedChemicalSplit",
    "assign_one_new_scaffold_folds",
    "audit_split_assignments",
    "build_chemical_clusters",
    "designated_endpoint",
    "load_benchmark_manifest",
    "load_one_new_scaffold_benchmark",
    "prepare_one_new_scaffold_benchmark",
]

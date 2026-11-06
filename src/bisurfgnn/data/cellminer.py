from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

from .batch import GraphItem


CELLMINER_PIPELINE_VERSION = "cellminer-m1b-v3"
PATHWAY_TOPOLOGY_VERSION = "hallmark-jaccard-v2"
PATHWAY_TRANSFORM_VERSION = "train-only-pathway-z-v3"

DEFAULT_EXPRESSION_MEMBER = "output/html/RNA__RNA_seq_composite_expression.html"
EXPRESSION_METADATA_COLUMNS = (
    "Gene name",
    "Entrez gene id",
    "Chromosome",
    "Start",
    "End",
    "Cytoband",
)


class CellMinerDataError(ValueError):
    """Raised when a frozen CellMiner/MSigDB input violates its data contract."""


def _hash_tensor(digest: Any, tensor: Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _compute_expression_content_hash(
    *,
    cell_ids: Sequence[str],
    gene_ids: Sequence[str],
    entrez_ids: Sequence[str],
    values: Tensor,
    observed_mask: Tensor,
    source_sha256: str,
    value_scale: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"expression-matrix-v1")
    for sequence in (cell_ids, gene_ids, entrez_ids):
        digest.update(json.dumps(list(sequence), ensure_ascii=True).encode("utf-8"))
    digest.update(source_sha256.encode("ascii"))
    digest.update(value_scale.encode("utf-8"))
    _hash_tensor(digest, values)
    _hash_tensor(digest, observed_mask)
    return digest.hexdigest()


def _compute_topology_hash(
    *,
    catalog_hash: str,
    top_k: int,
    pathway_ids: Sequence[str],
    adjacency: Tensor,
    member_counts: Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(PATHWAY_TOPOLOGY_VERSION.encode("ascii"))
    digest.update(catalog_hash.encode("ascii"))
    digest.update(str(top_k).encode("ascii"))
    digest.update("\n".join(pathway_ids).encode("utf-8"))
    _hash_tensor(digest, adjacency)
    _hash_tensor(digest, member_counts)
    return digest.hexdigest()


@dataclass(frozen=True)
class CellMinerConfig:
    cellminer_zip: Path
    cellminer_manifest: Path
    hallmark_gmt: Path
    hallmark_manifest: Path
    cell_aliases: Path
    processed_dir: Path
    expression_member: str = DEFAULT_EXPRESSION_MEMBER
    expected_gene_rows: int = 23_808
    expected_expression_cells: int = 60
    pathway_top_k: int = 3
    min_pathway_genes: int = 10
    clip_value: float = 5.0
    std_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for field_name in (
            "cellminer_zip",
            "cellminer_manifest",
            "hallmark_gmt",
            "hallmark_manifest",
            "cell_aliases",
            "processed_dir",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)).resolve())
        if self.expected_gene_rows < 1 or self.expected_expression_cells < 1:
            raise ValueError("Expected expression dimensions must be positive")
        if self.pathway_top_k < 0 or self.min_pathway_genes < 1:
            raise ValueError("Invalid pathway topology configuration")
        if self.clip_value <= 0 or self.std_epsilon <= 0:
            raise ValueError("clip_value and std_epsilon must be positive")
        member = Path(self.expression_member)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("expression_member must be a safe relative ZIP member")


@dataclass(frozen=True)
class ExpressionMatrix:
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    entrez_ids: tuple[str, ...]
    values: Tensor
    observed_mask: Tensor
    source_release: str
    source_sha256: str
    value_scale: str
    content_hash: str = ""

    def _current_content_hash(self) -> str:
        return _compute_expression_content_hash(
            cell_ids=self.cell_ids,
            gene_ids=self.gene_ids,
            entrez_ids=self.entrez_ids,
            values=self.values,
            observed_mask=self.observed_mask,
            source_sha256=self.source_sha256,
            value_scale=self.value_scale,
        )

    def assert_current_hash(self) -> None:
        if self.content_hash != self._current_content_hash():
            raise CellMinerDataError(
                "Expression tensors changed after their content hash was computed"
            )

    def __post_init__(self) -> None:
        expected = (len(self.cell_ids), len(self.gene_ids))
        if self.values.shape != expected or self.observed_mask.shape != expected:
            raise ValueError(
                f"Expression tensors must have shape {expected}; got "
                f"{tuple(self.values.shape)} and {tuple(self.observed_mask.shape)}"
            )
        if len(self.entrez_ids) != len(self.gene_ids):
            raise ValueError("entrez_ids and gene_ids must have identical length")
        if len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("Expression cell IDs must be unique")
        if len(set(self.gene_ids)) != len(self.gene_ids):
            raise ValueError("Expression gene IDs must be unique")
        if self.values.dtype != torch.float32:
            raise TypeError("Expression values must be float32")
        if self.observed_mask.dtype != torch.bool:
            raise TypeError("Expression observed_mask must be boolean")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("Stored expression values must be finite; missing values use the mask")
        expected_hash = self._current_content_hash()
        if self.content_hash and self.content_hash != expected_hash:
            raise ValueError("Expression content hash does not match its tensors")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", expected_hash)

    def cell_indices(self, requested: Collection[str]) -> list[int]:
        index = {cell_id: position for position, cell_id in enumerate(self.cell_ids)}
        normalized = [normalize_cell_label(cell_id) for cell_id in requested]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Cell selections must not contain duplicates")
        missing = sorted(set(normalized) - set(index))
        if missing:
            raise KeyError(f"Expression cells not found: {missing}")
        return [index[cell_id] for cell_id in normalized]


@dataclass(frozen=True)
class GeneSetCatalog:
    pathway_ids: tuple[str, ...]
    genes: tuple[tuple[str, ...], ...]
    descriptions: tuple[str, ...]
    source_release: str
    source_sha256: str
    catalog_hash: str

    def __post_init__(self) -> None:
        size = len(self.pathway_ids)
        if size < 1 or len(self.genes) != size or len(self.descriptions) != size:
            raise ValueError("Gene-set catalog fields must have equal non-zero length")
        if len(set(self.pathway_ids)) != size:
            raise ValueError("Pathway IDs must be unique")
        if any(not members for members in self.genes):
            raise ValueError("Every pathway must contain at least one gene")


@dataclass(frozen=True)
class PathwayTopology:
    pathway_ids: tuple[str, ...]
    adjacency: Tensor
    member_counts: Tensor
    catalog_hash: str
    topology_hash: str
    top_k: int

    def _current_content_hash(self) -> str:
        return _compute_topology_hash(
            catalog_hash=self.catalog_hash,
            top_k=self.top_k,
            pathway_ids=self.pathway_ids,
            adjacency=self.adjacency,
            member_counts=self.member_counts,
        )

    def assert_current_hash(self) -> None:
        if self.topology_hash != self._current_content_hash():
            raise CellMinerDataError(
                "Pathway topology tensors changed after their hash was computed"
            )

    def __post_init__(self) -> None:
        pathways = len(self.pathway_ids)
        if self.adjacency.shape != (pathways, pathways):
            raise ValueError("Pathway adjacency has the wrong shape")
        if self.member_counts.shape != (pathways,):
            raise ValueError("Pathway member_counts has the wrong shape")
        if self.adjacency.dtype != torch.float32:
            raise TypeError("Pathway adjacency must be float32")
        if not bool(torch.isfinite(self.adjacency).all()):
            raise ValueError("Pathway adjacency contains non-finite values")
        if self.member_counts.dtype != torch.float32:
            raise TypeError("Pathway member_counts must be float32")
        if not bool(torch.isfinite(self.member_counts).all()) or not bool(
            (self.member_counts > 0).all()
        ):
            raise ValueError("Pathway member_counts must be finite and positive")
        if not torch.equal(self.adjacency, self.adjacency.T):
            raise ValueError("Pathway adjacency must be symmetric")
        if not torch.equal(torch.diag(self.adjacency), torch.ones(pathways)):
            raise ValueError("Pathway adjacency must contain unit self-loops")
        expected_hash = self._current_content_hash()
        if self.topology_hash != expected_hash:
            raise ValueError("Pathway topology hash does not match its tensor content")


@dataclass(frozen=True)
class PathwayTransformState:
    gene_ids: tuple[str, ...]
    pathway_ids: tuple[str, ...]
    fit_cell_ids: tuple[str, ...]
    split_hash: str
    expression_hash: str
    catalog_hash: str
    topology_hash: str
    clip_value: float
    std_epsilon: float
    gene_mean: Tensor
    gene_std: Tensor
    gene_fit_counts: Tensor
    gene_active: Tensor
    membership: Tensor
    available_member_counts: Tensor
    pathway_mean: Tensor
    pathway_std: Tensor
    pathway_active: Tensor
    state_hash: str

    def assert_current_hash(self) -> None:
        if not self.state_hash or self.state_hash != _compute_transform_state_hash(self):
            raise CellMinerDataError(
                "Pathway transform tensors changed after their hash was computed"
            )

    def __post_init__(self) -> None:
        genes = len(self.gene_ids)
        pathways = len(self.pathway_ids)
        if not self.split_hash or not self.fit_cell_ids:
            raise ValueError("Transform state requires a split hash and fit cells")
        if tuple(sorted(set(self.fit_cell_ids))) != self.fit_cell_ids:
            raise ValueError("fit_cell_ids must be sorted and unique")
        for name in ("gene_mean", "gene_std", "gene_fit_counts", "gene_active"):
            if getattr(self, name).shape != (genes,):
                raise ValueError(f"{name} has the wrong shape")
        if self.membership.shape != (pathways, genes):
            raise ValueError("Pathway membership has the wrong shape")
        for name in (
            "available_member_counts",
            "pathway_mean",
            "pathway_std",
            "pathway_active",
        ):
            if getattr(self, name).shape != (pathways,):
                raise ValueError(f"{name} has the wrong shape")
        if self.gene_active.dtype != torch.bool or self.pathway_active.dtype != torch.bool:
            raise TypeError("Active masks must be boolean")
        if self.membership.dtype != torch.bool:
            raise TypeError("Pathway membership must be boolean")
        for tensor in (
            self.gene_mean,
            self.gene_std,
            self.gene_fit_counts,
            self.available_member_counts,
            self.pathway_mean,
            self.pathway_std,
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Transform state contains non-finite values")


def default_cellminer_config(project_root: str | Path) -> CellMinerConfig:
    root = Path(project_root).resolve()
    return CellMinerConfig(
        cellminer_zip=(
            root / "data/raw/cellminer/nci60_RNA__RNA_seq_composite_expression.zip"
        ),
        cellminer_manifest=root / "data/raw_manifests/cellminer_rnaseq.json",
        hallmark_gmt=root / "data/raw/msigdb/h.all.v2026.1.Hs.symbols.gmt",
        hallmark_manifest=root / "data/raw_manifests/msigdb_hallmark.json",
        cell_aliases=root / "data/mappings/nci60_cell_aliases.json",
        processed_dir=root / "data/processed/cellminer",
    )


def load_cellminer_config(path: str | Path) -> CellMinerConfig:
    """Load an M1b YAML file while retaining deterministic project-relative defaults.

    Both concise flat keys and grouped ``cellminer``, ``hallmark``, ``mapping``,
    ``pathway``, and ``transform`` sections are accepted. Missing values use the
    repository defaults so a config can override only scientific parameters.
    """

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise CellMinerDataError("CellMiner configuration must be a YAML mapping")
    configured_root = raw.get("project_root")
    if configured_root is None:
        project_root = config_path.parent.parent
    else:
        candidate = Path(str(configured_root))
        project_root = (
            candidate.resolve()
            if candidate.is_absolute()
            else (config_path.parent / candidate).resolve()
        )
    defaults = default_cellminer_config(project_root)
    cellminer = dict(raw.get("cellminer", {}))
    hallmark = dict(raw.get("hallmark", {}))
    mapping = dict(raw.get("mapping", {}))
    pathway = dict(raw.get("pathway", {}))
    transform = dict(raw.get("transform", {}))

    def resolve_path(value: Any, fallback: Path) -> Path:
        if value is None:
            return fallback
        candidate = Path(str(value))
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (project_root / candidate).resolve()
        )

    return CellMinerConfig(
        cellminer_zip=resolve_path(
            raw.get("cellminer_zip", cellminer.get("zip")), defaults.cellminer_zip
        ),
        cellminer_manifest=resolve_path(
            raw.get("cellminer_manifest", cellminer.get("manifest")),
            defaults.cellminer_manifest,
        ),
        hallmark_gmt=resolve_path(
            raw.get("hallmark_gmt", hallmark.get("gmt")), defaults.hallmark_gmt
        ),
        hallmark_manifest=resolve_path(
            raw.get("hallmark_manifest", hallmark.get("manifest")),
            defaults.hallmark_manifest,
        ),
        cell_aliases=resolve_path(
            raw.get("cell_aliases", mapping.get("aliases")), defaults.cell_aliases
        ),
        processed_dir=resolve_path(raw.get("processed_dir"), defaults.processed_dir),
        expression_member=str(
            raw.get(
                "expression_member",
                cellminer.get("expression_member", defaults.expression_member),
            )
        ),
        expected_gene_rows=int(
            raw.get(
                "expected_gene_rows",
                cellminer.get("expected_gene_rows", defaults.expected_gene_rows),
            )
        ),
        expected_expression_cells=int(
            raw.get(
                "expected_expression_cells",
                cellminer.get(
                    "expected_expression_cells", defaults.expected_expression_cells
                ),
            )
        ),
        pathway_top_k=int(
            raw.get("pathway_top_k", pathway.get("top_k", defaults.pathway_top_k))
        ),
        min_pathway_genes=int(
            raw.get(
                "min_pathway_genes",
                pathway.get("min_genes", defaults.min_pathway_genes),
            )
        ),
        clip_value=float(
            raw.get("clip_value", transform.get("clip", defaults.clip_value))
        ),
        std_epsilon=float(
            raw.get("std_epsilon", transform.get("std_epsilon", defaults.std_epsilon))
        ),
    )


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise CellMinerDataError(f"JSON root must be an object: {path}")
    return payload


def _manifest_entry(manifest: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise CellMinerDataError("Source manifest has no files list")
    matches = [entry for entry in entries if Path(str(entry.get("filename", ""))).name == path.name]
    if len(matches) != 1:
        raise CellMinerDataError(f"Manifest must contain exactly one entry for {path.name}")
    return matches[0]


def _verify_regular_file(path: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != int(entry["bytes"]):
        raise CellMinerDataError(
            f"Size mismatch for {path.name}: expected {entry['bytes']}, got {actual_bytes}"
        )
    if actual_sha256.lower() != str(entry["sha256"]).lower():
        raise CellMinerDataError(f"SHA256 mismatch for {path.name}")
    return {"filename": path.name, "bytes": actual_bytes, "sha256": actual_sha256}


def verify_cellminer_source(config: CellMinerConfig) -> dict[str, Any]:
    manifest = _load_json(config.cellminer_manifest)
    entry = _manifest_entry(manifest, config.cellminer_zip)
    verified = _verify_regular_file(config.cellminer_zip, entry)
    expected_members = entry.get("zip_members")
    if not isinstance(expected_members, list) or not expected_members:
        raise CellMinerDataError("CellMiner manifest must enumerate ZIP members")

    expected_by_name = {str(item["filename"]): item for item in expected_members}
    if len(expected_by_name) != len(expected_members):
        raise CellMinerDataError("Duplicate ZIP member names in CellMiner manifest")
    with zipfile.ZipFile(config.cellminer_zip) as archive:
        infos = archive.infolist()
        actual_names = [info.filename for info in infos]
        for name in actual_names:
            member_path = Path(name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise CellMinerDataError(f"Unsafe ZIP member: {name}")
        if set(actual_names) != set(expected_by_name) or len(actual_names) != len(expected_by_name):
            raise CellMinerDataError(
                f"Unexpected CellMiner ZIP members: {sorted(actual_names)}"
            )
        member_audit: list[dict[str, Any]] = []
        for info in infos:
            expected = expected_by_name[info.filename]
            if info.file_size != int(expected["bytes"]):
                raise CellMinerDataError(f"Uncompressed size mismatch: {info.filename}")
            crc32 = f"{info.CRC:08x}"
            if crc32 != str(expected["crc32"]).lower():
                raise CellMinerDataError(f"CRC32 mismatch: {info.filename}")
            member_audit.append(
                {"filename": info.filename, "bytes": info.file_size, "crc32": crc32}
            )
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise CellMinerDataError(f"ZIP CRC verification failed: {corrupt_member}")
    if config.expression_member not in expected_by_name:
        raise CellMinerDataError(
            f"Expression member is not frozen in manifest: {config.expression_member}"
        )
    verified["zip_members"] = member_audit
    verified["dataset"] = str(manifest.get("dataset", ""))
    verified["release"] = str(manifest.get("release", ""))
    verified["value_scale"] = str(manifest.get("value_scale", ""))
    verified["embedded_metadata"] = dict(entry.get("embedded_metadata", {}))
    return verified


def verify_hallmark_source(config: CellMinerConfig) -> dict[str, Any]:
    manifest = _load_json(config.hallmark_manifest)
    entry = _manifest_entry(manifest, config.hallmark_gmt)
    verified = _verify_regular_file(config.hallmark_gmt, entry)
    verified["dataset"] = str(manifest.get("dataset", ""))
    verified["release"] = str(manifest.get("release", ""))
    verified["identifier_type"] = str(manifest.get("identifier_type", ""))
    verified["expected_gene_sets"] = int(entry.get("gene_sets", 0))
    return verified


def _download_manifest_file(
    destination: Path,
    manifest_path: Path,
    *,
    timeout_seconds: int = 120,
) -> None:
    manifest = _load_json(manifest_path)
    entry = _manifest_entry(manifest, destination)
    if destination.exists():
        try:
            _verify_regular_file(destination, entry)
        except Exception as error:
            raise CellMinerDataError(
                f"Existing source does not match its manifest: {destination}"
            ) from error
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    urls = [str(entry["url"])]
    if entry.get("fallback_url"):
        urls.append(str(entry["fallback_url"]))
    failures: list[str] = []
    for url in urls:
        partial.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "BiSurfGNN/0.1"})
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                with partial.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            _verify_regular_file(partial, entry)
            os.replace(partial, destination)
            return
        except Exception as error:
            failures.append(f"{url}: {type(error).__name__}: {error}")
    partial.unlink(missing_ok=True)
    raise RuntimeError("All frozen source downloads failed: " + " | ".join(failures))


def download_cellminer_sources(
    config: CellMinerConfig,
    *,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Download the two frozen M1b assets with .part and manifest verification."""

    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    _download_manifest_file(
        config.cellminer_zip,
        config.cellminer_manifest,
        timeout_seconds=timeout_seconds,
    )
    _download_manifest_file(
        config.hallmark_gmt,
        config.hallmark_manifest,
        timeout_seconds=timeout_seconds,
    )
    return {
        "cellminer": verify_cellminer_source(config),
        "hallmark": verify_hallmark_source(config),
    }


def normalize_cell_label(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(character for character in text if unicodedata.category(character) != "Cc")
    text = re.sub(r"\s+", " ", text).strip().upper()
    if not text:
        raise CellMinerDataError("Cell label became empty after normalization")
    return text


def normalize_gene_symbol(value: Any) -> str:
    symbol = unicodedata.normalize("NFKC", str(value)).strip().upper()
    if not symbol:
        raise CellMinerDataError("Gene symbol must not be empty")
    return symbol


def load_cell_alias_rules(path: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(path).resolve())
    normalization = payload.get("normalization")
    expected_normalization = {
        "unicode": "NFKC",
        "remove_unicode_control_characters": True,
        "collapse_whitespace": True,
        "case": "upper",
    }
    if normalization != expected_normalization:
        raise CellMinerDataError("Cell alias normalization rules do not match the code contract")
    required = (
        "panel_prefixes",
        "aliases",
        "excluded_almanac_cells",
        "expression_only_cells",
        "known_source_alias_groups",
        "lineage_groups",
        "lineage_policy",
    )
    if any(key not in payload for key in required):
        raise CellMinerDataError("Cell alias JSON is missing required sections")

    excluded = {normalize_cell_label(key) for key in payload["excluded_almanac_cells"]}
    expression_only = {
        normalize_cell_label(key) for key in payload["expression_only_cells"]
    }
    if "MDA-MB-468" not in excluded:
        raise CellMinerDataError("MDA-MB-468 must be explicitly excluded")
    if "ME:MDA-N" not in expression_only:
        raise CellMinerDataError("ME:MDA-N must be explicitly expression_only")
    aliases = {
        normalize_cell_label(key): normalize_cell_label(value)
        for key, value in payload["aliases"].items()
    }
    if "ME:MDA-N" in aliases.values():
        raise CellMinerDataError("MDA-N must never be used as an ALMANAC alias")
    raw_lineage_groups = payload["lineage_groups"]
    if not isinstance(raw_lineage_groups, dict):
        raise CellMinerDataError("lineage_groups must be a mapping")
    lineage_groups: dict[str, str] = {}
    for raw_cell_id, raw_group_id in raw_lineage_groups.items():
        cell_id = normalize_cell_label(raw_cell_id)
        group_id = normalize_cell_label(raw_group_id)
        previous = lineage_groups.get(cell_id)
        if previous is not None and previous != group_id:
            raise CellMinerDataError(
                f"Conflicting lineage groups after normalization for {cell_id}"
            )
        lineage_groups[cell_id] = group_id
    required_lineage_pairs = (
        ("ME:M14", "ME:MDA-MB-435"),
        ("CNS:SNB-19", "CNS:U251"),
        ("OV:OVCAR-8", "OV:NCI/ADR-RES"),
    )
    for left, right in required_lineage_pairs:
        if left not in lineage_groups or lineage_groups.get(left) != lineage_groups.get(right):
            raise CellMinerDataError(
                f"Required shared-lineage pair is not indivisible: {left}, {right}"
            )
    if not str(payload["lineage_policy"]).strip():
        raise CellMinerDataError("lineage_policy must be non-empty")
    validated = dict(payload)
    validated["lineage_groups"] = lineage_groups
    return validated


def _validate_embedded_metadata(html_bytes: bytes, verification: Mapping[str, Any]) -> None:
    head = html_bytes[:20_000].decode("utf-8", errors="replace")
    metadata = verification.get("embedded_metadata", {})
    required_text = (
        f"CellMiner Database Version: {metadata.get('cellminer_database_version', '')}",
        f"Human Genome Version: {metadata.get('human_genome_version', '')}",
        f"Date: {metadata.get('date', '')}",
        str(metadata.get("value_description", "")),
    )
    missing = [text for text in required_text if text and text not in head]
    if missing:
        raise CellMinerDataError(f"CellMiner HTML metadata mismatch: {missing}")


def parse_cellminer_expression(
    config: CellMinerConfig,
    verification: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, ExpressionMatrix]:
    verification = dict(verification or verify_cellminer_source(config))
    with zipfile.ZipFile(config.cellminer_zip) as archive:
        html_bytes = archive.read(config.expression_member)
    _validate_embedded_metadata(html_bytes, verification)
    try:
        tables = pd.read_html(io.BytesIO(html_bytes))
    except ImportError as error:
        raise RuntimeError(
            "Parsing the CellMiner HTML table requires pandas HTML support"
        ) from error
    if len(tables) != 1:
        raise CellMinerDataError(f"Expected one expression table, found {len(tables)}")
    frame = tables[0].dropna(how="all").reset_index(drop=True)
    if tuple(str(column) for column in frame.columns[:6]) != EXPRESSION_METADATA_COLUMNS:
        raise CellMinerDataError("Unexpected CellMiner expression metadata columns")
    if frame.shape[0] != config.expected_gene_rows:
        raise CellMinerDataError(
            f"Expected {config.expected_gene_rows} genes, found {frame.shape[0]}"
        )
    if frame.shape[1] != len(EXPRESSION_METADATA_COLUMNS) + config.expected_expression_cells:
        raise CellMinerDataError(f"Unexpected expression table shape: {frame.shape}")

    raw_cell_columns = [str(column) for column in frame.columns[6:]]
    cell_ids = tuple(normalize_cell_label(column) for column in raw_cell_columns)
    if len(set(cell_ids)) != config.expected_expression_cells:
        raise CellMinerDataError("CellMiner expression columns are not unique after normalization")
    if "BR:MDA-MB-468" in cell_ids:
        raise CellMinerDataError("Frozen CellMiner matrix unexpectedly contains MDA-MB-468")
    if "ME:MDA-N" not in cell_ids:
        raise CellMinerDataError("Frozen CellMiner matrix must contain expression-only ME:MDA-N")
    frame = frame.rename(columns=dict(zip(raw_cell_columns, cell_ids, strict=True)))

    frame["Gene name"] = frame["Gene name"].astype(str).str.strip()
    gene_ids = tuple(normalize_gene_symbol(value) for value in frame["Gene name"])
    if len(set(gene_ids)) != len(gene_ids):
        raise CellMinerDataError("Gene names are not unique after normalization")
    entrez = pd.to_numeric(frame["Entrez gene id"], errors="coerce").astype("Int64")
    frame["Entrez gene id"] = entrez
    entrez_ids = tuple("" if pd.isna(value) else str(int(value)) for value in entrez)

    numeric = frame.loc[:, list(cell_ids)].apply(pd.to_numeric, errors="coerce")
    values_numpy = numeric.to_numpy(dtype=np.float32, copy=True).T
    observed_numpy = np.isfinite(values_numpy)
    if np.isinf(values_numpy).any():
        raise CellMinerDataError("Expression matrix contains infinite values")
    if observed_numpy.any() and float(values_numpy[observed_numpy].min()) < 0:
        raise CellMinerDataError("log2(FPKM + 1) expression values must be non-negative")
    values_numpy[~observed_numpy] = 0.0
    frame.loc[:, list(cell_ids)] = numeric.astype(np.float32)

    expression = ExpressionMatrix(
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        entrez_ids=entrez_ids,
        values=torch.from_numpy(values_numpy),
        observed_mask=torch.from_numpy(observed_numpy),
        source_release=str(verification.get("release", "")),
        source_sha256=str(verification["sha256"]),
        value_scale=str(verification.get("value_scale", "")),
    )
    return frame, expression


def _read_almanac_cells(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        frame = pd.read_csv(Path(source), dtype=str, keep_default_na=False)
    required = {"cell_id", "cell_name", "panel"}
    if not required.issubset(frame.columns):
        raise CellMinerDataError(f"ALMANAC cell table is missing: {sorted(required - set(frame))}")
    return frame


def _source_labels_from_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    canonical = str(record["cell_name"])
    encoded = str(record.get("source_cell_names", "")).strip()
    if not encoded:
        return (canonical,)
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise CellMinerDataError(
            f"Invalid source_cell_names JSON for {record['cell_id']}"
        ) from error
    if not isinstance(decoded, list) or not decoded:
        raise CellMinerDataError("source_cell_names must be a non-empty JSON list")
    labels = tuple(str(label) for label in decoded)
    if {normalize_cell_label(label) for label in labels} != {
        normalize_cell_label(canonical)
    }:
        raise CellMinerDataError(
            f"Source aliases do not collapse to canonical cell name: {record['cell_id']}"
        )
    declared_count = str(record.get("source_alias_count", "")).strip()
    if declared_count and int(declared_count) != len(labels):
        raise CellMinerDataError(f"source_alias_count mismatch for {record['cell_id']}")
    return labels


def build_strict_cell_mapping(
    almanac_cells: str | Path | pd.DataFrame,
    expression_cell_ids: Sequence[str],
    alias_json: str | Path,
) -> pd.DataFrame:
    rules = load_cell_alias_rules(alias_json)
    source = _read_almanac_cells(almanac_cells)
    expression_lookup = {
        normalize_cell_label(cell_id): normalize_cell_label(cell_id)
        for cell_id in expression_cell_ids
    }
    if len(expression_lookup) != len(expression_cell_ids):
        raise CellMinerDataError("Expression cell IDs are not unique")

    panel_prefixes = {
        normalize_cell_label(panel): normalize_cell_label(prefix)
        for panel, prefix in rules["panel_prefixes"].items()
    }
    aliases = {
        normalize_cell_label(key): normalize_cell_label(value)
        for key, value in rules["aliases"].items()
    }
    excluded = {
        normalize_cell_label(key): value
        for key, value in rules["excluded_almanac_cells"].items()
    }
    expression_only = {
        normalize_cell_label(key): value
        for key, value in rules["expression_only_cells"].items()
    }
    known_groups: dict[str, dict[str, Any]] = {}
    allowed_duplicate_biological_ids: set[str] = set()
    for group in rules["known_source_alias_groups"]:
        biological_id = normalize_cell_label(group["biological_cell_id"])
        source_labels = list(group.get("source_labels", []))
        if len(source_labels) < 2:
            raise CellMinerDataError("Known source alias groups must enumerate source labels")
        normalized_labels = {normalize_cell_label(label) for label in source_labels}
        if len(normalized_labels) != 1:
            raise CellMinerDataError(
                "Known source aliases must collapse under the frozen normalization rules"
            )
        normalized_name = next(iter(normalized_labels))
        known_groups[normalized_name] = dict(group, biological_cell_id=biological_id)
        allowed_duplicate_biological_ids.add(biological_id)
    sf_group = known_groups.get("SF-539")
    if sf_group is None or sf_group["biological_cell_id"] != "CNS:SF-539":
        raise CellMinerDataError("SF-539 source aliases must share CNS:SF-539")

    rows: list[dict[str, Any]] = []
    seen_exclusions: set[str] = set()
    for record in source.to_dict(orient="records"):
        source_cell_id = str(record["cell_id"])
        source_label = str(record["cell_name"])
        source_labels = _source_labels_from_record(record)
        normalized_name = normalize_cell_label(source_label)
        panel = normalize_cell_label(record["panel"])
        if panel not in panel_prefixes:
            raise CellMinerDataError(f"No frozen panel prefix for {record['panel']}")
        expected_prefix = panel_prefixes[panel]
        base = {
            "source_cell_id": source_cell_id,
            "source_label": source_label,
            "source_labels": json.dumps(source_labels, ensure_ascii=True),
            "source_alias_count": len(source_labels),
            "normalized_name": normalized_name,
            "panel": panel,
        }
        if normalized_name in excluded:
            excluded_entry = excluded[normalized_name]
            biological_id = normalize_cell_label(excluded_entry["biological_cell_id"])
            if biological_id.partition(":")[0] != expected_prefix:
                raise CellMinerDataError(
                    f"Excluded cell panel/target mismatch for {source_label}: "
                    f"{panel} -> {biological_id}"
                )
            rows.append(
                {
                    **base,
                    "expression_cell_id": "",
                    "biological_cell_id": biological_id,
                    "status": "excluded",
                    "mapping_method": "explicit_exclusion",
                    "reason": str(excluded_entry["reason"]),
                }
            )
            seen_exclusions.add(normalized_name)
            continue

        if normalized_name in known_groups:
            target = known_groups[normalized_name]["biological_cell_id"]
            method = "known_source_alias_group"
        elif normalized_name in aliases:
            target = aliases[normalized_name]
            method = "explicit_alias"
        else:
            target = f"{expected_prefix}:{normalized_name}"
            method = "panel_prefix_exact"
        target = normalize_cell_label(target)
        if target.partition(":")[0] != expected_prefix:
            raise CellMinerDataError(
                f"Cell panel/target mismatch for {source_label}: {panel} -> {target}"
            )
        if target not in expression_lookup:
            raise CellMinerDataError(
                f"Strict cell mapping has no expression column for {source_label}: {target}"
            )
        rows.append(
            {
                **base,
                "expression_cell_id": expression_lookup[target],
                "biological_cell_id": target,
                "status": "mapped",
                "mapping_method": method,
                "reason": "",
            }
        )

    missing_exclusions = set(excluded) - seen_exclusions
    if missing_exclusions:
        raise CellMinerDataError(
            f"Frozen exclusions were not found in ALMANAC cells: {sorted(missing_exclusions)}"
        )
    mapped = [row for row in rows if row["status"] == "mapped"]
    duplicates: dict[str, list[str]] = {}
    for row in mapped:
        duplicates.setdefault(row["biological_cell_id"], []).append(row["source_cell_id"])
    unexpected_duplicates = {
        biological_id: source_ids
        for biological_id, source_ids in duplicates.items()
        if len(source_ids) > 1 and biological_id not in allowed_duplicate_biological_ids
    }
    if unexpected_duplicates:
        raise CellMinerDataError(f"Unexpected duplicate biological cells: {unexpected_duplicates}")

    mapped_expression = {row["expression_cell_id"] for row in mapped}
    for expression_id, entry in sorted(expression_only.items()):
        if expression_id not in expression_lookup:
            raise CellMinerDataError(
                f"Frozen expression-only cell is absent from the matrix: {expression_id}"
            )
        if expression_id in mapped_expression:
            raise CellMinerDataError(
                f"Expression-only cell was incorrectly mapped from ALMANAC: {expression_id}"
            )
        rows.append(
            {
                "source_cell_id": "",
                "source_label": expression_id.split(":", 1)[-1],
                "source_labels": json.dumps(
                    [expression_id.split(":", 1)[-1]], ensure_ascii=True
                ),
                "source_alias_count": 1,
                "normalized_name": expression_id.split(":", 1)[-1],
                "panel": expression_id.split(":", 1)[0],
                "expression_cell_id": expression_id,
                "biological_cell_id": expression_id,
                "status": "expression_only",
                "mapping_method": "explicit_expression_only",
                "reason": str(entry["reason"]),
            }
        )

    accounted = mapped_expression | set(expression_only)
    unaccounted = set(expression_lookup) - accounted
    if unaccounted:
        raise CellMinerDataError(f"Unaccounted CellMiner expression columns: {sorted(unaccounted)}")
    return pd.DataFrame(rows).sort_values(
        ["status", "biological_cell_id", "source_cell_id"], kind="stable"
    ).reset_index(drop=True)


def load_hallmark_gene_sets(
    config: CellMinerConfig,
    verification: Mapping[str, Any] | None = None,
) -> GeneSetCatalog:
    verification = dict(verification or verify_hallmark_source(config))
    parsed: list[tuple[str, str, tuple[str, ...]]] = []
    with config.hallmark_gmt.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) < 3:
                raise CellMinerDataError(f"Invalid GMT line {line_number}")
            pathway_id = normalize_cell_label(fields[0])
            description = fields[1].strip()
            genes = tuple(
                sorted(
                    {normalize_gene_symbol(gene) for gene in fields[2:] if gene.strip()}
                )
            )
            if not genes:
                raise CellMinerDataError(f"Pathway has no genes: {pathway_id}")
            parsed.append((pathway_id, description, genes))
    parsed.sort(key=lambda item: item[0])
    pathway_ids = tuple(item[0] for item in parsed)
    if len(set(pathway_ids)) != len(pathway_ids):
        raise CellMinerDataError("GMT contains duplicate pathway IDs")
    expected = int(verification.get("expected_gene_sets", 0))
    if expected and len(parsed) != expected:
        raise CellMinerDataError(f"Expected {expected} Hallmark sets, found {len(parsed)}")
    canonical = "\n".join(
        "\t".join((pathway_id, description, *genes))
        for pathway_id, description, genes in parsed
    )
    catalog_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return GeneSetCatalog(
        pathway_ids=pathway_ids,
        genes=tuple(item[2] for item in parsed),
        descriptions=tuple(item[1] for item in parsed),
        source_release=str(verification.get("release", "")),
        source_sha256=str(verification["sha256"]),
        catalog_hash=catalog_hash,
    )


def build_pathway_topology(
    catalog: GeneSetCatalog,
    top_k: int = 3,
) -> PathwayTopology:
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    pathways = len(catalog.pathway_ids)
    adjacency = torch.zeros(pathways, pathways, dtype=torch.float32)
    similarities = torch.zeros_like(adjacency)
    gene_sets = [set(genes) for genes in catalog.genes]
    for left in range(pathways):
        for right in range(left + 1, pathways):
            union = gene_sets[left] | gene_sets[right]
            score = len(gene_sets[left] & gene_sets[right]) / len(union) if union else 0.0
            similarities[left, right] = score
            similarities[right, left] = score
    for left in range(pathways):
        ranked = sorted(
            (
                (float(similarities[left, right]), catalog.pathway_ids[right], right)
                for right in range(pathways)
                if right != left and similarities[left, right] > 0
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for score, _, right in ranked[:top_k]:
            adjacency[left, right] = score
    adjacency = torch.maximum(adjacency, adjacency.T)
    adjacency.fill_diagonal_(1.0)
    member_counts = torch.tensor(
        [len(genes) for genes in catalog.genes], dtype=torch.float32
    )
    topology_hash = _compute_topology_hash(
        catalog_hash=catalog.catalog_hash,
        top_k=top_k,
        pathway_ids=catalog.pathway_ids,
        adjacency=adjacency,
        member_counts=member_counts,
    )
    return PathwayTopology(
        pathway_ids=catalog.pathway_ids,
        adjacency=adjacency,
        member_counts=member_counts,
        catalog_hash=catalog.catalog_hash,
        topology_hash=topology_hash,
        top_k=top_k,
    )


def _pathway_membership(
    expression: ExpressionMatrix,
    catalog: GeneSetCatalog,
    topology: PathwayTopology,
    min_pathway_genes: int,
) -> tuple[Tensor, Tensor]:
    if topology.pathway_ids != catalog.pathway_ids:
        raise CellMinerDataError("Gene-set catalog and topology pathway order differs")
    gene_index = {gene_id: index for index, gene_id in enumerate(expression.gene_ids)}
    membership = torch.zeros(
        len(catalog.pathway_ids), len(expression.gene_ids), dtype=torch.bool
    )
    for pathway_index, genes in enumerate(catalog.genes):
        indices = [gene_index[gene] for gene in genes if gene in gene_index]
        if len(indices) < min_pathway_genes:
            raise CellMinerDataError(
                f"{catalog.pathway_ids[pathway_index]} has only {len(indices)} expression genes; "
                f"minimum is {min_pathway_genes}"
            )
        membership[pathway_index, indices] = True
    counts = membership.sum(dim=1).to(dtype=torch.float32)
    return membership, counts


def _gene_z_scores(values: Tensor, observed: Tensor, state: PathwayTransformState) -> Tensor:
    filled = torch.where(observed, values, state.gene_mean.unsqueeze(0))
    scores = (filled - state.gene_mean.unsqueeze(0)) / state.gene_std.unsqueeze(0)
    scores = scores.clamp(-state.clip_value, state.clip_value)
    return torch.where(state.gene_active.unsqueeze(0), scores, torch.zeros_like(scores))


def _raw_pathway_activity(gene_z: Tensor, state: PathwayTransformState) -> Tensor:
    membership = state.membership.to(dtype=gene_z.dtype)
    return (gene_z @ membership.T) / state.available_member_counts.unsqueeze(0)


def _pathway_z_scores(raw_activity: Tensor, state: PathwayTransformState) -> Tensor:
    scores = (
        raw_activity - state.pathway_mean.unsqueeze(0)
    ) / state.pathway_std.unsqueeze(0)
    scores = scores.clamp(-state.clip_value, state.clip_value)
    return torch.where(state.pathway_active.unsqueeze(0), scores, torch.zeros_like(scores))


def _compute_transform_state_hash(state: PathwayTransformState) -> str:
    digest = hashlib.sha256()
    metadata = {
        "version": PATHWAY_TRANSFORM_VERSION,
        "gene_ids": state.gene_ids,
        "pathway_ids": state.pathway_ids,
        "fit_cell_ids": state.fit_cell_ids,
        "split_hash": state.split_hash,
        "expression_hash": state.expression_hash,
        "catalog_hash": state.catalog_hash,
        "topology_hash": state.topology_hash,
        "clip_value": state.clip_value,
        "std_epsilon": state.std_epsilon,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for tensor in (
        state.gene_mean,
        state.gene_std,
        state.gene_fit_counts,
        state.gene_active,
        state.membership,
        state.available_member_counts,
        state.pathway_mean,
        state.pathway_std,
        state.pathway_active,
    ):
        _hash_tensor(digest, tensor)
    return digest.hexdigest()


def fit_train_only_pathway_transform(
    expression: ExpressionMatrix,
    catalog: GeneSetCatalog,
    topology: PathwayTopology,
    train_cell_ids: Collection[str],
    split_hash: str,
    *,
    min_pathway_genes: int = 10,
    clip_value: float = 5.0,
    std_epsilon: float = 1e-6,
) -> PathwayTransformState:
    expression.assert_current_hash()
    topology.assert_current_hash()
    if not isinstance(split_hash, str) or not split_hash.strip():
        raise ValueError("split_hash must be a non-empty string")
    if clip_value <= 0 or std_epsilon <= 0:
        raise ValueError("clip_value and std_epsilon must be positive")
    normalized_train = tuple(sorted({normalize_cell_label(cell) for cell in train_cell_ids}))
    if not normalized_train:
        raise ValueError("At least one train biological cell is required")
    train_indices = expression.cell_indices(normalized_train)
    membership, available_counts = _pathway_membership(
        expression, catalog, topology, min_pathway_genes
    )

    train_values = expression.values[train_indices]
    train_observed = expression.observed_mask[train_indices]
    observed_float = train_observed.to(dtype=train_values.dtype)
    gene_fit_counts = observed_float.sum(dim=0)
    gene_sum = (train_values * observed_float).sum(dim=0)
    gene_mean = gene_sum / gene_fit_counts.clamp_min(1.0)
    centered = torch.where(
        train_observed,
        train_values - gene_mean.unsqueeze(0),
        torch.zeros_like(train_values),
    )
    gene_variance = centered.square().sum(dim=0) / gene_fit_counts.clamp_min(1.0)
    raw_gene_std = gene_variance.sqrt()
    gene_active = (gene_fit_counts >= 2) & (raw_gene_std >= std_epsilon)
    gene_std = raw_gene_std.clamp_min(std_epsilon)

    provisional = PathwayTransformState(
        gene_ids=expression.gene_ids,
        pathway_ids=catalog.pathway_ids,
        fit_cell_ids=normalized_train,
        split_hash=split_hash,
        expression_hash=expression.content_hash,
        catalog_hash=catalog.catalog_hash,
        topology_hash=topology.topology_hash,
        clip_value=float(clip_value),
        std_epsilon=float(std_epsilon),
        gene_mean=gene_mean.float(),
        gene_std=gene_std.float(),
        gene_fit_counts=gene_fit_counts.float(),
        gene_active=gene_active,
        membership=membership,
        available_member_counts=available_counts,
        pathway_mean=torch.zeros(len(catalog.pathway_ids), dtype=torch.float32),
        pathway_std=torch.ones(len(catalog.pathway_ids), dtype=torch.float32),
        pathway_active=torch.ones(len(catalog.pathway_ids), dtype=torch.bool),
        state_hash="",
    )
    train_gene_z = _gene_z_scores(train_values, train_observed, provisional)
    train_pathway_raw = _raw_pathway_activity(train_gene_z, provisional)
    pathway_mean = train_pathway_raw.mean(dim=0)
    pathway_variance = (train_pathway_raw - pathway_mean.unsqueeze(0)).square().mean(dim=0)
    raw_pathway_std = pathway_variance.sqrt()
    pathway_active = (len(train_indices) >= 2) & (raw_pathway_std >= std_epsilon)
    pathway_std = raw_pathway_std.clamp_min(std_epsilon)
    state = replace(
        provisional,
        pathway_mean=pathway_mean.float(),
        pathway_std=pathway_std.float(),
        pathway_active=pathway_active,
    )
    return replace(state, state_hash=_compute_transform_state_hash(state))


def transform_cells_to_graphs(
    expression: ExpressionMatrix,
    topology: PathwayTopology,
    state: PathwayTransformState,
    cell_ids: Collection[str] | None = None,
) -> dict[str, GraphItem]:
    expression.assert_current_hash()
    topology.assert_current_hash()
    state.assert_current_hash()
    if state.expression_hash != expression.content_hash:
        raise CellMinerDataError("Expression content hash differs from transform state")
    if state.gene_ids != expression.gene_ids:
        raise CellMinerDataError("Expression gene order differs from transform state")
    if state.pathway_ids != topology.pathway_ids or state.topology_hash != topology.topology_hash:
        raise CellMinerDataError("Pathway topology differs from transform state")
    selected = tuple(expression.cell_ids if cell_ids is None else cell_ids)
    normalized = tuple(normalize_cell_label(cell) for cell in selected)
    indices = expression.cell_indices(normalized)
    values = expression.values[indices]
    observed = expression.observed_mask[indices]
    gene_z = _gene_z_scores(values, observed, state)
    pathway_raw = _raw_pathway_activity(gene_z, state)
    activity = _pathway_z_scores(pathway_raw, state)
    membership = state.membership.to(dtype=values.dtype)
    observed_fraction = (
        observed.to(dtype=values.dtype) @ membership.T
    ) / topology.member_counts.to(dtype=values.dtype).unsqueeze(0)
    if bool((observed_fraction < -1e-6).any()) or bool(
        (observed_fraction > 1.0 + 1e-6).any()
    ):
        raise CellMinerDataError("Pathway observed fractions must stay within [0, 1]")
    normalized_size = torch.log1p(state.available_member_counts)
    normalized_size = normalized_size / normalized_size.max().clamp_min(1.0)
    identity = torch.eye(len(state.pathway_ids), dtype=torch.float32)

    graphs: dict[str, GraphItem] = {}
    for row, cell_id in enumerate(normalized):
        node_features = torch.cat(
            [
                activity[row].unsqueeze(-1),
                observed_fraction[row].unsqueeze(-1),
                normalized_size.unsqueeze(-1),
                identity,
            ],
            dim=-1,
        ).float()
        if not bool(torch.isfinite(node_features).all()):
            raise CellMinerDataError(f"Non-finite pathway graph features for {cell_id}")
        graphs[cell_id] = GraphItem(
            node_features=node_features,
            adjacency=topology.adjacency.clone(),
        )
    return graphs


def audit_train_only_invariance(
    expression: ExpressionMatrix,
    catalog: GeneSetCatalog,
    topology: PathwayTopology,
    train_cell_ids: Collection[str],
    heldout_cell_ids: Collection[str],
    split_hash: str,
    *,
    extreme_value: float = 1e9,
    min_pathway_genes: int = 10,
    clip_value: float = 5.0,
    std_epsilon: float = 1e-6,
) -> dict[str, Any]:
    heldout = tuple(normalize_cell_label(cell) for cell in heldout_cell_ids)
    if not heldout:
        raise ValueError("At least one held-out cell is required for the leakage audit")
    train = tuple(normalize_cell_label(cell) for cell in train_cell_ids)
    if set(train) & set(heldout):
        raise ValueError("Train and held-out cells overlap")
    state_original = fit_train_only_pathway_transform(
        expression,
        catalog,
        topology,
        train,
        split_hash,
        min_pathway_genes=min_pathway_genes,
        clip_value=clip_value,
        std_epsilon=std_epsilon,
    )
    mutated_values = expression.values.clone()
    heldout_indices = expression.cell_indices(heldout)
    mutated_values[heldout_indices] = float(extreme_value)
    mutated = replace(expression, values=mutated_values, content_hash="")
    state_mutated = fit_train_only_pathway_transform(
        mutated,
        catalog,
        topology,
        train,
        split_hash,
        min_pathway_genes=min_pathway_genes,
        clip_value=clip_value,
        std_epsilon=std_epsilon,
    )
    tensor_fields = (
        "gene_mean",
        "gene_std",
        "gene_fit_counts",
        "gene_active",
        "membership",
        "available_member_counts",
        "pathway_mean",
        "pathway_std",
        "pathway_active",
    )
    state_equal = all(
        torch.equal(getattr(state_original, field), getattr(state_mutated, field))
        for field in tensor_fields
    )
    original_graphs = transform_cells_to_graphs(expression, topology, state_original, train)
    mutated_graphs = transform_cells_to_graphs(mutated, topology, state_mutated, train)
    train_graphs_equal = all(
        torch.equal(original_graphs[cell].node_features, mutated_graphs[cell].node_features)
        and torch.equal(original_graphs[cell].adjacency, mutated_graphs[cell].adjacency)
        for cell in train
    )
    cache_bindings_differ = state_original.state_hash != state_mutated.state_hash
    passed = state_equal and train_graphs_equal and cache_bindings_differ
    if not passed:
        raise AssertionError("Held-out expression changed train-only transform state")
    return {
        "passed": True,
        "train_cells": list(train),
        "heldout_cells": list(heldout),
        "extreme_value": float(extreme_value),
        "state_hash": state_original.state_hash,
        "state_tensors_equal": state_equal,
        "train_graphs_equal": train_graphs_equal,
        "heldout_change_alters_content_bound_cache_key": cache_bindings_differ,
    }


def _atomic_path(destination: Path, suffix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = _atomic_path(path, ".json.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = _atomic_path(path, ".parquet.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = _atomic_path(path, ".csv.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_torch_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = _atomic_path(path, ".pt.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_record(path: Path) -> dict[str, Any]:
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _almanac_cells_source_record(
    source: str | Path | pd.DataFrame,
) -> dict[str, Any]:
    frame = _read_almanac_cells(source).fillna("").copy()
    frame.columns = [str(column) for column in frame.columns]
    frame = frame.astype(str)
    canonical = json.dumps(
        {
            "columns": list(frame.columns),
            "rows": frame.to_numpy(dtype=str).tolist(),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    record: dict[str, Any] = {
        "rows": len(frame),
        "columns": list(frame.columns),
        "table_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    if not isinstance(source, pd.DataFrame):
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        record["file"] = _file_record(source_path)
    return record


def _verify_almanac_cells_source(
    expected: Mapping[str, Any],
    source: str | Path | pd.DataFrame,
) -> dict[str, Any]:
    current = _almanac_cells_source_record(source)
    for field in ("rows", "columns", "table_sha256"):
        if current.get(field) != expected.get(field):
            raise CellMinerDataError(f"Prepared ALMANAC cells source mismatch: {field}")
    expected_file = expected.get("file")
    if expected_file is not None:
        current_file = current.get("file")
        if current_file is None:
            raise CellMinerDataError("Prepared mapping requires its frozen ALMANAC cells file")
        for field in ("bytes", "sha256"):
            if current_file.get(field) != expected_file.get(field):
                raise CellMinerDataError(
                    f"Prepared ALMANAC cells file mismatch: {field}"
                )
    return current


def _verify_file_record(directory: Path, record: Mapping[str, Any]) -> Path:
    path = directory / Path(str(record["filename"])).name
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise CellMinerDataError(f"Cached file size mismatch: {path.name}")
    if sha256_file(path) != str(record["sha256"]):
        raise CellMinerDataError(f"Cached file SHA256 mismatch: {path.name}")
    return path


def cache_pathway_topology(
    config: CellMinerConfig,
    catalog: GeneSetCatalog,
    topology: PathwayTopology,
) -> dict[str, Any]:
    topology.assert_current_hash()
    if catalog.catalog_hash != topology.catalog_hash:
        raise CellMinerDataError("Cannot cache topology with a different catalog")
    expected_hash = _compute_topology_hash(
        catalog_hash=topology.catalog_hash,
        top_k=topology.top_k,
        pathway_ids=topology.pathway_ids,
        adjacency=topology.adjacency,
        member_counts=topology.member_counts,
    )
    if topology.topology_hash != expected_hash:
        raise CellMinerDataError("Cannot cache topology with a stale content hash")
    tensor_path = config.processed_dir / "pathway_topology.pt"
    manifest_path = config.processed_dir / "pathway_manifest.json"
    _write_torch_atomic(
        tensor_path,
        {"adjacency": topology.adjacency, "member_counts": topology.member_counts},
    )
    manifest = {
        "pipeline_version": CELLMINER_PIPELINE_VERSION,
        "topology_version": PATHWAY_TOPOLOGY_VERSION,
        "pathway_ids": list(topology.pathway_ids),
        "descriptions": list(catalog.descriptions),
        "catalog_hash": catalog.catalog_hash,
        "catalog_source_sha256": catalog.source_sha256,
        "catalog_release": catalog.source_release,
        "topology_hash": topology.topology_hash,
        "top_k": topology.top_k,
        "tensor_file": _file_record(tensor_path),
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def load_cached_pathway_topology(
    config: CellMinerConfig,
    expected_catalog_hash: str | None = None,
) -> PathwayTopology:
    manifest = _load_json(config.processed_dir / "pathway_manifest.json")
    if manifest.get("pipeline_version") != CELLMINER_PIPELINE_VERSION:
        raise CellMinerDataError("Cached topology pipeline version mismatch")
    if manifest.get("topology_version") != PATHWAY_TOPOLOGY_VERSION:
        raise CellMinerDataError("Cached topology version mismatch")
    if expected_catalog_hash is not None and manifest["catalog_hash"] != expected_catalog_hash:
        raise CellMinerDataError("Cached topology catalog hash mismatch")
    if int(manifest["top_k"]) != config.pathway_top_k:
        raise CellMinerDataError("Cached topology top_k differs from configuration")
    tensor_path = _verify_file_record(config.processed_dir, manifest["tensor_file"])
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    pathway_ids = tuple(manifest["pathway_ids"])
    adjacency = tensors["adjacency"].float()
    member_counts = tensors["member_counts"].float()
    computed_hash = _compute_topology_hash(
        catalog_hash=str(manifest["catalog_hash"]),
        top_k=int(manifest["top_k"]),
        pathway_ids=pathway_ids,
        adjacency=adjacency,
        member_counts=member_counts,
    )
    if computed_hash != manifest["topology_hash"]:
        raise CellMinerDataError("Cached topology content hash mismatch")
    topology = PathwayTopology(
        pathway_ids=pathway_ids,
        adjacency=adjacency,
        member_counts=member_counts,
        catalog_hash=str(manifest["catalog_hash"]),
        topology_hash=computed_hash,
        top_k=int(manifest["top_k"]),
    )
    return topology


def _transform_tensor_payload(state: PathwayTransformState) -> dict[str, Tensor]:
    return {
        "gene_mean": state.gene_mean,
        "gene_std": state.gene_std,
        "gene_fit_counts": state.gene_fit_counts,
        "gene_active": state.gene_active,
        "membership": state.membership,
        "available_member_counts": state.available_member_counts,
        "pathway_mean": state.pathway_mean,
        "pathway_std": state.pathway_std,
        "pathway_active": state.pathway_active,
    }


def save_pathway_transform_state(
    directory: str | Path,
    state: PathwayTransformState,
) -> dict[str, Any]:
    output = Path(directory)
    tensor_path = output / "transform_state.pt"
    manifest_path = output / "transform_manifest.json"
    expected_hash = _compute_transform_state_hash(replace(state, state_hash=""))
    if state.state_hash != expected_hash:
        raise CellMinerDataError("Transform state hash is stale")
    _write_torch_atomic(tensor_path, _transform_tensor_payload(state))
    manifest = {
        "transform_version": PATHWAY_TRANSFORM_VERSION,
        "gene_ids": list(state.gene_ids),
        "pathway_ids": list(state.pathway_ids),
        "fit_cell_ids": list(state.fit_cell_ids),
        "split_hash": state.split_hash,
        "expression_hash": state.expression_hash,
        "catalog_hash": state.catalog_hash,
        "topology_hash": state.topology_hash,
        "clip_value": state.clip_value,
        "std_epsilon": state.std_epsilon,
        "state_hash": state.state_hash,
        "tensor_file": _file_record(tensor_path),
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def load_pathway_transform_state(
    directory: str | Path,
    *,
    expected_split_hash: str | None = None,
    expected_fit_cell_ids: Collection[str] | None = None,
) -> PathwayTransformState:
    output = Path(directory)
    manifest = _load_json(output / "transform_manifest.json")
    if manifest.get("transform_version") != PATHWAY_TRANSFORM_VERSION:
        raise CellMinerDataError("Transform version mismatch")
    if expected_split_hash is not None and manifest["split_hash"] != expected_split_hash:
        raise CellMinerDataError("Transform split hash mismatch")
    if expected_fit_cell_ids is not None:
        expected = sorted({normalize_cell_label(cell) for cell in expected_fit_cell_ids})
        if manifest["fit_cell_ids"] != expected:
            raise CellMinerDataError("Transform fit-cell set mismatch")
    tensor_path = _verify_file_record(output, manifest["tensor_file"])
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    state = PathwayTransformState(
        gene_ids=tuple(manifest["gene_ids"]),
        pathway_ids=tuple(manifest["pathway_ids"]),
        fit_cell_ids=tuple(manifest["fit_cell_ids"]),
        split_hash=str(manifest["split_hash"]),
        expression_hash=str(manifest["expression_hash"]),
        catalog_hash=str(manifest["catalog_hash"]),
        topology_hash=str(manifest["topology_hash"]),
        clip_value=float(manifest["clip_value"]),
        std_epsilon=float(manifest["std_epsilon"]),
        gene_mean=tensors["gene_mean"].float(),
        gene_std=tensors["gene_std"].float(),
        gene_fit_counts=tensors["gene_fit_counts"].float(),
        gene_active=tensors["gene_active"].bool(),
        membership=tensors["membership"].bool(),
        available_member_counts=tensors["available_member_counts"].float(),
        pathway_mean=tensors["pathway_mean"].float(),
        pathway_std=tensors["pathway_std"].float(),
        pathway_active=tensors["pathway_active"].bool(),
        state_hash=str(manifest["state_hash"]),
    )
    if _compute_transform_state_hash(replace(state, state_hash="")) != state.state_hash:
        raise CellMinerDataError("Loaded transform state hash mismatch")
    return state


def prepare_cellminer_assets(
    config: CellMinerConfig,
    almanac_cells: str | Path | pd.DataFrame,
) -> dict[str, Any]:
    cellminer_verification = verify_cellminer_source(config)
    hallmark_verification = verify_hallmark_source(config)
    frame, expression = parse_cellminer_expression(config, cellminer_verification)
    mapping = build_strict_cell_mapping(
        almanac_cells, expression.cell_ids, config.cell_aliases
    )
    catalog = load_hallmark_gene_sets(config, hallmark_verification)
    topology = build_pathway_topology(catalog, config.pathway_top_k)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = config.processed_dir / "expression.parquet"
    tensor_path = config.processed_dir / "expression_tensors.pt"
    mapping_path = config.processed_dir / "cell_mapping.csv"
    _write_parquet_atomic(parquet_path, frame)
    _write_torch_atomic(
        tensor_path,
        {"values": expression.values, "observed_mask": expression.observed_mask},
    )
    _write_csv_atomic(mapping_path, mapping)
    mapping_counts = {
        str(key): int(value) for key, value in mapping["status"].value_counts().items()
    }
    expression_manifest = {
        "pipeline_version": CELLMINER_PIPELINE_VERSION,
        "source": cellminer_verification,
        "source_release": expression.source_release,
        "source_sha256": expression.source_sha256,
        "content_hash": expression.content_hash,
        "value_scale": expression.value_scale,
        "gene_rows": len(expression.gene_ids),
        "expression_cells": len(expression.cell_ids),
        "gene_ids": list(expression.gene_ids),
        "entrez_ids": list(expression.entrez_ids),
        "cell_ids": list(expression.cell_ids),
        "observed_values": int(expression.observed_mask.sum()),
        "missing_values": int((~expression.observed_mask).sum()),
        "minimum": float(expression.values[expression.observed_mask].min()),
        "maximum": float(expression.values[expression.observed_mask].max()),
        "parquet_file": _file_record(parquet_path),
        "tensor_file": _file_record(tensor_path),
        "mapping_file": _file_record(mapping_path),
        "mapping_counts": mapping_counts,
        "alias_mapping_sha256": sha256_file(config.cell_aliases),
        "almanac_cells_source": _almanac_cells_source_record(almanac_cells),
    }
    _write_json_atomic(config.processed_dir / "expression_manifest.json", expression_manifest)
    pathway_manifest = cache_pathway_topology(config, catalog, topology)
    audit = {
        "pipeline_version": CELLMINER_PIPELINE_VERSION,
        "expression_shape": [len(expression.gene_ids), len(expression.cell_ids)],
        "mapping_counts": mapping_counts,
        "unique_mapped_biological_cells": int(
            mapping.loc[mapping["status"] == "mapped", "biological_cell_id"].nunique()
        ),
        "pathway_nodes": len(catalog.pathway_ids),
        "pathway_feature_dim": len(catalog.pathway_ids) + 3,
        "topology_hash": topology.topology_hash,
        "cellminer_source_verified": True,
        "hallmark_source_verified": True,
    }
    _write_json_atomic(config.processed_dir / "asset_audit.json", audit)
    return {
        "expression": expression,
        "mapping": mapping,
        "catalog": catalog,
        "topology": topology,
        "expression_manifest": expression_manifest,
        "pathway_manifest": pathway_manifest,
        "audit": audit,
    }


def load_prepared_expression(
    config: CellMinerConfig,
    *,
    source_verification: Mapping[str, Any] | None = None,
) -> ExpressionMatrix:
    manifest = _load_json(config.processed_dir / "expression_manifest.json")
    if manifest.get("pipeline_version") != CELLMINER_PIPELINE_VERSION:
        raise CellMinerDataError("Prepared expression pipeline version mismatch")
    current_source = dict(source_verification or verify_cellminer_source(config))
    if manifest.get("source_sha256") != current_source.get("sha256"):
        raise CellMinerDataError("Prepared expression is bound to another CellMiner source")
    if manifest.get("source_release") != current_source.get("release"):
        raise CellMinerDataError("Prepared expression release differs from current source")
    if manifest.get("value_scale") != current_source.get("value_scale"):
        raise CellMinerDataError("Prepared expression scale differs from current source")
    if int(manifest.get("gene_rows", -1)) != config.expected_gene_rows:
        raise CellMinerDataError("Prepared expression gene count differs from configuration")
    if int(manifest.get("expression_cells", -1)) != config.expected_expression_cells:
        raise CellMinerDataError("Prepared expression cell count differs from configuration")
    tensor_path = _verify_file_record(config.processed_dir, manifest["tensor_file"])
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    return ExpressionMatrix(
        cell_ids=tuple(manifest["cell_ids"]),
        gene_ids=tuple(manifest["gene_ids"]),
        entrez_ids=tuple(manifest["entrez_ids"]),
        values=tensors["values"].float(),
        observed_mask=tensors["observed_mask"].bool(),
        source_release=str(manifest["source_release"]),
        source_sha256=str(manifest["source_sha256"]),
        value_scale=str(manifest["value_scale"]),
        content_hash=str(manifest["content_hash"]),
    )


def load_prepared_cell_mapping(
    config: CellMinerConfig,
    almanac_cells: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    manifest = _load_json(config.processed_dir / "expression_manifest.json")
    if manifest.get("pipeline_version") != CELLMINER_PIPELINE_VERSION:
        raise CellMinerDataError("Prepared mapping pipeline version mismatch")
    current_alias_sha256 = sha256_file(config.cell_aliases)
    if manifest.get("alias_mapping_sha256") != current_alias_sha256:
        raise CellMinerDataError("Prepared mapping is bound to another alias rule file")
    expected_cells_source = manifest.get("almanac_cells_source")
    if not isinstance(expected_cells_source, dict):
        raise CellMinerDataError("Prepared mapping has no ALMANAC cells source binding")
    _verify_almanac_cells_source(expected_cells_source, almanac_cells)
    mapping_path = _verify_file_record(config.processed_dir, manifest["mapping_file"])
    mapping = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
    required = {
        "source_cell_id",
        "source_alias_count",
        "expression_cell_id",
        "biological_cell_id",
        "status",
    }
    if not required.issubset(mapping.columns):
        raise CellMinerDataError(
            f"Prepared cell mapping is missing: {sorted(required - set(mapping.columns))}"
        )
    unexpected_statuses = sorted(
        set(mapping["status"]) - {"mapped", "excluded", "expression_only"}
    )
    if unexpected_statuses:
        raise CellMinerDataError(
            f"Prepared cell mapping has invalid statuses: {unexpected_statuses}"
        )
    actual_counts = {
        str(key): int(value) for key, value in mapping["status"].value_counts().items()
    }
    if actual_counts != manifest.get("mapping_counts"):
        raise CellMinerDataError("Prepared cell mapping counts differ from manifest")
    return mapping


def audit_prepared_cellminer(
    config: CellMinerConfig,
    almanac_cells: str | Path | pd.DataFrame,
) -> dict[str, Any]:
    """Verify cached expression, crosswalk, and fixed Hallmark topology."""

    cellminer_source = verify_cellminer_source(config)
    hallmark_source = verify_hallmark_source(config)
    expression_manifest = _load_json(config.processed_dir / "expression_manifest.json")
    _verify_file_record(config.processed_dir, expression_manifest["parquet_file"])
    expression = load_prepared_expression(
        config, source_verification=cellminer_source
    )
    mapping = load_prepared_cell_mapping(config, almanac_cells)
    catalog = load_hallmark_gene_sets(config, hallmark_source)
    topology = load_cached_pathway_topology(
        config, expected_catalog_hash=catalog.catalog_hash
    )
    expected_topology = build_pathway_topology(catalog, config.pathway_top_k)
    source_rows = mapping[mapping["status"] != "expression_only"]
    mapped = mapping[mapping["status"] == "mapped"]
    excluded = mapping[mapping["status"] == "excluded"]
    expression_only = mapping[mapping["status"] == "expression_only"]
    counts = {
        str(key): int(value) for key, value in mapping["status"].value_counts().items()
    }
    raw_almanac_labels: list[str] = []
    for encoded in source_rows["source_labels"]:
        decoded = json.loads(str(encoded))
        if not isinstance(decoded, list):
            raise CellMinerDataError("Prepared source_labels must contain JSON lists")
        raw_almanac_labels.extend(str(label) for label in decoded)
    checks = {
        "expression_shape_matches_contract": tuple(expression.values.shape)
        == (config.expected_expression_cells, config.expected_gene_rows),
        "expression_is_finite": bool(torch.isfinite(expression.values).all()),
        "expression_scale_is_frozen": expression.value_scale == "log2(FPKM + 1)",
        "expression_is_bound_to_current_cellminer_source": expression.source_sha256
        == cellminer_source["sha256"],
        "expression_tensor_content_hash_matches_manifest": expression.content_hash
        == expression_manifest["content_hash"],
        "mapping_is_bound_to_current_alias_rules": expression_manifest[
            "alias_mapping_sha256"
        ]
        == sha256_file(config.cell_aliases),
        "mapping_is_bound_to_current_almanac_cells": _almanac_cells_source_record(
            almanac_cells
        )["table_sha256"]
        == expression_manifest["almanac_cells_source"]["table_sha256"],
        "canonical_almanac_cells_are_60": len(source_rows) == 60,
        "raw_almanac_source_labels_are_61_and_unique": len(raw_almanac_labels) == 61
        and len(set(raw_almanac_labels)) == 61,
        "mapped_intersection_is_59": len(mapped) == 59
        and mapped["biological_cell_id"].nunique() == 59,
        "one_cell_is_explicitly_excluded": len(excluded) == 1
        and set(excluded["biological_cell_id"]) == {"BR:MDA-MB-468"},
        "mda_n_is_expression_only": len(expression_only) == 1
        and set(expression_only["biological_cell_id"]) == {"ME:MDA-N"},
        "mapped_expression_columns_exist": set(mapped["expression_cell_id"])
        <= set(expression.cell_ids),
        "pathway_catalog_has_50_nodes": len(catalog.pathway_ids) == 50,
        "pathway_topology_is_symmetric": torch.equal(
            topology.adjacency, topology.adjacency.T
        ),
        "pathway_topology_has_unit_self_loops": torch.equal(
            torch.diag(topology.adjacency), torch.ones(len(topology.pathway_ids))
        ),
        "pathway_topology_matches_frozen_catalog": topology.topology_hash
        == expected_topology.topology_hash
        and torch.equal(topology.adjacency, expected_topology.adjacency)
        and torch.equal(topology.member_counts, expected_topology.member_counts),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "cellminer_source_sha256": cellminer_source["sha256"],
        "hallmark_source_sha256": hallmark_source["sha256"],
        "expression_cells": len(expression.cell_ids),
        "expression_genes": len(expression.gene_ids),
        "expression_content_hash": expression.content_hash,
        "expression_min": float(expression.values.min()),
        "expression_max": float(expression.values.max()),
        "mapping_counts": counts,
        "almanac_source_label_count": len(raw_almanac_labels),
        "unique_almanac_source_label_count": len(set(raw_almanac_labels)),
        "mapped_biological_cells": int(mapped["biological_cell_id"].nunique()),
        "pathway_nodes": len(topology.pathway_ids),
        "pathway_feature_dim": len(topology.pathway_ids) + 3,
        "undirected_nonself_edges": int(
            ((topology.adjacency > 0).sum().item() - len(topology.pathway_ids)) // 2
        ),
        "topology_hash": topology.topology_hash,
    }
    _write_json_atomic(config.processed_dir / "prepared_audit.json", report)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Prepared CellMiner audit failed: {failed}")
    return report


def _split_cache_key(split_hash: str) -> str:
    if not isinstance(split_hash, str) or not split_hash:
        raise ValueError("split_hash must be non-empty")
    return hashlib.sha256(split_hash.encode("utf-8")).hexdigest()[:20]


def cache_split_cell_graphs(
    config: CellMinerConfig,
    expression: ExpressionMatrix,
    topology: PathwayTopology,
    state: PathwayTransformState,
    cell_ids: Collection[str] | None = None,
    *,
    expected_split_hash: str,
) -> dict[str, Any]:
    expression.assert_current_hash()
    topology.assert_current_hash()
    state.assert_current_hash()
    if state.split_hash != expected_split_hash:
        raise CellMinerDataError("Refusing to cache graphs for a different split hash")
    if topology.top_k != config.pathway_top_k:
        raise CellMinerDataError("Refusing to cache topology from another configuration")
    if state.catalog_hash != topology.catalog_hash:
        raise CellMinerDataError("Refusing to cache transform from another pathway catalog")
    if state.clip_value != config.clip_value or state.std_epsilon != config.std_epsilon:
        raise CellMinerDataError("Refusing to cache transform from another configuration")
    graphs = transform_cells_to_graphs(expression, topology, state, cell_ids)
    cache_key = _split_cache_key(state.split_hash)
    output = config.processed_dir / "folds" / cache_key
    transform_manifest = save_pathway_transform_state(output, state)
    graph_path = output / "cell_graphs.pt"
    _write_torch_atomic(
        graph_path,
        {
            cell_id: {
                "node_features": graph.node_features,
                "adjacency": graph.adjacency,
            }
            for cell_id, graph in graphs.items()
        },
    )
    observed_fractions = torch.stack(
        [graph.node_features[:, 1] for graph in graphs.values()]
    )
    manifest = {
        "pipeline_version": CELLMINER_PIPELINE_VERSION,
        "split_hash": state.split_hash,
        "split_cache_key": cache_key,
        "fit_cell_ids": list(state.fit_cell_ids),
        "state_hash": state.state_hash,
        "expression_hash": state.expression_hash,
        "catalog_hash": state.catalog_hash,
        "topology_hash": state.topology_hash,
        "cell_ids": sorted(graphs),
        "pathway_nodes": len(state.pathway_ids),
        "node_feature_dim": len(state.pathway_ids) + 3,
        "min_pathway_genes": config.min_pathway_genes,
        "graph_file": _file_record(graph_path),
        "transform_manifest": transform_manifest,
    }
    _write_json_atomic(output / "cell_graph_manifest.json", manifest)
    audit = {
        "split_hash_verified": True,
        "graphs": len(graphs),
        "all_finite": all(
            bool(torch.isfinite(graph.node_features).all()) for graph in graphs.values()
        ),
        "shared_topology": all(
            torch.equal(graph.adjacency, topology.adjacency) for graph in graphs.values()
        ),
        "observed_fraction_min": float(observed_fractions.min()),
        "observed_fraction_max": float(observed_fractions.max()),
    }
    _write_json_atomic(output / "audit.json", audit)
    return {"directory": output, "manifest": manifest, "audit": audit, "graphs": graphs}


def load_cached_split_cell_graphs(
    config: CellMinerConfig,
    split_hash: str,
    *,
    expression: ExpressionMatrix,
    topology: PathwayTopology,
    expected_fit_cell_ids: Collection[str],
) -> tuple[PathwayTransformState, dict[str, GraphItem], dict[str, Any]]:
    expression.assert_current_hash()
    topology.assert_current_hash()
    output = config.processed_dir / "folds" / _split_cache_key(split_hash)
    manifest = _load_json(output / "cell_graph_manifest.json")
    if manifest.get("pipeline_version") != CELLMINER_PIPELINE_VERSION:
        raise CellMinerDataError("Cached graph pipeline version mismatch")
    if manifest["split_hash"] != split_hash:
        raise CellMinerDataError("Cached cell graphs have the wrong split hash")
    expected_context = {
        "expression_hash": expression.content_hash,
        "catalog_hash": topology.catalog_hash,
        "topology_hash": topology.topology_hash,
        "pathway_nodes": len(topology.pathway_ids),
        "node_feature_dim": len(topology.pathway_ids) + 3,
        "min_pathway_genes": config.min_pathway_genes,
    }
    mismatched = [
        field
        for field, expected in expected_context.items()
        if manifest.get(field) != expected
    ]
    if mismatched:
        raise CellMinerDataError(
            f"Cached cell graph context mismatch: {sorted(mismatched)}"
        )
    if topology.top_k != config.pathway_top_k:
        raise CellMinerDataError("Current topology top_k differs from configuration")
    state = load_pathway_transform_state(
        output,
        expected_split_hash=split_hash,
        expected_fit_cell_ids=expected_fit_cell_ids,
    )
    if state.expression_hash != expression.content_hash:
        raise CellMinerDataError("Cached transform expression content hash mismatch")
    if state.gene_ids != expression.gene_ids:
        raise CellMinerDataError("Cached transform gene order mismatch")
    if state.catalog_hash != topology.catalog_hash:
        raise CellMinerDataError("Cached transform pathway catalog mismatch")
    if state.pathway_ids != topology.pathway_ids:
        raise CellMinerDataError("Cached transform pathway order mismatch")
    if state.topology_hash != topology.topology_hash:
        raise CellMinerDataError("Cached transform topology mismatch")
    if state.clip_value != config.clip_value or state.std_epsilon != config.std_epsilon:
        raise CellMinerDataError("Cached transform configuration mismatch")
    if state.state_hash != manifest["state_hash"]:
        raise CellMinerDataError("Cached graph/state hash mismatch")
    if manifest.get("fit_cell_ids") != list(state.fit_cell_ids):
        raise CellMinerDataError("Cached graph manifest fit-cell set mismatch")
    transform_manifest = _load_json(output / "transform_manifest.json")
    if manifest.get("transform_manifest") != transform_manifest:
        raise CellMinerDataError("Embedded transform manifest differs from cached state")
    graph_path = _verify_file_record(output, manifest["graph_file"])
    tensors = torch.load(graph_path, map_location="cpu", weights_only=True)
    graphs = {
        str(cell_id): GraphItem(
            node_features=payload["node_features"].float(),
            adjacency=payload["adjacency"].float(),
        )
        for cell_id, payload in tensors.items()
    }
    if sorted(graphs) != sorted(manifest["cell_ids"]):
        raise CellMinerDataError("Cached graph cell IDs differ from manifest")
    unknown_cells = sorted(set(graphs) - set(expression.cell_ids))
    if unknown_cells:
        raise CellMinerDataError(f"Cached graphs contain unknown expression cells: {unknown_cells}")
    expected_shape = (len(topology.pathway_ids), len(topology.pathway_ids) + 3)
    for cell_id, graph in graphs.items():
        if tuple(graph.node_features.shape) != expected_shape:
            raise CellMinerDataError(f"Cached graph feature shape mismatch for {cell_id}")
        if not bool(torch.isfinite(graph.node_features).all()):
            raise CellMinerDataError(f"Cached graph features are non-finite for {cell_id}")
        if not torch.equal(graph.adjacency, topology.adjacency):
            raise CellMinerDataError(f"Cached graph topology mismatch for {cell_id}")
    return state, graphs, manifest


__all__ = [
    "CELLMINER_PIPELINE_VERSION",
    "PATHWAY_TOPOLOGY_VERSION",
    "PATHWAY_TRANSFORM_VERSION",
    "CellMinerConfig",
    "CellMinerDataError",
    "ExpressionMatrix",
    "GeneSetCatalog",
    "PathwayTopology",
    "PathwayTransformState",
    "audit_train_only_invariance",
    "audit_prepared_cellminer",
    "build_pathway_topology",
    "build_strict_cell_mapping",
    "cache_pathway_topology",
    "cache_split_cell_graphs",
    "default_cellminer_config",
    "download_cellminer_sources",
    "fit_train_only_pathway_transform",
    "load_cached_pathway_topology",
    "load_cached_split_cell_graphs",
    "load_cell_alias_rules",
    "load_cellminer_config",
    "load_hallmark_gene_sets",
    "load_pathway_transform_state",
    "load_prepared_cell_mapping",
    "load_prepared_expression",
    "normalize_cell_label",
    "normalize_gene_symbol",
    "parse_cellminer_expression",
    "prepare_cellminer_assets",
    "save_pathway_transform_state",
    "sha256_file",
    "transform_cells_to_graphs",
    "verify_cellminer_source",
    "verify_hallmark_source",
]

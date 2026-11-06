from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from rdkit import Chem, rdBase

from .chemistry import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    StandardizationPolicy,
    StructureStandardizationError,
    smiles_to_graph,
    structure_cache_key,
)


EXPECTED_COLUMNS = (
    "COMBODRUGSEQ",
    "SCREENER",
    "STUDY",
    "TESTDATE",
    "PLATE",
    "PANELNBR",
    "CELLNBR",
    "PREFIX1",
    "NSC1",
    "SAMPLE1",
    "CONCINDEX1",
    "CONC1",
    "CONCUNIT1",
    "PREFIX2",
    "NSC2",
    "SAMPLE2",
    "CONCINDEX2",
    "CONC2",
    "CONCUNIT2",
    "PERCENTGROWTH",
    "PERCENTGROWTHNOTZ",
    "TESTVALUE",
    "CONTROLVALUE",
    "TZVALUE",
    "EXPECTEDGROWTH",
    "SCORE",
    "VALID",
    "PANEL",
    "CELLNAME",
)


@dataclass(frozen=True)
class ResponseTransform:
    """Scalar conversion of the source endpoint into the model target.

    ``clip_min`` and ``clip_max`` are the declared reference bounds of the
    endpoint. ``apply_clip`` decides whether the target is truncated to them.
    When it is false the target keeps its unclipped value, while
    ``response_clipped_low`` and ``response_clipped_high`` still mark the points
    that fall outside the reference bounds, so clipping-state stratification
    stays available. See ``docs/ENDPOINT_DECISION.md`` for why D1 turns the
    truncation off.
    """

    field: str = "PERCENTGROWTHNOTZ"
    divide_by: float = 100.0
    clip_min: float = 0.0
    clip_max: float = 1.0
    apply_clip: bool = True


@dataclass(frozen=True)
class DoseTransform:
    accepted_units: tuple[str, ...] = ("M",)
    model_reference_molar: float = 1e-12


@dataclass(frozen=True)
class NCIAlmanacConfig:
    dataset_release: str
    raw_dir: Path
    processed_dir: Path
    source_manifest: Path
    chunk_size: int
    response: ResponseTransform
    dose: DoseTransform
    structure: StandardizationPolicy
    real_smoke: dict[str, Any]


def load_nci_almanac_config(path: str | Path) -> NCIAlmanacConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("NCI-ALMANAC configuration must be a mapping")
    base = config_path.parent.parent

    def resolve_project_path(value: str) -> Path:
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

    dose_raw = dict(raw.get("dose", {}))
    dose_raw["accepted_units"] = tuple(str(unit) for unit in dose_raw["accepted_units"])
    config = NCIAlmanacConfig(
        dataset_release=str(raw["dataset_release"]),
        raw_dir=resolve_project_path(str(raw["raw_dir"])),
        processed_dir=resolve_project_path(str(raw["processed_dir"])),
        source_manifest=resolve_project_path(str(raw["source_manifest"])),
        chunk_size=int(raw.get("chunk_size", 250_000)),
        response=ResponseTransform(**dict(raw.get("response", {}))),
        dose=DoseTransform(**dose_raw),
        structure=StandardizationPolicy(**dict(raw.get("structure", {}))),
        real_smoke=dict(raw.get("real_smoke", {})),
    )
    if config.response.field not in EXPECTED_COLUMNS:
        raise ValueError(f"Unknown response field: {config.response.field}")
    if config.response.divide_by <= 0 or config.response.clip_max <= config.response.clip_min:
        raise ValueError("Invalid response transform")
    if config.dose.model_reference_molar <= 0:
        raise ValueError("dose.model_reference_molar must be positive")
    if config.chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return config


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _load_source_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("files"), list):
        raise ValueError("Source manifest has no files list")
    return manifest


def verify_source_files(config: NCIAlmanacConfig) -> dict[str, Any]:
    manifest = _load_source_manifest(config.source_manifest)
    verified: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        path = (config.raw_dir / Path(entry["filename"]).name).resolve()
        if path.parent != config.raw_dir.resolve():
            raise ValueError("Manifest filename escaped the configured raw directory")
        if not path.is_file():
            if entry.get("required", True):
                raise FileNotFoundError(path)
            continue
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != int(entry["bytes"]):
            raise ValueError(f"Size mismatch for {path.name}: {actual_bytes}")
        if actual_sha256.lower() != str(entry["sha256"]).lower():
            raise ValueError(f"SHA256 mismatch for {path.name}")
        item = {
            "filename": path.name,
            "bytes": actual_bytes,
            "sha256": actual_sha256,
        }
        if "zip_member" in entry:
            with zipfile.ZipFile(path) as archive:
                if archive.namelist() != [entry["zip_member"]]:
                    raise ValueError(f"Unexpected members in {path.name}: {archive.namelist()}")
                info = archive.getinfo(entry["zip_member"])
                if info.file_size != int(entry["zip_member_bytes"]):
                    raise ValueError("Uncompressed NCI CSV size mismatch")
                if f"{info.CRC:08x}" != str(entry["zip_member_crc32"]).lower():
                    raise ValueError("NCI CSV CRC32 mismatch")
                item["zip_member"] = info.filename
                item["zip_member_bytes"] = info.file_size
                item["zip_member_crc32"] = f"{info.CRC:08x}"
        verified.append(item)
    return {"dataset": manifest["dataset"], "release": manifest["release"], "files": verified}


def _download_one(entry: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    urls = [entry["url"]]
    if entry.get("fallback_url"):
        urls.append(entry["fallback_url"])
    errors: list[str] = []
    for url in urls:
        if partial.exists():
            partial.unlink()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "BiSurfGNN/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if partial.stat().st_size != int(entry["bytes"]):
                raise ValueError("downloaded byte count differs from manifest")
            if sha256_file(partial).lower() != str(entry["sha256"]).lower():
                raise ValueError("downloaded SHA256 differs from manifest")
            partial.replace(destination)
            return
        except Exception as error:
            errors.append(f"{url}: {type(error).__name__}: {error}")
    if partial.exists():
        partial.unlink()
    raise RuntimeError("All download sources failed: " + " | ".join(errors))


def download_source_files(config: NCIAlmanacConfig) -> dict[str, Any]:
    manifest = _load_source_manifest(config.source_manifest)
    for entry in manifest["files"]:
        destination = config.raw_dir / Path(entry["filename"]).name
        if destination.exists():
            if (
                destination.stat().st_size == int(entry["bytes"])
                and sha256_file(destination).lower() == str(entry["sha256"]).lower()
            ):
                continue
            raise ValueError(f"Existing source file does not match manifest: {destination}")
        _download_one(entry, destination)
    return verify_source_files(config)


def _read_names(path: Path) -> dict[int, list[str]]:
    names: dict[int, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            try:
                nsc = int(parts[0])
            except ValueError:
                continue
            name = parts[1].strip()
            if name and name not in names[nsc]:
                names[nsc].append(name)
    return dict(names)


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def build_drug_structure_cache(
    used_nscs: set[int],
    config: NCIAlmanacConfig,
) -> dict[str, Any]:
    names_by_nsc = _read_names(config.raw_dir / "ComboCompoundNames_all.txt")
    sdf_candidates: dict[int, str] = {}
    sdf_supplier = Chem.SDMolSupplier(
        str(config.raw_dir / "ComboCompoundSet.sdf"), removeHs=True, sanitize=True
    )
    invalid_sdf_records = 0
    for molecule in sdf_supplier:
        if molecule is None:
            invalid_sdf_records += 1
            continue
        nsc_value = (
            molecule.GetProp("NSC")
            if molecule.HasProp("NSC")
            else molecule.GetProp("_Name")
        )
        sdf_candidates[int(nsc_value)] = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )

    fallback_frame = pd.read_csv(
        config.raw_dir / "nsc_smiles.csv",
        usecols=["NSC", "SMILES"],
        dtype={"NSC": "Int64", "SMILES": "string"},
    )
    fallback_frame = fallback_frame[fallback_frame["NSC"].isin(sorted(used_nscs))]
    fallback_candidates = {
        int(row.NSC): str(row.SMILES)
        for row in fallback_frame.itertuples(index=False)
        if pd.notna(row.NSC) and pd.notna(row.SMILES) and str(row.SMILES).strip()
    }

    # The official cisplatin SMILES uses ordinary bonds to chloride and cannot be
    # sanitized by RDKit. This explicit dative-bond rewrite is kept in the audit.
    structure_overrides = {
        119875: "[NH3]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-]",
    }
    graph_cache: dict[str, dict[str, torch.Tensor]] = {}
    metadata_by_key: dict[str, dict[str, Any]] = {}
    mapping: dict[int, str] = {}
    source_by_nsc: dict[int, str] = {}
    rejected: dict[int, str] = {}

    for nsc in sorted(used_nscs):
        if nsc in structure_overrides:
            candidates = [("explicit_audited_override", structure_overrides[nsc])]
        else:
            candidates = []
            if nsc in sdf_candidates:
                candidates.append(("ComboCompoundSet.sdf", sdf_candidates[nsc]))
            if nsc in fallback_candidates:
                candidates.append(("nsc_smiles.csv", fallback_candidates[nsc]))
        if not candidates:
            rejected[nsc] = "no_structure_candidate"
            continue
        errors: list[str] = []
        for source, smiles in candidates:
            try:
                structure, graph = smiles_to_graph(smiles, config.structure)
                cache_key = structure_cache_key(structure, config.structure)
                graph_cache.setdefault(cache_key, graph.to_tensor_dict())
                metadata_by_key.setdefault(
                    cache_key,
                    {
                        **asdict(structure),
                        "source_nsc": nsc,
                        "structure_source": source,
                    },
                )
                mapping[nsc] = cache_key
                source_by_nsc[nsc] = source
                rejected.pop(nsc, None)
                break
            except StructureStandardizationError as error:
                errors.append(f"{source}: {error}")
        if nsc not in mapping:
            rejected[nsc] = " | ".join(errors) or "standardization_failed"

    name_to_structured_nscs: dict[str, set[int]] = defaultdict(set)
    for nsc in mapping:
        for name in names_by_nsc.get(nsc, []):
            normalized = _normalized_name(name)
            if normalized:
                name_to_structured_nscs[normalized].add(nsc)
    aliases: dict[int, int] = {}
    for nsc in sorted(set(used_nscs) - set(mapping)):
        candidates: set[int] = set()
        for name in names_by_nsc.get(nsc, []):
            candidates.update(name_to_structured_nscs.get(_normalized_name(name), set()))
        candidate_keys = {mapping[candidate] for candidate in candidates if candidate in mapping}
        if len(candidate_keys) == 1:
            alias_of = min(
                candidate
                for candidate in candidates
                if mapping[candidate] in candidate_keys
            )
            mapping[nsc] = mapping[alias_of]
            source_by_nsc[nsc] = "name_alias"
            aliases[nsc] = alias_of
            rejected.pop(nsc, None)
        elif candidates:
            rejected[nsc] = f"ambiguous_name_alias:{sorted(candidates)}"

    drug_rows: list[dict[str, Any]] = []
    for nsc in sorted(used_nscs):
        cache_key = mapping.get(nsc)
        if cache_key is None:
            continue
        metadata = metadata_by_key[cache_key]
        connectivity = metadata["connectivity_key"] or (
            "SMILES-" + hashlib.sha256(metadata["canonical_smiles"].encode()).hexdigest()[:14]
        )
        drug_rows.append(
            {
                "drug_id": f"S{nsc}",
                "nsc": nsc,
                "preferred_name": names_by_nsc.get(nsc, [""])[0],
                "all_names": json.dumps(names_by_nsc.get(nsc, []), ensure_ascii=False),
                "canonical_smiles": metadata["canonical_smiles"],
                "raw_smiles": metadata["raw_smiles"],
                "inchikey": metadata["inchikey"],
                "neutral_identity_inchikey": metadata["neutral_identity_inchikey"],
                "identity_group": connectivity,
                "murcko_scaffold": metadata["murcko_scaffold"],
                "structure_key": cache_key,
                "structure_source": source_by_nsc[nsc],
                "structure_alias_of": f"S{aliases[nsc]}" if nsc in aliases else "",
                "removed_fragments": json.dumps(metadata["removed_fragments"]),
                "structure_flags": json.dumps(metadata["flags"]),
            }
        )

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graph_cache, config.processed_dir / "structures.pt")
    pd.DataFrame(drug_rows).to_csv(
        config.processed_dir / "drugs.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"drug_id": f"S{nsc}", "nsc": nsc, "reason": reason}
            for nsc, reason in sorted(rejected.items())
        ],
        columns=["drug_id", "nsc", "reason"],
    ).to_csv(config.processed_dir / "rejected_structures.csv", index=False, encoding="utf-8")
    with (config.processed_dir / "drug_id_to_structure_key.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({f"S{nsc}": key for nsc, key in sorted(mapping.items())}, handle, indent=2)
    with (config.processed_dir / "structures.jsonl").open("w", encoding="utf-8") as handle:
        for cache_key, metadata in sorted(metadata_by_key.items()):
            handle.write(json.dumps({"structure_key": cache_key, **metadata}) + "\n")
    structure_report = {
        "rdkit_version": rdBase.rdkitVersion,
        "policy": asdict(config.structure),
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "used_nscs": len(used_nscs),
        "mapped_nscs": len(mapping),
        "unique_structures": len(graph_cache),
        "rejected_nscs": len(rejected),
        "invalid_sdf_records": invalid_sdf_records,
        "explicit_overrides": {f"S{key}": value for key, value in structure_overrides.items()},
        "name_aliases": {f"S{key}": f"S{value}" for key, value in aliases.items()},
    }
    with (config.processed_dir / "structure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(structure_report, handle, indent=2)
    return structure_report


def _clean_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _clean_cell_name(value: Any) -> str:
    """Canonicalize cell labels without fuzzy punctuation substitutions."""

    text = unicodedata.normalize("NFKC", _clean_text(value))
    text = "".join(
        character for character in text if unicodedata.category(character) != "Cc"
    )
    return " ".join(text.split())


def _clean_int(value: Any) -> int:
    if pd.isna(value):
        return -1
    return int(value)


def _dose_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.12g}"


def _mono_key(values: tuple[Any, ...]) -> tuple[Any, ...]:
    (
        screener,
        study,
        test_date,
        plate,
        panel_nbr,
        cell_nbr,
        nsc,
        sample,
        concentration,
        unit,
    ) = values
    return (
        _clean_text(screener),
        _clean_text(study),
        _clean_text(test_date),
        _clean_text(plate),
        _clean_int(panel_nbr),
        _clean_int(cell_nbr),
        _clean_int(nsc),
        _clean_int(sample),
        _dose_key(concentration),
        _clean_text(unit),
    )


def _stable_ids(prefix: str, values: pd.Series) -> list[str]:
    return [
        f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:20]}"
        for value in values.astype(str).tolist()
    ]


def _raw_chunks(
    config: NCIAlmanacConfig,
    max_chunks: int | None,
) -> Iterator[pd.DataFrame]:
    path = config.raw_dir / "ComboDrugGrowth_Nov2017.zip"
    for index, chunk in enumerate(
        pd.read_csv(path, chunksize=config.chunk_size, low_memory=False)
    ):
        if tuple(chunk.columns) != EXPECTED_COLUMNS:
            raise ValueError("NCI-ALMANAC raw schema does not match the frozen 29-column schema")
        yield chunk
        if max_chunks is not None and index + 1 >= max_chunks:
            break


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _clear_generated_outputs(processed_dir: Path) -> None:
    generated = (
        "points.parquet",
        "monotherapy.parquet",
        "surface_index.parquet",
        "cells.csv",
        "drugs.csv",
        "rejected_structures.csv",
        "structures.pt",
        "structures.jsonl",
        "structure_manifest.json",
        "drug_id_to_structure_key.json",
        "audit.json",
        "prepare_config.json",
        "prepared_audit.json",
        "real_smoke.json",
    )
    for filename in generated:
        path = processed_dir / filename
        if path.is_file():
            path.unlink()


def prepare_nci_almanac(
    config: NCIAlmanacConfig,
    *,
    overwrite: bool = False,
    verify_sources: bool = True,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """Create normalized combination points, monotherapy lookup, and graph caches.

    The raw ZIP is read twice and never extracted. The first pass builds an exact
    same-assay monotherapy lookup and QC ledger; the second writes combination
    points in chunked Parquet row groups.
    """

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    existing = config.processed_dir / "points.parquet"
    if existing.exists() and not overwrite:
        raise FileExistsError(
            f"Processed dataset already exists: {existing}. Pass overwrite=True explicitly."
        )
    if overwrite:
        _clear_generated_outputs(config.processed_dir)
    source_report = verify_source_files(config) if verify_sources else {"skipped": True}

    mono_accumulator: dict[tuple[Any, ...], list[float | int]] = {}
    used_nscs: set[int] = set()
    cells: dict[tuple[int, int, str], dict[str, Any]] = {}
    missing_by_column: Counter[str] = Counter()
    valid_values: Counter[str] = Counter()
    unit_pairs: Counter[tuple[str, str]] = Counter()
    screeners: set[str] = set()
    studies: set[str] = set()
    raw_rows = 0
    combo_rows_first_pass = 0
    mono_rows = 0
    response_min = float("inf")
    response_max = float("-inf")
    response_below_clip = 0
    response_above_clip = 0
    recompute_count = 0
    recompute_within_001 = 0
    recompute_max_abs_error = 0.0

    mono_columns = [
        "SCREENER",
        "STUDY",
        "TESTDATE",
        "PLATE",
        "PANELNBR",
        "CELLNBR",
        "NSC1",
        "SAMPLE1",
        "CONC1",
        "CONCUNIT1",
        config.response.field,
        "COMBODRUGSEQ",
    ]
    for chunk in _raw_chunks(config, max_chunks):
        raw_rows += len(chunk)
        missing_by_column.update(
            {column: int(value) for column, value in chunk.isna().sum().items() if value}
        )
        valid_values.update(chunk["VALID"].fillna("<NA>").astype(str).tolist())
        screeners.update(chunk["SCREENER"].dropna().astype(str))
        studies.update(chunk["STUDY"].dropna().astype(str))
        used_nscs.update(chunk["NSC1"].dropna().astype(int).tolist())
        used_nscs.update(chunk["NSC2"].dropna().astype(int).tolist())
        for row in chunk[
            ["PANELNBR", "CELLNBR", "CELLNAME", "PANEL"]
        ].drop_duplicates().itertuples(index=False, name=None):
            raw_cell_name = _clean_text(row[2])
            key = (_clean_int(row[0]), _clean_int(row[1]), _clean_cell_name(row[2]))
            if key not in cells:
                cells[key] = {
                    "cell_id": f"NCI60:P{key[0]}:C{key[1]}:{key[2]}",
                    "panel_nbr": key[0],
                    "cell_nbr": key[1],
                    "cell_name": key[2],
                    "panel": _clean_text(row[3]),
                    "source_cell_names": set(),
                }
            cells[key]["source_cell_names"].add(raw_cell_name)

        is_combo = chunk["NSC2"].notna()
        combo_rows_first_pass += int(is_combo.sum())
        mono_rows += int((~is_combo).sum())
        unit_counts = (
            chunk.loc[is_combo, ["CONCUNIT1", "CONCUNIT2"]]
            .fillna("<NA>")
            .value_counts()
        )
        for pair, count in unit_counts.items():
            unit_pairs[(_clean_text(pair[0]), _clean_text(pair[1]))] += int(count)

        raw_response = pd.to_numeric(chunk[config.response.field], errors="coerce")
        finite_response = raw_response[np.isfinite(raw_response)]
        if len(finite_response):
            ratio = finite_response / config.response.divide_by
            response_min = min(response_min, float(finite_response.min()))
            response_max = max(response_max, float(finite_response.max()))
            response_below_clip += int((ratio < config.response.clip_min).sum())
            response_above_clip += int((ratio > config.response.clip_max).sum())

        denominator = pd.to_numeric(chunk["CONTROLVALUE"], errors="coerce")
        numerator = pd.to_numeric(chunk["TESTVALUE"], errors="coerce")
        reported = pd.to_numeric(chunk["PERCENTGROWTHNOTZ"], errors="coerce")
        recomputable = (
            denominator.notna()
            & numerator.notna()
            & reported.notna()
            & (denominator != 0)
        )
        if bool(recomputable.any()):
            difference = (
                100.0 * numerator[recomputable] / denominator[recomputable]
                - reported[recomputable]
            ).abs()
            recompute_count += len(difference)
            recompute_within_001 += int((difference <= 0.01).sum())
            recompute_max_abs_error = max(recompute_max_abs_error, float(difference.max()))

        mono_frame = chunk.loc[~is_combo, mono_columns]
        for values in mono_frame.itertuples(index=False, name=None):
            key = _mono_key(values[:10])
            response_value = values[10]
            if pd.isna(response_value):
                continue
            ratio = float(response_value) / config.response.divide_by
            source_seq = _clean_int(values[11])
            if key not in mono_accumulator:
                mono_accumulator[key] = [ratio, 1, source_seq]
            else:
                mono_accumulator[key][0] = float(mono_accumulator[key][0]) + ratio
                mono_accumulator[key][1] = int(mono_accumulator[key][1]) + 1
                mono_accumulator[key][2] = min(int(mono_accumulator[key][2]), source_seq)

    mono_lookup: dict[tuple[Any, ...], tuple[float, float, int, int]] = {}
    mono_rows_out: list[tuple[Any, ...]] = []
    for key, (response_sum, count, source_seq) in mono_accumulator.items():
        raw_ratio = float(response_sum) / int(count)
        clipped = (
            float(np.clip(raw_ratio, config.response.clip_min, config.response.clip_max))
            if config.response.apply_clip
            else raw_ratio
        )
        mono_lookup[key] = (raw_ratio, clipped, int(count), int(source_seq))
        mono_rows_out.append((*key, raw_ratio, clipped, int(count), int(source_seq)))
    mono_frame_out = pd.DataFrame(
        mono_rows_out,
        columns=[
            "screener",
            "study",
            "test_date",
            "plate",
            "panel_nbr",
            "cell_nbr",
            "nsc",
            "sample",
            "dose_key",
            "dose_unit",
            "response_raw_ratio",
            "response",
            "replicate_count",
            "source_combodrugseq",
        ],
    )
    mono_frame_out.to_parquet(
        config.processed_dir / "monotherapy.parquet", index=False, compression="zstd"
    )

    structure_report = build_drug_structure_cache(used_nscs, config)
    cell_rows = []
    for cell in sorted(cells.values(), key=lambda row: row["cell_id"]):
        source_names = sorted(cell.pop("source_cell_names"))
        cell_rows.append(
            {
                **cell,
                "source_cell_names": json.dumps(source_names, ensure_ascii=True),
                "source_alias_count": len(source_names),
            }
        )
    for index, row in enumerate(cell_rows):
        row["cell_index"] = index
        row["feature_status"] = "cell_context_mapping_pending"
    pd.DataFrame(cell_rows).to_csv(
        config.processed_dir / "cells.csv", index=False, encoding="utf-8"
    )

    points_path = config.processed_dir / "points.parquet"
    parquet_writer: pq.ParquetWriter | None = None
    surface_counts: Counter[str] = Counter()
    surface_mono_complete: Counter[str] = Counter()
    surface_primary_points: Counter[str] = Counter()
    surface_metadata: dict[str, tuple[Any, ...]] = {}
    prediction_groups: set[str] = set()
    matched_mono_a = 0
    matched_mono_b = 0
    matched_mono_both = 0
    canonical_swaps = 0
    self_pairs = 0
    written_rows = 0
    dose_min = float("inf")
    dose_max = float("-inf")
    clipped_low = 0
    clipped_high = 0
    response_missing = 0

    for chunk in _raw_chunks(config, max_chunks):
        combo = chunk.loc[chunk["NSC2"].notna()].copy()
        if combo.empty:
            continue
        units = set(combo["CONCUNIT1"].dropna().astype(str)) | set(
            combo["CONCUNIT2"].dropna().astype(str)
        )
        unsupported = units - set(config.dose.accepted_units)
        if unsupported:
            raise ValueError(f"Unsupported dose units in combination records: {unsupported}")

        mono_a_lookup: list[tuple[float, float, int, int] | None] = []
        mono_b_lookup: list[tuple[float, float, int, int] | None] = []
        key_columns_a = [
            "SCREENER",
            "STUDY",
            "TESTDATE",
            "PLATE",
            "PANELNBR",
            "CELLNBR",
            "NSC1",
            "SAMPLE1",
            "CONC1",
            "CONCUNIT1",
        ]
        key_columns_b = [
            "SCREENER",
            "STUDY",
            "TESTDATE",
            "PLATE",
            "PANELNBR",
            "CELLNBR",
            "NSC2",
            "SAMPLE2",
            "CONC2",
            "CONCUNIT2",
        ]
        for values in combo[key_columns_a].itertuples(index=False, name=None):
            mono_a_lookup.append(mono_lookup.get(_mono_key(values)))
        for values in combo[key_columns_b].itertuples(index=False, name=None):
            mono_b_lookup.append(mono_lookup.get(_mono_key(values)))

        raw_nsc_1 = combo["NSC1"].astype(int).to_numpy()
        raw_nsc_2 = combo["NSC2"].astype(int).to_numpy()
        swap = raw_nsc_1 > raw_nsc_2
        canonical_swaps += int(swap.sum())
        self_pairs += int((raw_nsc_1 == raw_nsc_2).sum())
        nsc_a = np.where(swap, raw_nsc_2, raw_nsc_1)
        nsc_b = np.where(swap, raw_nsc_1, raw_nsc_2)

        def choose(column_1: str, column_2: str) -> np.ndarray:
            left = combo[column_1].to_numpy()
            right = combo[column_2].to_numpy()
            return np.where(swap, right, left)

        dose_a = choose("CONC1", "CONC2").astype(float)
        dose_b = choose("CONC2", "CONC1").astype(float)
        dose_min = min(dose_min, float(min(dose_a.min(), dose_b.min())))
        dose_max = max(dose_max, float(max(dose_a.max(), dose_b.max())))
        mono_a_raw_unswapped = np.array(
            [value[0] if value is not None else np.nan for value in mono_a_lookup]
        )
        mono_b_raw_unswapped = np.array(
            [value[0] if value is not None else np.nan for value in mono_b_lookup]
        )
        mono_a_unswapped = np.array(
            [value[1] if value is not None else np.nan for value in mono_a_lookup]
        )
        mono_b_unswapped = np.array(
            [value[1] if value is not None else np.nan for value in mono_b_lookup]
        )
        mono_count_a_unswapped = np.array(
            [value[2] if value is not None else 0 for value in mono_a_lookup], dtype=int
        )
        mono_count_b_unswapped = np.array(
            [value[2] if value is not None else 0 for value in mono_b_lookup], dtype=int
        )
        mono_source_a_unswapped = np.array(
            [value[3] if value is not None else -1 for value in mono_a_lookup], dtype=int
        )
        mono_source_b_unswapped = np.array(
            [value[3] if value is not None else -1 for value in mono_b_lookup], dtype=int
        )
        mono_a_raw = np.where(swap, mono_b_raw_unswapped, mono_a_raw_unswapped)
        mono_b_raw = np.where(swap, mono_a_raw_unswapped, mono_b_raw_unswapped)
        mono_a = np.where(swap, mono_b_unswapped, mono_a_unswapped)
        mono_b = np.where(swap, mono_a_unswapped, mono_b_unswapped)
        mono_count_a = np.where(swap, mono_count_b_unswapped, mono_count_a_unswapped)
        mono_count_b = np.where(swap, mono_count_a_unswapped, mono_count_b_unswapped)
        mono_source_a = np.where(swap, mono_source_b_unswapped, mono_source_a_unswapped)
        mono_source_b = np.where(swap, mono_source_a_unswapped, mono_source_b_unswapped)
        mono_mask_a = np.isfinite(mono_a)
        mono_mask_b = np.isfinite(mono_b)
        matched_mono_a += int(mono_mask_a.sum())
        matched_mono_b += int(mono_mask_b.sum())
        matched_mono_both += int((mono_mask_a & mono_mask_b).sum())

        response_raw_ratio = (
            pd.to_numeric(combo[config.response.field], errors="coerce").to_numpy(dtype=float)
            / config.response.divide_by
        )
        response = (
            np.clip(response_raw_ratio, config.response.clip_min, config.response.clip_max)
            if config.response.apply_clip
            else response_raw_ratio.copy()
        )
        response_mask = np.isfinite(response_raw_ratio)
        response_missing += int((~response_mask).sum())
        low_flags = response_raw_ratio < config.response.clip_min
        high_flags = response_raw_ratio > config.response.clip_max
        clipped_low += int(np.nansum(low_flags))
        clipped_high += int(np.nansum(high_flags))

        canonical_cell_name = combo["CELLNAME"].map(_clean_cell_name)
        cell_key = (
            "P"
            + combo["PANELNBR"].astype("Int64").astype(str)
            + ":C"
            + combo["CELLNBR"].astype("Int64").astype(str)
            + ":"
            + canonical_cell_name
        )
        drug_a_id = pd.Series([f"S{value}" for value in nsc_a], index=combo.index)
        drug_b_id = pd.Series([f"S{value}" for value in nsc_b], index=combo.index)
        prediction_text = cell_key + "|" + drug_a_id + "|" + drug_b_id
        prediction_group_id = _stable_ids("pg", prediction_text)
        assay_text = (
            config.dataset_release
            + "|"
            + combo["SCREENER"].fillna("").astype(str)
            + "|"
            + combo["STUDY"].fillna("").astype(str)
            + "|"
            + combo["TESTDATE"].fillna("").astype(str)
            + "|"
            + combo["PLATE"].fillna("").astype(str)
            + "|"
            + cell_key
        )
        assay_block_id = _stable_ids("ab", assay_text)
        sample_a = choose("SAMPLE1", "SAMPLE2")
        sample_b = choose("SAMPLE2", "SAMPLE1")
        surface_text = (
            pd.Series(assay_block_id, index=combo.index)
            + "|"
            + drug_a_id
            + "|"
            + drug_b_id
        )
        surface_text = surface_text + "|" + pd.Series(sample_a, index=combo.index).astype(str)
        surface_text = surface_text + "|" + pd.Series(sample_b, index=combo.index).astype(str)
        surface_replicate_id = _stable_ids("sr", surface_text)

        dose_reference = config.dose.model_reference_molar
        output = pd.DataFrame(
            {
                "source_combodrugseq": combo["COMBODRUGSEQ"].astype("int64").to_numpy(),
                "dataset_release": config.dataset_release,
                "prediction_group_id": prediction_group_id,
                "assay_block_id": assay_block_id,
                "surface_replicate_id": surface_replicate_id,
                "screener": combo["SCREENER"].astype(str).to_numpy(),
                "study": combo["STUDY"].astype(str).to_numpy(),
                "test_date": combo["TESTDATE"].astype(str).to_numpy(),
                "plate": combo["PLATE"].astype(str).to_numpy(),
                "panel_nbr": combo["PANELNBR"].astype("int64").to_numpy(),
                "cell_nbr": combo["CELLNBR"].astype("int64").to_numpy(),
                "panel": combo["PANEL"].fillna("").astype(str).to_numpy(),
                "cell_name": canonical_cell_name.to_numpy(),
                "cell_id": ("NCI60:" + cell_key).to_numpy(),
                "drug_a_id": drug_a_id.to_numpy(),
                "drug_b_id": drug_b_id.to_numpy(),
                "nsc_a": nsc_a,
                "nsc_b": nsc_b,
                "sample_a": sample_a,
                "sample_b": sample_b,
                "conc_index_a": choose("CONCINDEX1", "CONCINDEX2"),
                "conc_index_b": choose("CONCINDEX2", "CONCINDEX1"),
                "dose_a_raw": dose_a,
                "dose_b_raw": dose_b,
                "dose_a_unit_raw": choose("CONCUNIT1", "CONCUNIT2"),
                "dose_b_unit_raw": choose("CONCUNIT2", "CONCUNIT1"),
                "dose_a_M": dose_a,
                "dose_b_M": dose_b,
                "log10_dose_a_M": np.log10(dose_a),
                "log10_dose_b_M": np.log10(dose_b),
                "dose_a_model": np.log10(1.0 + dose_a / dose_reference),
                "dose_b_model": np.log10(1.0 + dose_b / dose_reference),
                "mono_response_a_raw_ratio": mono_a_raw,
                "mono_response_b_raw_ratio": mono_b_raw,
                "mono_response_a": mono_a,
                "mono_response_b": mono_b,
                "mono_mask_a": mono_mask_a,
                "mono_mask_b": mono_mask_b,
                "mono_replicate_count_a": mono_count_a,
                "mono_replicate_count_b": mono_count_b,
                "mono_source_combodrugseq_a": mono_source_a,
                "mono_source_combodrugseq_b": mono_source_b,
                "percent_growth_raw": combo["PERCENTGROWTH"].to_numpy(dtype=float),
                "percent_growth_no_tz_raw": combo["PERCENTGROWTHNOTZ"].to_numpy(dtype=float),
                "response_raw_ratio": response_raw_ratio,
                "response": response,
                "response_mask": response_mask,
                "response_clipped_low": low_flags,
                "response_clipped_high": high_flags,
                "audit_expected_growth": combo["EXPECTEDGROWTH"].to_numpy(dtype=float),
                "audit_official_score": combo["SCORE"].to_numpy(dtype=float),
                "official_valid": combo["VALID"].astype(str).to_numpy(),
                "primary_metric_mask": response_mask & (dose_a > 0) & (dose_b > 0),
                "is_boundary": False,
            }
        )
        table = pa.Table.from_pandas(output, preserve_index=False)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(points_path, table.schema, compression="zstd")
        parquet_writer.write_table(table)

        surface_ids = output["surface_replicate_id"].tolist()
        surface_counts.update(surface_ids)
        complete = output.loc[
            output["mono_mask_a"] & output["mono_mask_b"], "surface_replicate_id"
        ].tolist()
        surface_mono_complete.update(complete)
        surface_primary_points.update(
            output.loc[output["primary_metric_mask"], "surface_replicate_id"].tolist()
        )
        prediction_groups.update(output["prediction_group_id"].tolist())
        for row in output[
            [
                "surface_replicate_id",
                "prediction_group_id",
                "assay_block_id",
                "drug_a_id",
                "drug_b_id",
                "cell_id",
                "screener",
                "study",
                "plate",
            ]
        ].drop_duplicates("surface_replicate_id").itertuples(index=False, name=None):
            surface_metadata.setdefault(str(row[0]), tuple(row[1:]))
        written_rows += len(output)

    if parquet_writer is None:
        raise RuntimeError("No combination records were written")
    parquet_writer.close()

    surface_rows = []
    for surface_id, metadata in surface_metadata.items():
        points = surface_counts[surface_id]
        surface_rows.append(
            {
                "surface_replicate_id": surface_id,
                "prediction_group_id": metadata[0],
                "assay_block_id": metadata[1],
                "drug_a_id": metadata[2],
                "drug_b_id": metadata[3],
                "cell_id": metadata[4],
                "screener": metadata[5],
                "study": metadata[6],
                "plate": metadata[7],
                "point_count": points,
                "mono_complete_points": surface_mono_complete[surface_id],
                "mono_complete_fraction": surface_mono_complete[surface_id] / points,
                "primary_points": surface_primary_points[surface_id],
                "primary_fraction": surface_primary_points[surface_id] / points,
            }
        )
    surface_frame = pd.DataFrame(surface_rows)
    surface_frame.to_parquet(
        config.processed_dir / "surface_index.parquet", index=False, compression="zstd"
    )

    raw_qc = {
        "rows": raw_rows,
        "combination_rows": combo_rows_first_pass,
        "monotherapy_rows": mono_rows,
        "columns": list(EXPECTED_COLUMNS),
        "missing_by_column": dict(sorted(missing_by_column.items())),
        "valid_values": dict(valid_values),
        "unit_pairs": {f"{left}/{right}": count for (left, right), count in unit_pairs.items()},
        "screeners": sorted(screeners),
        "studies": len(studies),
        "cells": len(cells),
        "used_nscs": len(used_nscs),
        "response_field": config.response.field,
        "response_raw_min": response_min,
        "response_raw_max": response_max,
        "response_below_clip": response_below_clip,
        "response_above_clip": response_above_clip,
        "percent_growth_no_tz_recompute_count": recompute_count,
        "percent_growth_no_tz_recompute_within_0_01": recompute_within_001,
        "percent_growth_no_tz_recompute_max_abs_error": recompute_max_abs_error,
    }
    processed_qc = {
        "written_combination_points": written_rows,
        "surface_replicates": len(surface_counts),
        "prediction_groups": len(prediction_groups),
        "surface_point_count_distribution": {
            str(count): frequency for count, frequency in Counter(surface_counts.values()).items()
        },
        "canonical_orientation_swaps": canonical_swaps,
        "self_pair_rows": self_pairs,
        "mono_match_a": matched_mono_a,
        "mono_match_b": matched_mono_b,
        "mono_match_both": matched_mono_both,
        "mono_match_both_fraction": matched_mono_both / max(written_rows, 1),
        "dose_M_min": dose_min,
        "dose_M_max": dose_max,
        "response_clipped_low": clipped_low,
        "response_clipped_high": clipped_high,
        "response_missing": response_missing,
        "trainable_primary_points": sum(surface_primary_points.values()),
        "partial_run": max_chunks is not None,
        "max_chunks": max_chunks,
    }
    audit = {
        "status": "completed",
        "dataset_release": config.dataset_release,
        "source_verification": source_report,
        "response_transform": asdict(config.response),
        "dose_transform": asdict(config.dose),
        "raw_qc": raw_qc,
        "processed_qc": processed_qc,
        "structures": structure_report,
    }
    _write_json(config.processed_dir / "audit.json", audit)
    _write_json(
        config.processed_dir / "prepare_config.json",
        {
            "dataset_release": config.dataset_release,
            "raw_dir": str(config.raw_dir),
            "processed_dir": str(config.processed_dir),
            "source_manifest": str(config.source_manifest),
            "chunk_size": config.chunk_size,
            "response": asdict(config.response),
            "dose": asdict(config.dose),
            "structure": asdict(config.structure),
        },
    )
    return audit


def audit_prepared_nci(config: NCIAlmanacConfig) -> dict[str, Any]:
    points_path = config.processed_dir / "points.parquet"
    surface_path = config.processed_dir / "surface_index.parquet"
    mono_path = config.processed_dir / "monotherapy.parquet"
    cells_path = config.processed_dir / "cells.csv"
    required = [
        points_path,
        surface_path,
        mono_path,
        cells_path,
        config.processed_dir / "audit.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared files are missing: {missing}")

    columns = [
        "surface_replicate_id",
        "prediction_group_id",
        "cell_id",
        "drug_a_id",
        "drug_b_id",
        "nsc_a",
        "nsc_b",
        "dose_a_M",
        "dose_b_M",
        "response_mask",
        "primary_metric_mask",
    ]
    points = pq.read_table(points_path, columns=columns)
    frame = points.to_pandas()
    surface_index = pd.read_parquet(surface_path)
    cells = pd.read_csv(cells_path, keep_default_na=False)
    mapping = json.loads(
        (config.processed_dir / "drug_id_to_structure_key.json").read_text(
            encoding="utf-8"
        )
    )
    source_a = set(frame["drug_a_id"].astype(str))
    source_b = set(frame["drug_b_id"].astype(str))
    primary = frame["primary_metric_mask"].astype(bool)
    checks = {
        "canonical_drug_order": bool((frame["nsc_a"] <= frame["nsc_b"]).all()),
        "positive_combination_doses": bool(
            ((frame["dose_a_M"] > 0) & (frame["dose_b_M"] > 0)).all()
        ),
        "primary_points_have_response": bool(frame.loc[primary, "response_mask"].all()),
        "all_drugs_have_graph_mapping": (source_a | source_b) <= set(mapping),
        "surface_point_totals_match": int(surface_index["point_count"].sum())
        == len(frame),
        "surface_ids_unique_in_index": not bool(
            surface_index["surface_replicate_id"].duplicated().any()
        ),
        "surface_count_matches_points": int(
            frame["surface_replicate_id"].nunique()
        )
        == len(surface_index),
        "prediction_group_count_matches_index": int(
            frame["prediction_group_id"].nunique()
        )
        == int(surface_index["prediction_group_id"].nunique()),
        "cell_ids_unique": not bool(cells["cell_id"].duplicated().any()),
        "point_cells_have_canonical_record": set(frame["cell_id"].astype(str))
        <= set(cells["cell_id"].astype(str)),
        "canonical_cell_names_have_no_control_characters": not any(
            unicodedata.category(character) == "Cc"
            for name in cells["cell_name"].astype(str)
            for character in name
        ),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "combination_points": len(frame),
        "primary_points": int(primary.sum()),
        "surface_replicates": int(frame["surface_replicate_id"].nunique()),
        "prediction_groups": int(frame["prediction_group_id"].nunique()),
        "monotherapy_records": pq.ParquetFile(mono_path).metadata.num_rows,
        "mapped_drugs": len(mapping),
        "canonical_cells": len(cells),
        "source_alias_groups": int((cells["source_alias_count"] > 1).sum()),
    }
    _write_json(config.processed_dir / "prepared_audit.json", report)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Prepared NCI audit failed: {failed}")
    return report

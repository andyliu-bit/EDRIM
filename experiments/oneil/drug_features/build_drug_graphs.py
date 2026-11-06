"""O'Neil drug graph cache and Morgan-512 fingerprints (launch card E v2, docs/ONEIL_COMPARISON_CARD_20261020.md; D13).

Mirrors bisurfgnn.data.nci_almanac.build_drug_structure_cache for the 38 O'Neil drugs: same StandardizationPolicy
(read from the NCI structure_manifest.json), same smiles_to_graph and structure_cache_key, same output files
(structures.pt, drug_id_to_structure_key.json, structure_manifest.json), keyed by the O'Neil drug name from
validated_identity.json. Input is the audited salt-stripped canonical parent SMILES with stereo kept.
morgan512.csv uses the generator of run_comparator_pilot.morgan512 (radius 2, 512 bits) on the same SMILES.
Reads no response file. The self-check loads the cache the way independent_graph_benchmark.build_graph_feature_banks
does and compares drugs shared with NCI (same full InChIKey) against the NCI cache.

Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/drug_features/build_drug_graphs.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, DataStructs, rdBase  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

from bisurfgnn import independent_graph_benchmark as graph  # noqa: E402
from bisurfgnn.data.chemistry import (  # noqa: E402
    ATOM_FEATURE_DIM, BOND_FEATURE_DIM, StandardizationPolicy, smiles_to_graph, standardize_smiles,
    structure_cache_key)

MAIN = ROOT.parent / "bisurfgnn"
IDENTITY = MAIN / "runs/audit_revision/oneil_identity_audit_v1/validated_identity.json"
NCI = MAIN / "data/processed/nci_almanac_unclipped"
OUT = ROOT / "runs/oneil_v1/drug_features"
FORBIDDEN_PARTS = ("oneil_confirmation_eval_v1", "external_combokr2_oneil_newcell_v1")
FORBIDDEN_NAMES = ("oneil_per_surface.csv", "oneil_fitted_Braids.npy")
FP_COLUMNS = [f"fp_{i:03d}" for i in range(512)]


def allowed(path: Path) -> Path:
    path = Path(path).resolve()
    if any(p in FORBIDDEN_PARTS for p in path.parts) or path.name in FORBIDDEN_NAMES:
        raise PermissionError(f"Refusing to open a card-forbidden file: {path}")
    return path


def morgan_bits(smiles: str) -> np.ndarray:
    """Same generator and conversion as run_comparator_pilot.morgan512."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError(f"Invalid canonical SMILES: {smiles}")
    vector = np.zeros(512, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(
        rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512).GetFingerprint(molecule), vector)
    return vector


def build() -> None:
    nci_manifest = json.loads(allowed(NCI / "structure_manifest.json").read_text(encoding="utf-8"))
    if nci_manifest["rdkit_version"] != rdBase.rdkitVersion:  # cache keys embed the RDKit version
        raise RuntimeError(f"RDKit {rdBase.rdkitVersion} differs from the NCI cache ({nci_manifest['rdkit_version']})")
    policy = StandardizationPolicy(**nci_manifest["policy"])
    drugs = json.loads(allowed(IDENTITY).read_text(encoding="utf-8"))["drugs"]
    if len(drugs) != 38 or len({d["name"] for d in drugs}) != 38:
        raise RuntimeError("validated_identity.json must list 38 distinct drug names")

    cache, mapping, rows, fps, multi = {}, {}, [], [], {}
    for drug in sorted(drugs, key=lambda d: d["name"]):
        name, candidates = drug["name"], drug["candidates"]
        parents = {c["standard"]["canonical_smiles"] for c in candidates}
        # Each candidate's raw PubChem SMILES must reproduce the audited parent under this policy.
        recomputed = {standardize_smiles(c["standard"]["raw_smiles"], policy).canonical_smiles for c in candidates}
        if len(parents) != 1 or recomputed != parents:
            raise SystemExit(f"STOP {name}: candidates differ in parent structure: audit={parents} recomputed={recomputed}")
        smiles = parents.pop()
        structure, molecular_graph = smiles_to_graph(smiles, policy)
        if structure.canonical_smiles != smiles:
            raise RuntimeError(f"{name}: standardisation is not idempotent ({smiles} -> {structure.canonical_smiles})")
        if {c["standard"]["inchikey"] for c in candidates} != {structure.inchikey}:
            raise RuntimeError(f"{name}: recomputed InChIKey differs from the identity audit")
        key = structure_cache_key(structure, policy)
        cache.setdefault(key, molecular_graph.to_tensor_dict())
        mapping[name] = key
        if len(candidates) > 1:
            multi[name] = [c["cid"] for c in candidates]
        rows.append(dict(
            drug_id=name, pubchem_cids=json.dumps([c["cid"] for c in candidates]), identity_status=drug["status"],
            canonical_smiles=smiles, inchikey=structure.inchikey,
            neutral_identity_inchikey=structure.neutral_identity_inchikey, identity_group=structure.connectivity_key,
            murcko_scaffold=structure.murcko_scaffold, structure_key=key,
            identity_flags=json.dumps(sorted({f for c in candidates for f in c["standard"]["flags"]})),
            num_atoms=int(molecular_graph.node_features.shape[0])))
        fps.append(morgan_bits(smiles))

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(cache, OUT / "structures.pt")
    (OUT / "drug_id_to_structure_key.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "drugs.csv", index=False, encoding="utf-8")
    morgan = pd.DataFrame(np.stack(fps).astype(np.int8), columns=FP_COLUMNS)
    morgan.insert(0, "drug_id", [r["drug_id"] for r in rows])
    morgan.to_csv(OUT / "morgan512.csv", index=False, encoding="utf-8")
    (OUT / "structure_manifest.json").write_text(json.dumps({
        "rdkit_version": rdBase.rdkitVersion,
        "policy": asdict(policy),
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "drugs": len(drugs),
        "mapped_drugs": len(mapping),
        "unique_structures": len(cache),
        "rejected_drugs": 0,
        "source": str(IDENTITY),
        "smiles_field": "drugs[].candidates[].standard.canonical_smiles (salt-stripped parent, stereo kept)",
        "policy_source": str(NCI / "structure_manifest.json"),
        "multi_candidate_same_parent": multi,
        "morgan512": {"generator": "GetMorganGenerator(radius=2, fpSize=512)", "input": "canonical_smiles"},
    }, indent=2), encoding="utf-8")


def load_drug_graphs(directory: Path, drug_ids):
    """Drug half of independent_graph_benchmark.build_graph_feature_banks, pointed at another processed dir."""
    mapping = json.loads(allowed(directory / "drug_id_to_structure_key.json").read_text(encoding="utf-8"))
    cache = torch.load(allowed(directory / "structures.pt"), map_location="cpu", weights_only=True)
    manifest = json.loads(allowed(directory / "structure_manifest.json").read_text(encoding="utf-8"))
    graphs = {d: graph._as_graph_item(cache[mapping[d]], f"drug {d}") for d in sorted(drug_ids)}
    if any(g.node_features.shape[1] != int(manifest["atom_feature_dim"]) for g in graphs.values()):
        raise RuntimeError("Structure manifest atom feature dimension changed")
    return mapping, cache, graphs


def check() -> None:
    ours = pd.read_csv(allowed(OUT / "drugs.csv"), keep_default_na=False)
    mapping, cache, graphs = load_drug_graphs(OUT, ours.drug_id)
    assert len(graphs) == 38 == len(set(mapping.values())), "expected 38 drugs with 38 distinct structures"
    print(f"{'drug_id':<16}{'atoms':>6}{'bonds':>6}  flags")
    for row in ours.itertuples(index=False):
        g, payload = graphs[row.drug_id], cache[mapping[row.drug_id]]
        n, bonds = g.node_features.shape[0], payload["edge_index"].shape[1] // 2
        assert g.adjacency.shape == (n, n) and torch.equal(g.adjacency, g.adjacency.T)
        assert torch.all(g.adjacency.diagonal() == 1) and int(g.adjacency.sum()) == n + 2 * bonds
        assert payload["edge_features"].shape == (2 * bonds, BOND_FEATURE_DIM) and n == row.num_atoms
        print(f"{row.drug_id:<16}{n:>6}{bonds:>6}  {row.identity_flags}")

    morgan = pd.read_csv(allowed(OUT / "morgan512.csv")).set_index("drug_id")
    assert morgan.shape == (38, 512) and list(morgan.index) == list(ours.drug_id) and morgan.isin([0, 1]).all().all()

    nci = pd.read_csv(allowed(NCI / "drugs.csv"), keep_default_na=False)
    nci_mapping, nci_cache, _ = load_drug_graphs(NCI, [])
    spec = importlib.util.spec_from_file_location("nci_pilot", ROOT / "experiments/model_improvement/run_comparator_pilot.py")
    pilot = importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
    print("\nShared with NCI (full InChIKey -> structure key, 4 graph tensors, Morgan bits via pilot.morgan512):")
    full, identical = 0, 0
    for row in ours.itertuples(index=False):
        exact = nci[nci.inchikey == row.inchikey]
        loose = nci[(nci.identity_group == row.identity_group) & (nci.inchikey != row.inchikey)]
        for kind, matches in (("full", exact), ("connectivity-only", loose)):
            for match in matches.itertuples(index=False):
                ref = nci_cache[nci_mapping[match.drug_id]]
                same_graph = ref.keys() == cache[mapping[row.drug_id]].keys() and all(
                    torch.equal(ref[k], cache[mapping[row.drug_id]][k]) for k in ref)
                nci_fp = pilot.morgan512(MAIN, SimpleNamespace(drug_ids=[match.drug_id]))[0].numpy()
                same_fp = np.array_equal(nci_fp, morgan.loc[row.drug_id].to_numpy())
                same_key = nci_mapping[match.drug_id] == mapping[row.drug_id]
                print(f"  {kind:<18}{row.drug_id:<14}{match.drug_id:<9}{match.preferred_name[:22]:<23}"
                      f"key={same_key!s:<6}graph={same_graph!s:<6}morgan={same_fp!s:<6}"
                      f"atoms {cache[mapping[row.drug_id]]['node_features'].shape[0]}/{ref['node_features'].shape[0]}")
                if kind == "full":
                    full += 1
                    identical += same_key and same_graph and same_fp
    print(f"\nfull-InChIKey matches: {full}; identical key+graph+Morgan: {identical}")
    assert full == identical, "a drug with the same full InChIKey has a different NCI graph"
    print("CHECK OK")


if __name__ == "__main__":
    if "--check-only" not in sys.argv:
        build()
    check()

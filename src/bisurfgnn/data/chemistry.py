from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from rdkit import Chem, rdBase
from rdkit.Chem import inchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch import Tensor

from .batch import GraphItem


STANDARDIZATION_VERSION = "nci-v1"
ATOM_FEATURE_VERSION = "atom-v1"
BOND_FEATURE_VERSION = "bond-v1"

ELEMENTS = ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I")
DEGREES = (0, 1, 2, 3, 4, 5)
FORMAL_CHARGES = (-2, -1, 0, 1, 2)
TOTAL_VALENCES = (0, 1, 2, 3, 4, 5, 6)
TOTAL_HYDROGENS = (0, 1, 2, 3, 4)
HYBRIDIZATIONS = (
    Chem.HybridizationType.S,
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
    Chem.HybridizationType.UNSPECIFIED,
)
CHIRAL_TAGS = (
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)
RADICAL_ELECTRONS = (0, 1, 2)
BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)
BOND_STEREO = (
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOANY,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
)

ATOM_FEATURE_DIM = (
    len(ELEMENTS)
    + 1
    + len(DEGREES)
    + 1
    + len(FORMAL_CHARGES)
    + 1
    + len(TOTAL_VALENCES)
    + 1
    + len(TOTAL_HYDROGENS)
    + 1
    + len(HYBRIDIZATIONS)
    + 1
    + len(CHIRAL_TAGS)
    + 1
    + len(RADICAL_ELECTRONS)
    + 1
    + 5
)
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + 2 + len(BOND_STEREO) + 1 + 1


class StructureStandardizationError(ValueError):
    pass


@dataclass(frozen=True)
class StandardizationPolicy:
    reject_ambiguous_multicomponent: bool = True
    canonicalize_tautomer: bool = True
    preserve_formal_charge: bool = True
    max_heavy_atoms: int = 256


@dataclass(frozen=True)
class StandardizedStructure:
    raw_smiles: str
    canonical_smiles: str
    pre_tautomer_smiles: str
    neutral_identity_smiles: str
    inchikey: str
    neutral_identity_inchikey: str
    connectivity_key: str
    murcko_scaffold: str
    removed_fragments: tuple[str, ...]
    flags: tuple[str, ...]


@dataclass(frozen=True)
class MolecularGraph:
    node_features: Tensor
    adjacency: Tensor
    edge_index: Tensor
    edge_features: Tensor

    def to_graph_item(self) -> GraphItem:
        return GraphItem(node_features=self.node_features, adjacency=self.adjacency)

    def to_tensor_dict(self) -> dict[str, Tensor]:
        return {
            "node_features": self.node_features,
            "adjacency": self.adjacency,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
        }


def _one_hot(value: Any, vocabulary: Sequence[Any]) -> list[float]:
    encoded = [0.0] * (len(vocabulary) + 1)
    try:
        encoded[vocabulary.index(value)] = 1.0
    except ValueError:
        encoded[-1] = 1.0
    return encoded


def _safe_inchikey(molecule: Chem.Mol) -> str:
    try:
        return inchi.MolToInchiKey(molecule) or ""
    except Exception:
        return ""


def _fragment_smiles(molecule: Chem.Mol) -> tuple[str, ...]:
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    return tuple(sorted(Chem.MolToSmiles(fragment, isomericSmiles=True) for fragment in fragments))


def _contains_metal(molecule: Chem.Mol) -> bool:
    nonmetal_like = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
    return any(atom.GetAtomicNum() not in nonmetal_like for atom in molecule.GetAtoms())


def standardize_smiles(
    smiles: str,
    policy: StandardizationPolicy | None = None,
) -> StandardizedStructure:
    policy = policy or StandardizationPolicy()
    if not isinstance(smiles, str) or not smiles.strip():
        raise StructureStandardizationError("SMILES must be a non-empty string")
    raw_smiles = smiles.strip()
    molecule = Chem.MolFromSmiles(raw_smiles, sanitize=True)
    if molecule is None:
        raise StructureStandardizationError(f"RDKit could not sanitize SMILES: {raw_smiles}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)

    contains_metal = _contains_metal(molecule)
    flags: list[str] = []
    if contains_metal:
        # FragmentParent/Cleanup can silently reduce coordination complexes such as
        # cisplatin to ammonia. Preserve the connected, explicitly audited graph.
        cleaned = Chem.Mol(molecule)
        flags.extend(["contains_metal_or_heavy_element", "metal_standardization_bypassed"])
    else:
        try:
            cleaned = rdMolStandardize.Cleanup(molecule)
        except Exception as error:
            raise StructureStandardizationError(f"RDKit Cleanup failed: {error}") from error
    fragments = Chem.GetMolFrags(cleaned, asMols=True, sanitizeFrags=True)
    ambiguous = [
        fragment
        for fragment in fragments
        if fragment.GetNumHeavyAtoms() >= 3
        and any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms())
    ]
    if len(ambiguous) > 1:
        flags.append("ambiguous_multicomponent")
        if policy.reject_ambiguous_multicomponent:
            raise StructureStandardizationError(
                "Multiple substantial organic fragments require manual resolution"
            )

    parent = Chem.Mol(cleaned) if contains_metal else rdMolStandardize.FragmentParent(cleaned)
    if parent is None or parent.GetNumAtoms() == 0:
        raise StructureStandardizationError("No parent fragment remained after cleanup")
    if parent.GetNumHeavyAtoms() > policy.max_heavy_atoms:
        raise StructureStandardizationError(
            f"Structure has {parent.GetNumHeavyAtoms()} heavy atoms; "
            f"maximum is {policy.max_heavy_atoms}"
        )
    pre_tautomer_smiles = Chem.MolToSmiles(parent, isomericSmiles=True, canonical=True)
    standardized = parent
    if policy.canonicalize_tautomer and not contains_metal:
        standardized = rdMolStandardize.TautomerEnumerator().Canonicalize(parent)
    canonical_smiles = Chem.MolToSmiles(
        standardized, isomericSmiles=True, canonical=True
    )
    if canonical_smiles != pre_tautomer_smiles:
        flags.append("tautomer_changed")

    standardized = Chem.MolFromSmiles(canonical_smiles, sanitize=True)
    if standardized is None:
        raise StructureStandardizationError("Canonical SMILES failed round-trip parsing")
    if any(atom.GetIsotope() for atom in standardized.GetAtoms()):
        flags.append("contains_isotope")
    if _contains_metal(standardized):
        flags.append("contains_metal_or_heavy_element")

    neutral = Chem.Mol(standardized)
    if policy.preserve_formal_charge:
        try:
            neutral = rdMolStandardize.Uncharger().uncharge(neutral)
        except Exception:
            flags.append("neutral_identity_failed")
            neutral = Chem.Mol(standardized)
    neutral_smiles = Chem.MolToSmiles(neutral, isomericSmiles=True, canonical=True)
    exact_inchikey = _safe_inchikey(standardized)
    neutral_inchikey = _safe_inchikey(neutral)
    connectivity_key = (neutral_inchikey or exact_inchikey).split("-")[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=standardized, includeChirality=False
    )

    all_fragments = _fragment_smiles(cleaned)
    removed = tuple(fragment for fragment in all_fragments if fragment != pre_tautomer_smiles)
    if removed:
        flags.append("fragments_removed")
    return StandardizedStructure(
        raw_smiles=raw_smiles,
        canonical_smiles=canonical_smiles,
        pre_tautomer_smiles=pre_tautomer_smiles,
        neutral_identity_smiles=neutral_smiles,
        inchikey=exact_inchikey,
        neutral_identity_inchikey=neutral_inchikey,
        connectivity_key=connectivity_key,
        murcko_scaffold=scaffold,
        removed_fragments=removed,
        flags=tuple(sorted(set(flags))),
    )


def _atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ELEMENTS)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetTotalValence(), TOTAL_VALENCES)
        + _one_hot(atom.GetTotalNumHs(), TOTAL_HYDROGENS)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + _one_hot(atom.GetChiralTag(), CHIRAL_TAGS)
        + _one_hot(atom.GetNumRadicalElectrons(), RADICAL_ELECTRONS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetAtomicNum() / 100.0,
            atom.GetMass() / 250.0,
            float(atom.GetIsotope() != 0),
        ]
    )


def _bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + _one_hot(bond.GetStereo(), BOND_STEREO)
        + [float(bond.GetBondTypeAsDouble())]
    )


def standardized_smiles_to_graph(canonical_smiles: str) -> MolecularGraph:
    molecule = Chem.MolFromSmiles(canonical_smiles, sanitize=True)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise StructureStandardizationError("Cannot build graph from canonical SMILES")
    node_features = torch.tensor(
        [_atom_features(atom) for atom in molecule.GetAtoms()], dtype=torch.float32
    )
    if node_features.shape[1] != ATOM_FEATURE_DIM:
        raise AssertionError("Atom feature schema dimension drifted")
    nodes = molecule.GetNumAtoms()
    adjacency = torch.eye(nodes, dtype=torch.float32)
    edge_records: list[tuple[int, int, list[float]]] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        adjacency[begin, end] = 1.0
        adjacency[end, begin] = 1.0
        features = _bond_features(bond)
        edge_records.append((begin, end, features))
        edge_records.append((end, begin, features))
    edge_records.sort(key=lambda record: (record[0], record[1]))
    if edge_records:
        edge_index = torch.tensor(
            [[record[0], record[1]] for record in edge_records], dtype=torch.long
        ).T.contiguous()
        edge_features = torch.tensor(
            [record[2] for record in edge_records], dtype=torch.float32
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_features = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
    if edge_features.shape[1] != BOND_FEATURE_DIM:
        raise AssertionError("Bond feature schema dimension drifted")
    return MolecularGraph(
        node_features=node_features,
        adjacency=adjacency,
        edge_index=edge_index,
        edge_features=edge_features,
    )


def smiles_to_graph(
    smiles: str,
    policy: StandardizationPolicy | None = None,
) -> tuple[StandardizedStructure, MolecularGraph]:
    structure = standardize_smiles(smiles, policy)
    graph = standardized_smiles_to_graph(structure.canonical_smiles)
    return structure, graph


def structure_cache_key(
    structure: StandardizedStructure,
    policy: StandardizationPolicy,
) -> str:
    payload = "|".join(
        [
            structure.canonical_smiles,
            rdBase.rdkitVersion,
            STANDARDIZATION_VERSION,
            ATOM_FEATURE_VERSION,
            BOND_FEATURE_VERSION,
            repr(asdict(policy)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

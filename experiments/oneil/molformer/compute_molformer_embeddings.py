"""MoLFormer 768-d drug embeddings for O'Neil (launch card E v2, docs/ONEIL_COMPARISON_CARD_20261020.md; D13).

Recipe (card): HF ibm-research/MoLFormer-XL-both-10pct, deterministic_eval=True, eval + no_grad, one SMILES per forward
(no padding), mean pooling of last_hidden_state over the attention mask. Input SMILES are RDKit canonical with
isomericSmiles=False, the DeepSynBa author canonicalisation (blocks/drug_encoder.py L38-39).

Model files: code/config/tokenizer from commit 7b12d946 (data/external/molformer_hf_7b12d946; the 361063d0 modeling code
imports transformers.masking_utils, missing in transformers 4.46.3); weights from data/external/molformer_hf_361063d0
(model.safetensors LFS sha256 0795977f... is identical at 7b12d946, a14249e5 and 361063d0 per the HF tree API).
Both tokenizer revisions are run on the 104 DeepSynBa NCI drugs and compared with the author file; the better one is
used for O'Neil. Reads no response file of any dataset.

Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/molformer/compute_molformer_embeddings.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent / "bisurfgnn"
CODE_DIR = ROOT / "data/external/molformer_hf_7b12d946"
WEIGHT_DIR = ROOT / "data/external/molformer_hf_361063d0"
TOKENIZER_DIRS = {"7b12d946": CODE_DIR, "361063d0": WEIGHT_DIR}
AUTHOR_PT = MAIN / "data/raw/external_benchmarks/deepsynba/data/features/NCI-ALMANAC_drug_molformer_embeddings.pt"
NCI_DRUGS = MAIN / "data/processed/nci_almanac_unclipped/drugs.csv"
NCI_NAME_MAP = ROOT / "runs/model_improvement/public_split_gate_v1/s1/drug_mapping_used.csv"
IDENTITY = MAIN / "runs/audit_revision/oneil_identity_audit_v1/validated_identity.json"
GRAPH_DRUGS = ROOT / "runs/oneil_v1/drug_features/drugs.csv"
OUT = ROOT / "runs/oneil_v1/molformer"
FORBIDDEN_PARTS = ("oneil_confirmation_eval_v1", "external_combokr2_oneil_newcell_v1")
FORBIDDEN_NAMES = ("oneil_per_surface.csv", "oneil_fitted_Braids.npy")
NCI_SMILES_SOURCES = {"nci_sdf_raw": "raw_smiles", "parent": "canonical_smiles"}


def allowed(path: Path) -> Path:
    path = Path(path).resolve()
    if any(p in FORBIDDEN_PARTS for p in path.parts) or path.name in FORBIDDEN_NAMES:
        raise PermissionError(f"Refusing to open a card-forbidden file: {path}")
    return path


def author_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=False)


def load_model():
    config = AutoConfig.from_pretrained(allowed(CODE_DIR), trust_remote_code=True)
    config.deterministic_eval = True
    model_class = get_class_from_dynamic_module("modeling_molformer.MolformerModel", allowed(CODE_DIR))
    model, info = model_class.from_pretrained(allowed(WEIGHT_DIR), config=config, output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
        raise RuntimeError(f"MoLFormer weights did not load cleanly: {info}")
    return model.eval()


@torch.no_grad()
def embed(model, tokenizer, smiles: str) -> torch.Tensor:
    encoded = tokenizer(smiles, return_tensors="pt")
    output = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
    if not torch.allclose(pooled, output.pooler_output, atol=1e-6):  # pooler is the same masked mean; sanity only
        raise RuntimeError(f"Mean pooling differs from pooler_output for {smiles}")
    return pooled.float().contiguous()  # (1, 768)


def nci_smiles() -> pd.DataFrame:
    """Author key -> NCI drug row; 101 keys via the S1 mapping, the 3 unused author keys by exact name in all_names."""
    author = torch.load(allowed(AUTHOR_PT), weights_only=True)
    drugs = pd.read_csv(allowed(NCI_DRUGS)).set_index("drug_id")
    mapping = dict(pd.read_csv(allowed(NCI_NAME_MAP)).set_index("deepsynba_drug")["canonical_drug_id"])
    rows = []
    for key in sorted(author):
        drug_id = mapping.get(key)
        if drug_id is None:
            hits = [i for i, names in drugs["all_names"].items() if key.lower() in {n.lower() for n in json.loads(names)}]
            if len(hits) != 1:
                raise RuntimeError(f"Author key {key!r} matches {hits} in drugs.csv")
            drug_id = hits[0]
        row = drugs.loc[drug_id]
        rows.append(dict(author_key=key, drug_id=drug_id, nsc=int(row["nsc"]), structure_source=row["structure_source"],
                         **{f"smiles_{name}": author_smiles(row[column]) for name, column in NCI_SMILES_SOURCES.items()}))
    table = pd.DataFrame(rows)
    if len(table) != 104 or table["drug_id"].nunique() != 104:
        raise RuntimeError("Expected 104 author keys mapped one-to-one to NCI drug ids")
    return table, author


def oneil_smiles() -> dict[str, str]:
    drugs = json.loads(allowed(IDENTITY).read_text(encoding="utf-8"))["drugs"]
    smiles = {}
    for drug in drugs:
        parents = {c["standard"]["canonical_smiles"] for c in drug["candidates"]}
        if len(parents) != 1:
            raise RuntimeError(f"{drug['name']}: candidates differ in parent structure {parents}")
        smiles[drug["name"]] = parents.pop()
    graph = dict(pd.read_csv(allowed(GRAPH_DRUGS)).set_index("drug_id")["canonical_smiles"])
    if len(smiles) != 38 or smiles != graph:
        raise RuntimeError("O'Neil SMILES must be the 38 audited parents used by the drug-graph cache")
    return {name: author_smiles(s) for name, s in sorted(smiles.items())}


def compare(recomputed: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    cosine = torch.nn.functional.cosine_similarity(recomputed, reference).item()
    return cosine, (recomputed - reference).abs().max().item()


def main() -> None:
    start = time.perf_counter()
    torch.manual_seed(0)  # feature-map buffers come from the checkpoint; seed only guards against unexpected randomness
    model = load_model()
    tokenizers = {rev: AutoTokenizer.from_pretrained(allowed(path), trust_remote_code=(rev == "7b12d946"))
                  for rev, path in TOKENIZER_DIRS.items()}
    table, author = nci_smiles()
    oneil = oneil_smiles()
    load_seconds = time.perf_counter() - start

    all_smiles = sorted(set(table["smiles_nci_sdf_raw"]) | set(table["smiles_parent"]) | set(oneil.values()))
    same_ids = {s: tokenizers["7b12d946"](s)["input_ids"] == tokenizers["361063d0"](s)["input_ids"] for s in all_smiles}
    unk = {rev: sum(tok.unk_token_id in tok(s)["input_ids"] for s in all_smiles) for rev, tok in tokenizers.items()}
    cache = {(rev, s): embed(model, tokenizers[rev], s) for rev in tokenizers for s in all_smiles}
    repeat = embed(model, tokenizers["7b12d946"], all_smiles[0])
    if not torch.equal(repeat, cache[("7b12d946", all_smiles[0])]):
        raise RuntimeError("Embedding is not deterministic across repeated forwards")

    variants = [(rev, src) for rev in tokenizers for src in NCI_SMILES_SOURCES]
    for rev, src in variants:
        stats = [compare(cache[(rev, s)], author[k]) for s, k in zip(table[f"smiles_{src}"], table["author_key"])]
        table[f"cos_{rev}_{src}"] = [c for c, _ in stats]
        table[f"maxabs_{rev}_{src}"] = [m for _, m in stats]
    summary = {}
    for rev, src in variants:
        cos = table[f"cos_{rev}_{src}"]
        worst = table.nsmallest(5, f"cos_{rev}_{src}")
        summary[f"{rev}|{src}"] = dict(
            median_cos=float(cos.median()), min_cos=float(cos.min()), n_below_0_99=int((cos < 0.99).sum()),
            max_abs_diff=float(table[f"maxabs_{rev}_{src}"].max()),
            worst5=[(k, round(float(c), 6)) for k, c in zip(worst["author_key"], worst[f"cos_{rev}_{src}"])],
            passes_card=bool(cos.median() >= 0.999 and cos.min() >= 0.99))
    # Tokenizer choice: better (median, min) cosine on the NCI author file, taken over the better SMILES source.
    best_rev, best_src = max(variants, key=lambda v: (summary[f"{v[0]}|{v[1]}"]["median_cos"],
                                                       summary[f"{v[0]}|{v[1]}"]["min_cos"]))

    OUT.mkdir(parents=True, exist_ok=True)
    oneil_embeddings = {name: cache[(best_rev, s)].clone() for name, s in oneil.items()}
    nci_embeddings = {k: cache[(best_rev, s)].clone() for k, s in zip(table["author_key"], table[f"smiles_{best_src}"])}
    torch.save(oneil_embeddings, OUT / "ONEIL_drug_molformer_embeddings.pt")
    torch.save(nci_embeddings, OUT / "NCI-ALMANAC_drug_molformer_embeddings_recomputed.pt")
    table.to_csv(OUT / "nci_author_comparison.csv", index=False, encoding="utf-8")
    pd.DataFrame({"drug_id": list(oneil), "smiles_isomeric_false": list(oneil.values())}).to_csv(
        OUT / "oneil_molformer_inputs.csv", index=False, encoding="utf-8")

    # O'Neil drugs whose model input equals an NCI author key's input: their embedding vs the author vector.
    by_smiles = dict(zip(table[f"smiles_{best_src}"], table["author_key"]))
    shared = {n: round(compare(oneil_embeddings[n], author[by_smiles[s]])[0], 6) for n, s in oneil.items() if s in by_smiles}
    record = dict(
        recipe=dict(model="ibm-research/MoLFormer-XL-both-10pct", code_config_revision="7b12d946c181a37f6012b9dc3b002275de070314",
                    weights_revision="361063d0ad524ef77cf39b08469f6be770dc550f",
                    weights_lfs_sha256="0795977fe7192c4acdaf052f0e8464af57bc4bb59211271c5e61aaba2637b9c6",
                    tokenizer_revision_chosen=best_rev, nci_smiles_source_chosen=best_src,
                    oneil_smiles_source="validated_identity.json candidates[].standard.canonical_smiles",
                    smiles_canonicalisation="RDKit MolToSmiles(canonical=True, isomericSmiles=False)",
                    deterministic_eval=True, pooling="mean of last_hidden_state over attention_mask (incl. <bos>/<eos>)",
                    batch_size=1, device="cpu", transformers=__import__("transformers").__version__,
                    torch=torch.__version__, rdkit=Chem.rdBase.rdkitVersion),
        counts=dict(nci_author_keys=len(table), oneil_drugs=len(oneil), distinct_smiles=len(all_smiles)),
        tokenizer_input_ids_identical=f"{sum(same_ids.values())}/{len(same_ids)}", unk_token_smiles=unk,
        nci_comparison=summary, oneil_shared_with_nci_cosine_to_author=shared,
        runtime_seconds=dict(load=round(load_seconds, 1), total=round(time.perf_counter() - start, 1)))
    (OUT / "summary.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

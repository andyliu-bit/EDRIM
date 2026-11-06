"""Reconstruct the comboKR 2.0 ALMANAC input kernels (not shipped by the authors). Run INSIDE the container.

Paper (Huusari et al. 2025, Experimental details / Table 2):
  drugs: MACCS fingerprints from isomeric SMILES, Tanimoto kernel (104/104 drugs)
  cells: gene expression from Julkunen et al. 2020 [ref 20], RBF kernel, "kernel parameter is set as mean of
         pairwise distances in the data" (60/60 cells)
Our reconstruction (deviations are recorded in kernel_build_audit.json):
  drugs: NCI ComboCompoundSet.sdf -> RDKit isomeric SMILES -> MACCS (167 bits) -> Tanimoto;
         fallback for unparsable records: NCI nsc_smiles.csv.
  cells: CellMiner NCI-60 RNA-seq composite expression (log2(FPKM+1)); the [ref 20] matrix is not available
         locally. RBF gamma = 1/(2*sigma^2), sigma = mean pairwise Euclidean distance (the same convention the
         author code uses for its output kernel). MDA-MB-468 has no column in this CellMiner matrix -> it gets
         an identity kernel row (k=1 with itself, 0 with others) and is flagged for a sensitivity analysis.
Outputs: /out/inputs/kernels.npz (Kc, Kd, cell_ids, drug_ids), /out/inputs/kernel_build_audit.json
"""
import io
import json
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import xlrd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel

RDLogger.DisableLog("rdApp.*")
Z = "/mnt/zenodo"
OUT = "/out/inputs"
cdd = np.load(f"{Z}/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
cell_ids = np.array(sorted(np.unique(cdd[:, 0]).tolist()))
drug_ids = np.array(sorted(np.unique(cdd[:, 1:]).tolist(), key=int))
audit = {"n_cells": int(len(cell_ids)), "n_drugs": int(len(drug_ids))}

# ---------------------------------------------------------------- drugs
sdf_smiles, sdf_bad = {}, []
supp = Chem.SDMolSupplier("/mnt/almanac/ComboCompoundSet.sdf", sanitize=True, removeHs=True)
raw_blocks = open("/mnt/almanac/ComboCompoundSet.sdf", encoding="latin-1").read().split("$$$$")
for k, mol in enumerate(supp):
    title = raw_blocks[k].strip("\n").split("\n")[0].strip() if k < len(raw_blocks) else None
    if mol is None:
        sdf_bad.append(title)
        continue
    name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else title
    sdf_smiles[name] = Chem.MolToSmiles(mol, isomericSmiles=True)
nsc_smi = pd.read_csv("/mnt/almanac/nsc_smiles.csv", dtype={"NSC": str, "SMILES": str})
nsc_smi = dict(zip(nsc_smi.NSC.str.strip(), nsc_smi.SMILES))
MANUAL_SMILES = {"119875": "N.N.Cl[Pt]Cl"}  # cisplatin; PubChem CID 84691 style, sanitizable by RDKit
fps, source = [], {}
for d in drug_ids:
    smi = sdf_smiles.get(d)
    src = "ComboCompoundSet.sdf"
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        for alt in [d, "753082" if d == "761431" else None]:
            if alt and alt in nsc_smi and isinstance(nsc_smi[alt], str):
                mol = Chem.MolFromSmiles(nsc_smi[alt])
                if mol is not None:
                    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                    src = f"nsc_smiles.csv:{alt}"
                    break
    if mol is None and d in MANUAL_SMILES:
        smi, src = MANUAL_SMILES[d], "manual(PubChem-style):" + MANUAL_SMILES[d]
        mol = Chem.MolFromSmiles(smi)
        audit.setdefault("manual_smiles_reason", {})[d] = {
            "sdf_parsed": d in sdf_smiles, "nsc_smiles_csv": nsc_smi.get(d),
            "reason": "RDKit cannot sanitize the NCI Pt coordination SMILES/SDF record"}
    if mol is None:
        raise SystemExit(f"no structure for NSC {d}")
    source[d] = src
    fps.append(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smi)))
Kd = np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])
audit["drug_structure_sources"] = pd.Series(source).value_counts().to_dict()
audit["drug_structure_fallbacks"] = {k: v for k, v in source.items() if v != "ComboCompoundSet.sdf"}
audit["sdf_unparsable_titles"] = sdf_bad
audit["Kd_offdiag_mean"] = float(Kd[~np.eye(len(Kd), dtype=bool)].mean())
audit["Kd_identical_pairs"] = int(((Kd > 0.999999) & ~np.eye(len(Kd), dtype=bool)).sum() // 2)
audit["Kd_min_eig"] = float(np.linalg.eigvalsh(Kd).min())

# ---------------------------------------------------------------- cells
z = zipfile.ZipFile("/mnt/cellminer/nci60_RNA__RNA_seq_composite_expression.zip")
book = xlrd.open_workbook(file_contents=z.read("output/RNA__RNA_seq_composite_expression.xls"))
sh = book.sheet_by_index(0)
rows = [sh.row_values(r) for r in range(sh.nrows)]
hdr_i = next(i for i, r in enumerate(rows) if isinstance(r[0], str) and r[0].startswith("Gene name"))
hdr = rows[hdr_i]
cell_cols = [j for j, x in enumerate(hdr) if isinstance(x, str) and re.match(r"^(BR|CNS|CO|LE|ME|LC|OV|PR|RE):", x)]
gene_col = 0
data = []
genes = []
for r in rows[hdr_i + 1:]:
    vals = [r[j] for j in cell_cols]
    if any(v == "" or v is None for v in vals):
        continue
    try:
        data.append([float(v) for v in vals])
        genes.append(r[gene_col])
    except ValueError:
        continue
X = np.array(data).T  # cells x genes
cm_names = [hdr[j].split(":", 1)[1] for j in cell_cols]
audit["cellminer_header_row"] = hdr_i
audit["cellminer_n_cells"] = len(cm_names)
audit["cellminer_n_genes_complete"] = int(X.shape[1])


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


alias = {"MDA-MB-231_ATCC": "MDA-MB-231", "NCI_ADR-RES": "NCI/ADR-RES"}
cm_idx = {norm(nm): j for j, nm in enumerate(cm_names)}
mapping, missing = {}, []
for c in cell_ids:
    key = norm(alias.get(c, c))
    if key in cm_idx:
        mapping[c] = cm_names[cm_idx[key]]
    else:
        missing.append(c)
audit["cell_mapping"] = mapping
audit["cells_without_expression"] = missing
keep = X[[cm_idx[norm(alias.get(c, c))] for c in cell_ids if c not in missing]]
keep = keep[:, keep.std(0) > 0]
sigma = float(np.mean(pairwise_distances(keep)))
gamma = 1.0 / (2.0 * sigma ** 2)
Kpart = rbf_kernel(keep, gamma=gamma)
Kc = np.eye(len(cell_ids))
idx = [i for i, c in enumerate(cell_ids) if c not in missing]
Kc[np.ix_(idx, idx)] = Kpart
audit["cell_rbf_sigma_mean_pairwise_distance"] = sigma
audit["cell_rbf_gamma"] = gamma
audit["cell_n_genes_used"] = int(keep.shape[1])
audit["Kc_offdiag_mean_mapped"] = float(Kpart[~np.eye(len(Kpart), dtype=bool)].mean())
audit["Kc_min_eig"] = float(np.linalg.eigvalsh(Kc).min())
np.savez(f"{OUT}/kernels.npz", Kc=Kc, Kd=Kd, cell_ids=cell_ids, drug_ids=drug_ids)
with open(f"{OUT}/kernel_build_audit.json", "w") as f:
    json.dump(audit, f, indent=1)
print(json.dumps({k: v for k, v in audit.items() if k != "cell_mapping"}, indent=1))

"""D10 C2 (host side): map the DeepSynBa public new-combination split onto the comboKR 2.0 author indices.

  train  : S1 split-1 matrices -> author cdd row (either drug order; reversed rows get swapped BRAID parameters inside
           the container). Matrices without an author BRAID fit are dropped and counted.
  query  : --role validation -> S1 split-2 matrices (9 combination doses each, row-major i, j as in the point table);
           --role test        -> the S5/D7 evaluable test frame (doses only; labels are not written).
Writes runs/model_improvement/combokr2_original_v1/public_split/{train,query_<role>}.npz.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/combokr2_original/prep_public_split.py --role validation
  .venv/Scripts/python.exe -I -B experiments/model_improvement/combokr2_original/prep_public_split.py --role test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ZEN = Path("C:/LHZ/research/bisurfgnn/data/raw/external_benchmarks/combokr2_zenodo_14509938")
OUT = ROOT / "runs/model_improvement/combokr2_original_v1/public_split"
sys.path.insert(0, str(HERE.parent / "input_parity"))
import run_parity as rp  # noqa: E402

S1 = rp.gate.S1
COLS = ["matrix_key", "deepsynba_cell", "drug_a_id", "drug_b_id", "row_i", "col_j", "dose_a_M", "dose_b_M"]


def author_cell(name: str) -> str:
    return name.replace("/", "_")


def nsc(drug_id: str) -> str:
    if not drug_id.startswith("S"):
        raise ValueError(drug_id)
    return drug_id[1:]


def matrices(frame: pd.DataFrame) -> pd.DataFrame:
    n = len(frame) // 9
    if len(frame) != 9 * n or not ((frame.row_i.to_numpy() == np.tile(np.repeat([1, 2, 3], 3), n)).all()
                                    and (frame.col_j.to_numpy() == np.tile([1, 2, 3], 3 * n)).all()):
        raise RuntimeError("Expected 9 consecutive row-major points per matrix")
    first = frame.iloc[::9].reset_index(drop=True)
    return pd.DataFrame({"matrix_key": first.matrix_key, "cell": first.deepsynba_cell.map(author_cell),
                         "d1": first.drug_a_id.map(nsc), "d2": first.drug_b_id.map(nsc),
                         "c1": list(frame.dose_a_M.to_numpy(float).reshape(n, 9)),
                         "c2": list(frame.dose_b_M.to_numpy(float).reshape(n, 9))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["validation", "test"])
    role = ap.parse_args().role
    OUT.mkdir(parents=True, exist_ok=True)
    cdd = np.load(ZEN / "almanac_cv_splits/almanac_cell_drug_drug_all.npy", allow_pickle=True).astype(str)
    index = {tuple(r): i for i, r in enumerate(cdd.tolist())}
    ker = np.load(ROOT / "runs/model_improvement/combokr2_original_v1/inputs/kernels.npz", allow_pickle=True)
    cells, drugs = set(ker["cell_ids"].astype(str)), set(ker["drug_ids"].astype(str))
    info = {}
    if not (OUT / "train.npz").exists():
        tr = matrices(pd.read_parquet(S1 / "points_train.parquet", columns=COLS))
        rows, flip, keep = [], [], []
        for k, (c, a, b) in enumerate(zip(tr.cell, tr.d1, tr.d2)):
            if (c, a, b) in index:
                rows.append(index[(c, a, b)]); flip.append(False); keep.append(k)
            elif (c, b, a) in index:
                rows.append(index[(c, b, a)]); flip.append(True); keep.append(k)
        np.savez(OUT / "train.npz", cdd_rows=np.array(rows), flipped=np.array(flip), matrix_key=tr.matrix_key.to_numpy()[keep].astype(str))
        info["train"] = dict(matrices=len(tr), mapped=len(rows), reversed=int(np.sum(flip)),
                             unique_author_rows=int(len(set(rows))))
    if role == "validation":
        q = matrices(pd.read_parquet(S1 / "points_validation.parquet", columns=COLS))
    else:
        sys.path.insert(0, str(HERE.parent / "comparator_test"))
        import common  # noqa: E402
        frame, _ = common.load_test()
        frame = frame.assign(deepsynba_cell=frame.deepsynba_cell)
        q = matrices(frame[COLS])
    ok = q.cell.isin(cells) & q.d1.isin(drugs) & q.d2.isin(drugs)
    if not ok.all():
        raise RuntimeError(f"{int((~ok).sum())} {role} matrices have a cell or drug outside the author kernels")
    np.savez(OUT / f"query_{role}.npz", cdd=np.stack([q.cell, q.d1, q.d2], 1).astype(str),
             c1=np.stack(q.c1), c2=np.stack(q.c2), matrix_key=q.matrix_key.to_numpy().astype(str))
    info[role] = dict(matrices=len(q))
    print(json.dumps(info, indent=1))
    (OUT / f"prep_{role}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

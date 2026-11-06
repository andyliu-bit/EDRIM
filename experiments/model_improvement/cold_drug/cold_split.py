"""D11 (card D): the new-drug split, built from S1 split 1 + split 2 only (split 3 is never read).

101 split-1 drugs, permuted with numpy default_rng(20261016): the first 15 are new-drug TEST drugs, the next 10 are
new-drug VALIDATION drugs. A matrix containing any test drug goes to cold_test; otherwise one containing any
validation drug goes to cold_val; every other matrix trains. Monotherapy anchors keep the S1 lookup values (measured
monotherapy of the drug aggregated across plates, information level "monotherapy measured"; see the card).

  .venv/Scripts/python.exe -I -B experiments/model_improvement/cold_drug/cold_split.py      (writes split.json once)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "input_parity"))
import run_parity as rp  # noqa: E402

gate = rp.gate
OUT = rp.V2 / "cold_drug"
SPLIT = OUT / "split.json"
SPLIT_SEED, N_TEST, N_VAL = 20261016, 15, 10
ROLES = ("train", "cold_val", "cold_test")


def assign(frame: pd.DataFrame, test_drugs: set, val_drugs: set) -> np.ndarray:
    a, b = frame.drug_a_id.to_numpy(), frame.drug_b_id.to_numpy()
    in_test = np.isin(a, list(test_drugs)) | np.isin(b, list(test_drugs))
    in_val = np.isin(a, list(val_drugs)) | np.isin(b, list(val_drugs))
    return np.where(in_test, "cold_test", np.where(in_val, "cold_val", "train"))


def load_frames(split_path: Path = SPLIT) -> dict[str, pd.DataFrame]:
    """S1 train+validation point frames re-assigned to the cold roles (source ids kept unique). split_path: D11 split by
    default; D16 passes an eval-paper partition's split.json."""
    split = json.loads(split_path.read_text(encoding="utf-8"))
    both = pd.concat([gate.load_points("train"), gate.load_points("validation")], ignore_index=True)
    role = assign(both, set(split["test_drugs"]), set(split["val_drugs"]))
    return {r: both[role == r].reset_index(drop=True) for r in ROLES}


def main():
    if SPLIT.exists():
        raise SystemExit(f"{SPLIT} already written")
    tr = gate.load_points("train")
    drugs = np.array(sorted(set(tr.drug_a_id) | set(tr.drug_b_id)))
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(drugs))
    test, val = sorted(drugs[perm[:N_TEST]]), sorted(drugs[perm[N_TEST:N_TEST + N_VAL]])
    both = pd.concat([tr, gate.load_points("validation")], ignore_index=True)
    role = assign(both, set(test), set(val))
    first = both.drop_duplicates("matrix_key")
    first_role = assign(first, set(test), set(val))
    counts = {r: dict(points=int((role == r).sum()), matrices=int((first_role == r).sum()),
                      pairs=int(first.pair_id[first_role == r].nunique())) for r in ROLES}
    tr_frame = both[role == "train"]
    seen = set(tr_frame.drug_a_id) | set(tr_frame.drug_b_id)
    if seen & (set(test) | set(val)):
        raise RuntimeError("Held-out drug in training")
    names = both.drop_duplicates("drug_a_id").set_index("drug_a_id").deepsynba_drug_a.to_dict()
    names.update(both.drop_duplicates("drug_b_id").set_index("drug_b_id").deepsynba_drug_b.to_dict())
    OUT.mkdir(parents=True, exist_ok=True)
    record = dict(seed=SPLIT_SEED, n_drugs=len(drugs), test_drugs=test, val_drugs=val,
                  test_drug_names=[names[d] for d in test], val_drug_names=[names[d] for d in val], counts=counts,
                  source="S1 points_train + points_validation; split 3 not read", registered="DEVIATIONS.md D11")
    with SPLIT.open("x", encoding="utf-8") as h:
        h.write(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

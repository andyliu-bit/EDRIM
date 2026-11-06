"""D14 H1: O'Neil point tables with NCI-style lookup anchors instead of same-batch interpolated ones.

Anchor for (cell, drug, dose) = mean over the train-role surfaces' anchors at that (cell, drug, dose) across batches and
partner drugs (each surface counted once, like NCI s1_prepare's split-1 means); fallback (drug, dose) mean, then the
global mean. Only train rows feed the lookup. Responses and every other column are unchanged.
Writes runs/oneil_v1/analysis/h1/data/points_{train,validation,test_inputs}.parquet; use with ONEIL_DATA_ROOT.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

common.install_guard()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC = common.OUT / "data"
OUT = common.ANALYSIS / "h1/data"
FILES = {"train": "points_train.parquet", "validation": "points_validation.parquet", "test": "points_test_inputs.parquet"}


def sides(frame):
    """Long table of (surface, cell, drug, dose key, anchor), one row per surface side and dose."""
    parts = [frame[["matrix_key", "cell_id", f"drug_{s}_id", f"dose_{s}_M", f"mono_{s}_ratio_raw"]]
             .set_axis(["matrix_key", "cell_id", "drug", "dose", "anchor"], axis=1) for s in "ab"]
    long = pd.concat(parts).drop_duplicates(["matrix_key", "drug", "dose"])
    long["dose_key"] = long["dose"].map(lambda d: f"{d:.6g}")
    return long


def main():
    frames = {r: pd.read_parquet(SRC / f) for r, f in FILES.items()}
    train = sides(frames["train"])
    by_cell = train.groupby(["cell_id", "drug", "dose_key"])["anchor"].mean()
    by_drug = train.groupby(["drug", "dose_key"])["anchor"].mean()
    overall = float(train["anchor"].mean())
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for role, frame in frames.items():
        frame = frame.copy()
        used = {"cell_drug_dose": 0, "drug_dose": 0, "global": 0}
        for s in "ab":
            keys = frame[f"dose_{s}_M"].map(lambda d: f"{d:.6g}")
            a = pd.Series(list(zip(frame["cell_id"], frame[f"drug_{s}_id"], keys))).map(by_cell).to_numpy()
            b = pd.Series(list(zip(frame[f"drug_{s}_id"], keys))).map(by_drug).to_numpy()
            used["cell_drug_dose"] += int(np.isfinite(a).sum())
            used["drug_dose"] += int((~np.isfinite(a) & np.isfinite(b)).sum())
            value = np.where(np.isfinite(a), a, np.where(np.isfinite(b), b, overall))
            used["global"] += int((~np.isfinite(a) & ~np.isfinite(b)).sum())
            frame[f"mono_{s}_ratio_raw"] = value
            frame[f"mono_{s}_percent"] = value * 100
            frame[f"mono_response_{s}"] = np.clip(value, 0.0, 3.0)
        frame.to_parquet(OUT / FILES[role], index=False)
        old = frames[role]
        report[role] = dict(rows=len(frame), sources=used,
                            mean_abs_change=float(np.mean(np.abs(frame["mono_response_a"] - old["mono_response_a"]))))
        assert len(frame) == len(old) and frame["matrix_key"].equals(old["matrix_key"])
        assert np.isfinite(frame[["mono_response_a", "mono_response_b"]]).all().all()
    print(report)


if __name__ == "__main__":
    main()

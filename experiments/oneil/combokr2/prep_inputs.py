"""O'Neil stage 2 comboKR 2.0 (DEVIATIONS D13 supplement, card E v2 "第二阶段"), host side: container inputs.

Writes runs/oneil_v1/combokr2/inputs/:
  mono_curves.csv        same-batch monotherapy curves (mean of finite replicates per dose, prepare_points.mono_curves,
                         i.e. the anchor rule) for every (batch, cell, drug) used by train, validation or test surfaces;
  surfaces_<role>.npz    cdd (unit, drug_a, drug_b), c1/c2 (n, 16) molar doses in the author's Fortran vectorisation
                         (k = (row_i-1) + 4*(col_j-1), so k=15 is the max-max dose), matrix_key;
                         train only: y (n, 16) = 100 * response (max_is_100). Validation/test responses are not written.
The comboKR "cell" is the unit batch|cell_id, so the author's (cell, drug) Hill lookup is the same-batch Hill.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/combokr2/prep_inputs.py
"""
import json
import sys
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

common.install_guard()
sys.path.insert(0, str(HERE.parent / "data"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import prepare_points as pp  # noqa: E402  (also installs its own guard; main() is not run)

OUT = common.OUT / "combokr2/inputs"


def unit(batch, cell):
    return f"{batch}|{cell}"


def surfaces(frame, with_y):
    f = frame.assign(k=(frame.row_i - 1) + 4 * (frame.col_j - 1)).sort_values(["surface_index", "k"])
    n = f.matrix_key.nunique()
    assert len(f) == 16 * n and (f.k.to_numpy() == np.tile(np.arange(16), n)).all()
    s = f.iloc[::16]
    out = dict(cdd=np.stack([[unit(b, c) for b, c in zip(s.batch, s.cell_id)], s.drug_a_id, s.drug_b_id], 1).astype(str),
               c1=f.dose_a_M.to_numpy(float).reshape(n, 16), c2=f.dose_b_M.to_numpy(float).reshape(n, 16),
               matrix_key=s.matrix_key.to_numpy().astype(str))
    if with_y:
        out["y"] = 100.0 * f.response.to_numpy(float).reshape(n, 16)
        assert np.isfinite(out["y"]).all()
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {r: common.load_points(r) for r in common.ROLES}
    report = {}
    for role, f in frames.items():
        z = surfaces(f, role == "train")
        np.savez(OUT / f"surfaces_{role}.npz", **z)
        report[role] = dict(surfaces=len(z["cdd"]), units=len(set(z["cdd"][:, 0])))
    ctx = {(b, raw, d, cid) for f in frames.values()
           for b, raw, cid, x, y in f[["batch", "deepsynba_cell", "cell_id", "drug_a_id", "drug_b_id"]]
           .drop_duplicates().itertuples(index=False) for d in (x, y)}
    curves = pp.mono_curves({c[:3] for c in ctx})
    rows = []
    for b, raw, d, cid in sorted(ctx):
        curve = curves[(b, raw, d)]
        assert curve, (b, raw, d)
        rows += [dict(unit=unit(b, cid), batch=b, cell_id=cid, drug=d, dose_M=float(dose * Decimal("1e-6")),
                      response_ratio=float(v)) for dose, v in sorted(curve.items())]
    mono = pd.DataFrame(rows)
    mono.to_csv(OUT / "mono_curves.csv", index=False)
    per = mono.groupby(["unit", "drug"]).size()
    report["mono"] = dict(contexts=int(len(per)), rows=len(mono), doses_per_context=per.value_counts().sort_index().to_dict(),
                          ratio_range=[float(mono.response_ratio.min()), float(mono.response_ratio.max())])
    (OUT / "prep_report.json").write_text(json.dumps(report, indent=1, default=int), encoding="utf-8")
    print(json.dumps(report, indent=1, default=int))


if __name__ == "__main__":
    main()

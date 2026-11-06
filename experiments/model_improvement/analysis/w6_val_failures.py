"""W6: validation-set failure analysis of Ours (D7 = phase2_long_both seeds 41-45) against XGBoost+identity (D8) and lookup HSA.

Read-only. Split 2 rows come only from gate.load_points("validation") (it refuses every role except train/validation);
split 1 is read only for pair overlap and six per-drug train means of (y - HSA). Predictions are the saved validation
files; nothing is trained and split 3 is never opened. Percent-point scale.
Errors are per seed, then averaged over the 5 seeds (RMSE = mean over seeds of the per-seed group RMSE; SSE share =
mean over seeds of group SSE / total SSE). The printed 4x4 surfaces use the seed-mean prediction for display only.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/analysis/w6_val_failures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/model_improvement/public_split_gate"))
import gate_common as gate  # noqa: E402

V2 = ROOT / "runs/model_improvement/bisurfgnn_v2_dev"
SEEDS = (41, 42, 43, 44, 45)
MIN_PAIR_CELLS = 90  # at least 10 surfaces (9 combination cells each)
LEVEL = {1: "低", 2: "中", 3: "高"}

sys.stdout.reconfigure(encoding="utf-8")
f = gate.load_points("validation")
y = f.response_percent.to_numpy(float)
hsa = f.hsa_lookup_percent.to_numpy(float)
assert np.allclose(hsa, np.minimum(f.mono_a_percent, f.mono_b_percent)) and np.allclose(y, 100 * f.response)

ours, xid = [], []
for s in SEEDS:
    p = pd.read_csv(V2 / f"phase2_long_both/long_both_seed_{s}/validation_predictions.csv").set_index("source_row")
    p = p.loc[f.source_combodrugseq.to_numpy()]
    assert np.allclose(p.target.to_numpy(), f.response.to_numpy())
    ours.append(100 * p.prediction.to_numpy(float))
    xid.append(100 * np.load(V2 / f"xgb_identity_final/seed_{s}/validation_predictions.npz")["prediction"].astype(float))
ours, xid = np.stack(ours), np.stack(xid)

# alignment check: per-seed pooled RMSE must reproduce the stored selection metrics
sel = {m["seed"]: m for m in json.loads((V2 / "xgb_identity_final/selection_summary.json").read_text("utf-8"))["seeds"]}
for i, s in enumerate(SEEDS):
    stored = json.loads((V2 / f"phase2_long_both/long_both_seed_{s}/metrics.json").read_text("utf-8"))
    assert abs(np.sqrt(np.mean((ours[i] - y) ** 2)) - stored["validation_pooled_rmse_percent"]) < 1e-3
    assert abs(np.sqrt(np.mean((xid[i] - y) ** 2)) - sel[s]["validation_pooled_rmse_percent"]) < 1e-3
SE = {"ours": (ours - y) ** 2, "xid": (xid - y) ** 2, "hsa": ((hsa - y) ** 2)[None]}
print(f"rows={len(f)} surfaces={f.matrix_key.nunique()} pairs={f.pair_id.nunique()} cells={f.cell_id.nunique()}")
print("pooled RMSE (mean of seeds): " + ", ".join(f"{m}={np.sqrt(e.mean(1)).mean():.3f}" for m, e in SE.items())
      + f"; Ours seed-mean prediction {np.sqrt(np.mean((ours.mean(0) - y) ** 2)):.3f}")


def by(keys, idx=None) -> pd.DataFrame:
    """Per-group n, RMSE and share of the total SSE (per seed, averaged), Ours mean signed error, mean (y - HSA).
    idx selects (possibly repeated) rows, e.g. every cell once per drug partner; shares stay relative to the full SSE."""
    idx = np.arange(len(y)) if idx is None else idx
    keys = pd.Series(np.asarray(keys), name="key")
    out = {"n": keys.value_counts()}
    for m, e in SE.items():
        g = pd.DataFrame(e[:, idx].T).groupby(keys)
        out[f"{m}_rmse"] = np.sqrt(g.mean()).mean(axis=1)
        out[f"{m}_share%"] = 100 * (g.sum() / e.sum(axis=1)).mean(axis=1)
    out["ours_bias"] = pd.Series((ours - y).mean(0)[idx]).groupby(keys).mean()
    out["y_minus_hsa"] = pd.Series((y - hsa)[idx]).groupby(keys).mean()
    out = pd.DataFrame(out)
    out["ours/hsa"] = out.ours_rmse / out.hsa_rmse
    return out


def show(title, t, rows=10):
    cols = ["n", "ours_rmse", "ours_share%", "xid_rmse", "hsa_rmse", "ours/hsa", "ours_bias", "y_minus_hsa"]
    print(f"\n## {title}\n" + t[cols].head(rows).to_markdown(floatfmt=".2f"))


pair_name = np.where(f.deepsynba_drug_a < f.deepsynba_drug_b, f.deepsynba_drug_a + " + " + f.deepsynba_drug_b,
                     f.deepsynba_drug_b + " + " + f.deepsynba_drug_a)
pairs = by(pair_name)
show("pairs by SSE share", pairs.sort_values("ours_share%", ascending=False))
show(f"pairs by RMSE (n >= {MIN_PAIR_CELLS})", pairs[pairs.n >= MIN_PAIR_CELLS].sort_values("ours_rmse", ascending=False))
print(f"pairs below n>={MIN_PAIR_CELLS}: {(pairs.n < MIN_PAIR_CELLS).sum()}; top-10 pairs hold "
      f"{pairs['ours_share%'].nlargest(10).sum():.1f}% of Ours SSE ({pairs.n[pairs['ours_share%'].nlargest(10).index].sum() / len(y):.1%} of cells)")
s_mean = pd.DataFrame({"y_h": y - hsa, "o_h": ours.mean(0) - hsa, "x_h": xid.mean(0) - hsa, "pair": pair_name}).groupby(f.matrix_key.to_numpy())
s_mean = s_mean.agg(y_h=("y_h", "mean"), o_h=("o_h", "mean"), x_h=("x_h", "mean"), pair=("pair", "first"))
for name in pairs.sort_values("ours_share%", ascending=False).index[:3]:
    s = s_mean[s_mean.pair == name]
    print(f"{name}: surfaces={len(s)} share with surface-mean y-HSA <= -10: {(s.y_h <= -10).mean():.0%}, >= +10: "
          f"{(s.y_h >= 10).mean():.0%}; mean over surfaces of (pred - HSA): Ours {s.o_h.mean():+.1f}, XGB+ID {s.x_h.mean():+.1f}, "
          f"truth {s.y_h.mean():+.1f}")

cells = by(f.deepsynba_cell)
show("cells by SSE share", cells.sort_values("ours_share%", ascending=False))
print("top-10 cells by RMSE:", ", ".join(cells.sort_values("ours_rmse", ascending=False).index[:10]))
show("panels", by(f.biological_cell_id.str.split(":").str[0]).sort_values("ours_rmse", ascending=False), rows=20)

lo, hi = np.minimum(f.row_i, f.col_j), np.maximum(f.row_i, f.col_j)  # order-free: drug A/B is alphabetical
show("dose level (lower, higher)", by([f"{LEVEL[a]}+{LEVEL[b]}" for a, b in zip(lo, hi)]).sort_index(), rows=20)
dev = np.abs(y - hsa)
show("|y - HSA| bins", by(np.select([dev < 3, dev < 10], ["a <3", "b 3-10"], "c >=10")).sort_index())
show(">=10 bin split by sign (y below HSA = more inhibition than the better monotherapy)",
     by(np.select([y - hsa <= -10, y - hsa >= 10], ["y-HSA <= -10", "y-HSA >= +10"], "rest")).drop("rest"))

# per drug as either partner: each cell counted once for each of its two drugs
rows2 = np.concatenate([np.arange(len(f))] * 2)
drugs = by(np.concatenate([f.deepsynba_drug_a, f.deepsynba_drug_b]), rows2).sort_values("ours_share%", ascending=False)
show("drugs by share of total SSE (cells involving the drug, either partner)", drugs)
print("pralatrexate:", drugs.loc["pralatrexate", ["n", "ours_rmse", "ours_share%", "xid_rmse", "hsa_rmse"]].round(3).to_dict(),
      f"rank by share {list(drugs.index).index('pralatrexate') + 1}, rank by RMSE "
      f"{list(drugs.sort_values('ours_rmse', ascending=False).index).index('pralatrexate') + 1} of {len(drugs)}")
prala = (f.deepsynba_drug_a == "pralatrexate") | (f.deepsynba_drug_b == "pralatrexate")
for label, keep in (("all", np.ones(len(y), bool)), ("without pralatrexate pairs", ~prala.to_numpy())):
    r = {m: np.sqrt(e[:, keep].mean(1)).mean() for m, e in SE.items()}
    print(f"{label}: " + ", ".join(f"{m}={v:.3f}" for m, v in r.items())
          + f"; Ours vs XGB+ID {r['ours'] / r['xid'] - 1:+.1%}, Ours vs HSA {r['ours'] / r['hsa'] - 1:+.1%}")
print(f"pralatrexate pairs={f.pair_id[prala].nunique()} cells={prala.sum()} ({prala.mean():.2%})")

# train context (split 1, labels used only for these descriptive means): pair overlap and per-drug mean (y - HSA)
t = gate.load_points("train")
print(f"\ntrain: pairs={t.pair_id.nunique()}, validation pairs also in train={len(set(f.pair_id) & set(t.pair_id))}, "
      f"(pair, cell) overlap={len(set(zip(f.pair_id, f.cell_id)) & set(zip(t.pair_id, t.cell_id)))}")
for d in ("Pentostatin", "Streptozocin", "Vincristine sulfate", "pralatrexate", "Mercaptopurine", "Romidepsin"):
    m = (t.deepsynba_drug_a == d) | (t.deepsynba_drug_b == d)
    dt = t.response_percent[m] - t.hsa_lookup_percent[m]
    print(f"train {d}: cells={m.sum()} pairs={t.pair_id[m].nunique()} mean(y-HSA)={dt.mean():+.2f} share<=-10={(dt <= -10).mean():.1%}")
del t

# worst surfaces by Ours surface RMSE (per seed, averaged), one per drug pair
sse_surf = pd.DataFrame(SE["ours"].T).groupby(f.matrix_key.to_numpy()).sum()
surf = (sse_surf / 9).pow(0.5).mean(axis=1).sort_values(ascending=False)
top1 = surf.index[: len(surf) // 100]
print("\nsurface RMSE quantiles (Ours):", surf.quantile([.5, .9, .99, 1]).round(2).to_dict(),
      f"| worst 1% surfaces ({len(top1)}) hold {100 * (sse_surf.loc[top1].sum() / SE['ours'].sum(1)).mean():.1f}% of Ours SSE")
seen = set()
for key in surf.index:
    rows = f.index[f.matrix_key == key]
    r0 = f.loc[rows[0]]
    if r0.pair_id in seen:
        continue
    seen.add(r0.pair_id)
    print(f"\n### {key}  Ours {surf[key]:.2f}  XGB+ID {np.sqrt(SE['xid'][:, rows].mean(1)).mean():.2f}  "
          f"HSA {np.sqrt(SE['hsa'][:, rows].mean()):.2f}  pair Ours RMSE {pairs.loc[pair_name[rows[0]], 'ours_rmse']:.2f}")
    grid = pd.DataFrame("", index=[f"A0", "A1", "A2", "A3"], columns=["B0", "B1", "B2", "B3"])
    grid.loc["A0", "B0"] = "100"
    for i in rows:
        a, b = f.row_i[i], f.col_j[i]
        grid.iloc[a, 0], grid.iloc[0, b] = f"{f.mono_a_percent[i]:.1f}", f"{f.mono_b_percent[i]:.1f}"
        grid.iloc[a, b] = f"{y[i]:.1f} / {ours[:, i].mean():.1f} / {xid[:, i].mean():.1f} / {hsa[i]:.1f}"
    print(f"A={r0.deepsynba_drug_a} doses {sorted(set(f.dose_a_M[rows]))}  B={r0.deepsynba_drug_b} doses {sorted(set(f.dose_b_M[rows]))}")
    print("cells: y / Ours (seed mean) / XGB+ID (seed mean) / HSA; per-seed Ours surface RMSE "
          + ", ".join(f"{np.sqrt(SE['ours'][k, rows].mean()):.1f}" for k in range(len(SEEDS))))
    print(grid.to_markdown())
    if len(seen) == 5:
        break

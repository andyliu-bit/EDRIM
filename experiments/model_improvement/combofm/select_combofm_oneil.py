"""D21: tuning selection for the O'Neil comboFM runs (same rules as select_combofm.py; val surface-macro, %).

  --pick-reg   best reg among the rank-50, seed-41, 100-epoch runs (ties -> larger reg)
  --pick-best  best (reg, rank) among all 7 seed-41 tuning runs, printed as "<reg> <rank>"
"""
import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "runs/model_improvement/combofm/oneil"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pick-reg", action="store_true")
    g.add_argument("--pick-best", action="store_true")
    a = ap.parse_args()
    t = [json.loads(m.read_text()) for m in sorted(OUT.glob("reg*_rank*/seed_41_ep100/metrics.json"))]
    key = lambda r: (r["validation_surface_macro_percent"], -r["reg"], r["rank"])
    if a.pick_reg:
        regs = [r for r in t if r["rank"] == 50]
        assert len(regs) == 4, f"expected 4 rank-50 runs, found {len(regs)}"
        print(f"{min(regs, key=key)['reg']:g}")
    else:
        assert len(t) == 7, f"expected 7 tuning runs, found {len(t)}"
        best = min(t, key=key)
        print(f"{best['reg']:g} {best['rank']}")


if __name__ == "__main__":
    main()

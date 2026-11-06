"""D9 (card A), validation part: the 10-13 synergy check re-run with BiSurfGNN v2 (D7 final models, seeds 41-45)
in place of the gate BiSurfGNN. Same pre-declaration (hash-checked), metrics, bootstrap and guards; only the BiSurfGNN
prediction paths and the output directory change. Split 3 is never opened.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/synergy_check_v2/run_v2.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "synergy_check"))
import synergy_check as sc  # noqa: E402

V2 = sc.ROOT / "runs/model_improvement/bisurfgnn_v2_dev/phase2_long_both"
sc.BISURF = {f"BS_s{s}": V2 / f"long_both_seed_{s}" for s in (41, 42, 43, 44, 45)}
sc.OUT = sc.ROOT / "runs/model_improvement/bisurfgnn_v2_dev/synergy_check_v2/validation"

if __name__ == "__main__":
    if "--out" not in sys.argv:
        sys.argv += ["--out", str(sc.OUT)]
    sc.main()

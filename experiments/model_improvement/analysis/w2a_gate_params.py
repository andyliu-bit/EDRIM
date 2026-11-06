"""W2a: dose-gate stats on train/validation combination cells + D7 parameter counts. Split 3 is never read
(run_gate.prepare -> gate.load_points refuses roles other than train/validation)."""
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path("C:/LHZ/research/bisurfgnn-model-experiments")
sys.path.insert(0, str(ROOT / "experiments/model_improvement/input_parity"))
import run_parity as rp  # noqa: E402

import numpy as np  # noqa: E402

run_gate, gate = rp.run_gate, rp.gate
cfg = rp.effective_config()
frames, banks, tables = run_gate.prepare(cfg)
assert set(frames) == {"train", "validation"}

print("== gate ==")
for role, f in frames.items():
    combo = f[(f.dose_a_model > 0) & (f.dose_b_model > 0)]
    g = (1 - np.exp(-2 * combo.dose_a_model.to_numpy(float))) * (1 - np.exp(-2 * combo.dose_b_model.to_numpy(float)))
    # same values through the model's own float32 path (table.doses -> baselines._observed_reference)
    d = tables[role].doses.numpy()
    m = (d[:, 0] > 0) & (d[:, 1] > 0)
    g32 = (1 - np.exp(-2 * d[m, 0])) * (1 - np.exp(-2 * d[m, 1]))
    assert len(g32) == len(g), (len(g32), len(g))
    print(f"{role}: rows={len(f)} combo_cells={len(g)} min={g.min():.5f} q01={np.quantile(g, .01):.5f} "
          f"median={np.median(g):.5f} frac<0.99={(g < .99).mean():.4%} frac<0.9={(g < .9).mean():.4%} "
          f"| f32 min={g32.min():.5f} median={np.median(g32):.5f} | dose_model min={combo[['dose_a_model','dose_b_model']].values.min():.4f}")

print("== params (D7 = build_parity_model(arm 'both'), seed 41, CPU) ==")
model = rp.build_parity_model(banks, "both", 41, "cpu")
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"TOTAL {total}")
groups = OrderedDict()
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    parts = name.split(".")
    key = ".".join(parts[:2]) if parts[0] == "interaction_decoder" else parts[0]
    groups[key] = groups.get(key, 0) + p.numel()
for k, v in groups.items():
    print(f"{k}\t{v}\t{v / total:.2%}")
print("== top-level modules ==")
for n, c in model.named_children():
    print(n, type(c).__name__, sum(p.numel() for p in c.parameters()))

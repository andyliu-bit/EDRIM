"""BiSurfGNN v2 phase 1 random-vector control (protocol section 3): the 'both' arm with every drug's 768-d MoLFormer
row and every cell's 976-d expression row replaced by a fixed standard-normal random vector (same shape, per-dimension
standardised, RNG seed 20261014, identical across training seeds). Everything else equals the 'both' arm.
If this matches 'both', the gain is entity identity, not pretrained chemistry / expression content.

Imports run_parity.py and swaps its feature loader; the file itself is not modified.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/random_control/run_random.py --seed 41
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "input_parity"))
import run_parity as rp  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch is imported)

import argparse  # noqa: E402
import dataclasses  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

RANDOM_SEED = 20261014
OUT = rp.V2 / "phase1_random"
_real_loader = rp.load_deepsynba_features


def _standard(rng, shape):
    x = rng.standard_normal(shape)
    return torch.as_tensor(((x - x.mean(0)) / x.std(0)).astype(np.float32))


def random_features(*args, **kwargs):
    real = _real_loader(*args, **kwargs)
    rng = np.random.default_rng(RANDOM_SEED)
    drug = _standard(rng, tuple(real.drug.shape))
    cell = _standard(rng, tuple(real.cell.shape))
    record = dict(real.record, random_control={"rng_seed": RANDOM_SEED, "note": "real rows replaced by N(0,1), per-dim standardised"})
    return dataclasses.replace(real, drug=drug, cell=cell, record=record)


def main():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B`")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=(41, 42, 43, 44, 45), required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    output = OUT / f"{'smoke_' if a.smoke else ''}both_random_seed_{a.seed}"
    if output.exists():
        raise SystemExit(f"Output directory already exists: {output}")
    cfg = rp.effective_config()
    formal_max_epochs = cfg["training"]["max_epochs"]
    if a.smoke:
        cfg["training"]["max_epochs"] = 1
    output.mkdir(parents=True)
    rp.load_deepsynba_features = random_features
    args = argparse.Namespace(mode="smoke" if a.smoke else "train", arm="both", seed=a.seed, output_dir=output)
    rp.learned._write_json_atomic(output / "launch.json", {"run": "random-vector control of arm 'both'", "seed": a.seed,
                                                           "random_seed": RANDOM_SEED, "config": cfg})
    rp.run(args, cfg, formal_max_epochs, output)


if __name__ == "__main__":
    main()

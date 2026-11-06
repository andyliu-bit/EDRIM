"""D21 check (1) of docs/COMBOFM_CARD_20261027.md: the PyTorch port gives the same outputs and objective as tffm 1.0.1 (the PyPI release comboFM would have installed in 2020; 1.0.2 fails to import under TF 1.15).

Writes a random case (one-hot block + dense block, order 5, rank 6, reg 1e4), runs tffm_forward_tf1.py in the
tensorflow/tensorflow:1.15.5-py3 container (tffm installed from PyPI), and compares in float32: max |diff| of the
outputs <= 1e-5 x max(1, |output|) and of the objective <= 1e-5 relative. Needs Docker Desktop running.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/combofm/tffm_check/check_vs_tffm.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tffm_torch  # noqa: E402

IO = HERE.parents[3] / "runs/model_improvement/combofm/tffm_check"
DOCKER = r"C:\Users\user123\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe"
IMAGE = "tensorflow/tensorflow:1.15.5-py3"


def main():
    IO.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261027)
    n, n_onehot, order, rank, reg = 64, 12, 5, 6, 1e4
    onehot = np.zeros((n, n_onehot), np.float32)
    onehot[np.arange(n)[:, None], np.stack([rng.choice(n_onehot, 3, replace=False) for _ in range(n)])] = 1
    x = np.concatenate([onehot, rng.standard_normal((n, 9)).astype(np.float32)], 1)
    model = tffm_torch.TFFM(x.shape[1], order=order, rank=rank, init_std=0.3, seed=7)
    w = {f"w{k + 1}": model.w[k].detach().numpy() for k in range(order)}
    y = rng.standard_normal(n).astype(np.float32)
    np.savez(IO / "case.npz", x=x, y=y, order=order, rank=rank, reg=reg, b=np.float32(model.b.item()), **w)
    io = str(IO).replace("\\", "/")
    cmd = [DOCKER, "run", "--rm", "-v", f"{io}:/io", "-v", f"{str(HERE).replace(chr(92), '/')}:/code:ro", IMAGE,
           "bash", "-c", "pip install -q tffm==1.0.1 && python /code/tffm_forward_tf1.py"]
    subprocess.run(cmd, check=True)
    ref = np.load(IO / "tffm_out.npz")
    with torch.no_grad():
        xt = torch.from_numpy(x)
        ours = model(xt).numpy()
        target = float(((torch.from_numpy(y) - model(xt)) ** 2).mean() + reg * model.penalty())
        idx = torch.from_numpy(np.argwhere(onehot)[:, 1].reshape(n, 3))
        codes = model.forward_codes(idx, xt[:, n_onehot:]).numpy()
    out_diff = float(np.max(np.abs(ours - ref["outputs"]) / np.maximum(1, np.abs(ref["outputs"]))))
    codes_diff = float(np.max(np.abs(codes - ref["outputs"]) / np.maximum(1, np.abs(ref["outputs"]))))
    target_diff = abs(target - float(ref["target"])) / abs(float(ref["target"]))
    result = dict(image=IMAGE, tffm="1.0.1", n=n, order=order, rank=rank, reg=reg,
                  max_rel_output_diff=out_diff, max_rel_output_diff_forward_codes=codes_diff,
                  rel_objective_diff=target_diff, passed=bool(max(out_diff, codes_diff) <= 1e-5 and target_diff <= 1e-5))
    (IO / "check_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit("FAILED: the PyTorch port differs from tffm")


if __name__ == "__main__":
    main()

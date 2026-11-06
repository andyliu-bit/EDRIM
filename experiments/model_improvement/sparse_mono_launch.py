"""SparseMono-v1 configs, immutable snapshots and hidden launches (pattern of joint_feature_launch)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "runs/model_improvement/sparse_mono_v1"
CFG_DIR = ROOT / "configs/model_improvement/sparse_mono_v1"
PY = str((ROOT / ".venv/Scripts/python.exe").resolve())
BASE = dict(
    protocol="SparseMono-v1", claim_scope="engineering_pilot_no_performance_ranking",
    reference_run=str((ROOT / "runs/model_improvement/comparator_gpu_v2").resolve()).replace("\\", "/"),
    data_root=str((ROOT.parent / "bisurfgnn").resolve()).replace("\\", "/"),
    benchmark_config="configs/benchmark_bisurf_unclipped.yaml", cell_config="configs/cell_context.yaml",
    fold=0, sampling_seed=41, sampling_salt="model-comparator-gpu-v2", preserve_partner_cell_support=True,
    train_max_points=18000, validation_max_points=9000, device="cuda", threads=2,
    seeds=[41, 42], budgets=[1], pipelines=["A_ref", "B_xgb", "C_tabm", "D_gnn_g1", "D_gnn_g2"], oof_folds=4,
    xgb_prior=dict(eta=0.05, depth=6, trees=600, threads=6),
    xgb_correction=dict(eta=0.05, depth=4, trees=300, threads=6),
    xgb_head=dict(eta=0.03, depth=10, min_child=1, max_trees=3000, eval_every=50, threads=6),
    net_prior=dict(g1=dict(hidden=64, dropout=0.1, lr=1e-3, weight_decay=1e-4, steps=3000, batch=512),
                   g2=dict(hidden=128, dropout=0.3, lr=1e-3, weight_decay=1e-3, steps=3000, batch=512)),
    mlp_correction=dict(hidden=64, dropout=0.1, lr=1e-3, weight_decay=1e-4, steps=2000, batch=512),
    tabm_head=dict(k=32, width=256, blocks=2, dropout=0.1, lr=1e-3, weight_decay=1e-4, steps=6000, batch_surfaces=64, eval_every=100),
)


def merge(b, u):
    o = copy.deepcopy(b)
    for k, v in u.items():
        o[k] = merge(o[k], v) if isinstance(v, dict) and isinstance(o.get(k), dict) else v
    return o


def write_config(name, overrides):
    CFG_DIR.mkdir(parents=True, exist_ok=True); p = CFG_DIR / f"{name}.yaml"
    if p.exists():
        raise FileExistsError(p)
    cfg = merge(BASE, overrides); cfg["protocol"] = f"SparseMono-v1-{name}"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"); print(p)


def snapshot():
    files = [p for d in ("src", "experiments") for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts]
    files += list(CFG_DIR.glob("*.yaml")) + [ROOT / "docs/SPARSE_MONO_V1_PROTOCOL.md"]
    dg, rec = hashlib.sha256(), []
    for p in sorted(files):
        h = hashlib.sha256(p.read_bytes()).hexdigest(); r = p.relative_to(ROOT).as_posix(); dg.update(f"{r}:{h}\n".encode()); rec.append(dict(path=r, sha256=h))
    dest = SUITE / f"snap_{dg.hexdigest()[:12]}"
    if not dest.exists():
        for r in rec:
            t = dest / r["path"]; t.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / r["path"], t); t.chmod(0o444)
        (dest / "runs/model_improvement").mkdir(parents=True, exist_ok=True)
        import importlib.metadata as md
        deps = {p: md.version(p) for p in ("torch", "numpy", "pandas", "tabm", "rtdl_num_embeddings", "scikit-learn", "xgboost")}
        (dest / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(dict(files=rec, dependencies=deps), indent=1), encoding="utf-8")
    print(dest); return dest


def launch(snap, stages):
    snap = Path(snap).resolve(); log = SUITE / "launchers"; log.mkdir(parents=True, exist_ok=True); lines = []
    for s in stages:
        out = snap / "runs/model_improvement" / s
        lines += [f'"{s} start $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 "{log}/queue.log"',
                  f'$p = Start-Process "{PY}" -WindowStyle Hidden -PassThru -Wait -WorkingDirectory "{snap}" '
                  f'-RedirectStandardOutput "{log}/{s}.stdout.log" -RedirectStandardError "{log}/{s}.stderr.log" '
                  f'-ArgumentList "experiments/model_improvement/run_sparse_mono.py --config configs/model_improvement/sparse_mono_v1/{s}.yaml --output-dir {out}"',
                  f'"{s} exit=$($p.ExitCode) $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 "{log}/queue.log"']
    script = log / f"queue_{int(time.time())}.ps1"; script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Start-Process powershell -WindowStyle Hidden -PassThru "
                        f"-ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"{script}\"').Id"], capture_output=True, text=True, check=True)
    print("queue pid", int(r.stdout.strip().splitlines()[-1]))


if __name__ == "__main__":
    {"config": lambda: write_config(sys.argv[2], json.loads(sys.argv[3])), "snapshot": snapshot,
     "launch": lambda: launch(sys.argv[2], sys.argv[3:])}[sys.argv[1]]()

"""Write NumericSurface-v1 configs and launch stages from an immutable code snapshot.

  python numeric_surface_launch.py config <name> '<json overrides>'
  python numeric_surface_launch.py snapshot              -> prints snapshot dir
  python numeric_surface_launch.py launch <snapshot> <stage> [<stage> ...]   (sequential, hidden, detached)
"""
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
SUITE = ROOT / "runs/model_improvement/numeric_surface_v1"
CFG_DIR = ROOT / "configs/model_improvement/numeric_surface_v1"
PY = str((ROOT / ".venv/Scripts/python.exe").resolve())

BASE = dict(
    protocol="NumericSurface-v1", protocol_document="docs/NUMERIC_SURFACE_V1_PROTOCOL.md",
    claim_scope="engineering_pilot_no_performance_ranking",
    reference_run=str((ROOT / "runs/model_improvement/comparator_gpu_v2").resolve()).replace("\\", "/"),
    data_root=str((ROOT.parent / "bisurfgnn").resolve()).replace("\\", "/"),
    benchmark_config="configs/benchmark_bisurf_unclipped.yaml", cell_config="configs/cell_context.yaml",
    fold=0, seeds=[41, 42], sampling_seed=41, sampling_salt="model-comparator-gpu-v2",
    preserve_partner_cell_support=True, train_max_points=18000, validation_max_points=9000,
    device="cuda", threads=2, max_wall_seconds=10800,
    training=dict(epochs=0, mono_epochs=0, batch_size=512, evaluation_batch_size=256, batch_surfaces=128,
                  learning_rate=0.0003, weight_decay=0.0001, gradient_clip=1.0, min_delta=0.00001),
    numeric_arch=dict(ple=dict(d_embedding=16, arm_width=128, width=256, blocks=3, dropout=0.1),
                      tabm=dict(d_embedding=16, k=32, width=256, blocks=2, dropout=0.1)),
    numeric_opt={a: dict(o1=dict(lr=1e-3, weight_decay=1e-4, batch_surfaces=64, gradient_clip=1.0, eval_every=100, steps=0),
                         o2=dict(lr=3e-4, weight_decay=1e-4, batch_surfaces=64, gradient_clip=1.0, eval_every=100, steps=0))
                 for a in ("ple", "tabm")},
    graph=dict(hidden=64, layers=2, dropout=0.1),
    stage2=dict(lr=1e-3, weight_decay=1e-4, batch_surfaces=64, gradient_clip=1.0, eval_every=100, steps=0, crossfit_folds=4),
    stage3=dict(lr=1e-4, weight_decay=1e-4, batch_surfaces=64, gradient_clip=1.0, eval_every=100, steps=0),
    numeric_source_runs={},
)


def merge(base, update):
    out = copy.deepcopy(base)
    for k, v in update.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def write_config(name, overrides):
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{name}.yaml"
    if path.exists():
        raise FileExistsError(f"Frozen config exists: {path}")
    cfg = merge(BASE, overrides); cfg["protocol"] = f"NumericSurface-v1-{name}"; cfg["report_prefix"] = f"numsurf_v1_{name}"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"); print(path)


def snapshot():
    files = [p for d in ("src", "experiments") for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts]
    files += list(CFG_DIR.glob("*.yaml")) + list((ROOT / "data/raw_manifests").glob("*"))
    files += [ROOT / "docs/NUMERIC_SURFACE_V1_PROTOCOL.md"]
    digest = hashlib.sha256()
    records = []
    for p in sorted(files):
        h = hashlib.sha256(p.read_bytes()).hexdigest(); rel = p.relative_to(ROOT).as_posix()
        digest.update(f"{rel}:{h}\n".encode()); records.append(dict(path=rel, sha256=h))
    tag = digest.hexdigest()[:12]
    dest = SUITE / f"snap_{tag}"
    if not dest.exists():
        for r in records:
            target = dest / r["path"]; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / r["path"], target)
            target.chmod(0o444)
        (dest / "runs/model_improvement").mkdir(parents=True, exist_ok=True)
        (dest / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                     files=records, tag=tag), indent=1), encoding="utf-8")
    print(dest)
    return dest


def launch(snap, stages):
    snap = Path(snap).resolve(); log = SUITE / "launchers"; log.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in stages:
        out = snap / "runs/model_improvement" / s
        lines.append(f'"{s} start $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 "{log}/queue.log"')
        lines.append(f'$p = Start-Process "{PY}" -WindowStyle Hidden -PassThru -Wait -WorkingDirectory "{snap}" '
                     f'-RedirectStandardOutput "{log}/{s}.stdout.log" -RedirectStandardError "{log}/{s}.stderr.log" '
                     f'-ArgumentList "experiments/model_improvement/run_numeric_surface.py --config '
                     f'configs/model_improvement/numeric_surface_v1/{s}.yaml --output-dir {out}"')
        lines.append(f'"{s} exit=$($p.ExitCode) $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 "{log}/queue.log"')
    script = log / f"queue_{int(time.time())}.ps1"; script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # A detached Popen of powershell died silently on this machine; Start-Process survives the caller.
    started = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"(Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList "
                              f"'-NoProfile -ExecutionPolicy Bypass -File \"{script}\"').Id"],
                             capture_output=True, text=True, check=True)
    pid = int(started.stdout.strip().splitlines()[-1])
    (log / f"{script.stem}_process.json").write_text(json.dumps(dict(pid=pid, snapshot=str(snap), stages=stages,
                                                                     started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))), encoding="utf-8")
    print("queue pid", pid)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "config":
        write_config(sys.argv[2], json.loads(sys.argv[3]))
    elif cmd == "snapshot":
        snapshot()
    elif cmd == "launch":
        launch(sys.argv[2], sys.argv[3:])

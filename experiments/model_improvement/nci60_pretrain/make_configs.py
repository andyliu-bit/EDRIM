"""NCI60-Pretrain-v1 config writer (configs live in experiments/model_improvement/nci60_pretrain/configs/).

  python make_configs.py pretrain <max_steps> [name]   -> configs/<name>.yaml (input + exclusion hashes, schedule)
  python make_configs.py transfer <name> '<jobs-json>' -> configs/transfer_<name>.yaml
     base = SparseMono-v1 stab.yaml (identical data, sample, D settings); adds the pretrained checkpoints + hashes,
     the explicit job list and the reproduction-control references.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from bisurfgnn.pretrain.nci60 import sha256_file  # noqa: E402

HERE = ROOT / "experiments/model_improvement/nci60_pretrain/configs"
RUN = "runs/model_improvement/nci60_pretrain_v1"
DATA = "data/interim/nci60_pretrain_v1"
SM = "runs/model_improvement/sparse_mono_v1"
CONTROL_REFERENCE = {"D_gnn_g1": f"{SM}/snap_921bdc09d9a4/runs/model_improvement/screen1",
                     "D_gnn_g2": f"{SM}/snap_921bdc09d9a4/runs/model_improvement/screen1",
                     "D_morgan_g2": f"{SM}/snap_1e79403e10f6/runs/model_improvement/screen2"}
SCHEDULE = dict(eval_every=2000, patience_evals=10, min_delta=1.0e-5, time_limit_minutes=80, max_steps=None)


def write(path: Path, cfg: dict):
    HERE.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"); print(path)


def pretrain(max_steps: int, name: str = "pretrain"):
    frz = json.loads((ROOT / RUN / "freeze/exclusion_freeze.json").read_text(encoding="utf-8"))
    cfg = dict(protocol="NCI60-Pretrain-v1-pretrain", device="cuda", threads=2, seed=20261012,
               sparse_mono_config="configs/model_improvement/sparse_mono_v1/stab.yaml",
               points=f"{DATA}/nci60_points.parquet", graphs=f"{DATA}/nci60_graphs.pt", compounds=f"{DATA}/nci60_compounds.parquet",
               exclusion_list=f"{RUN}/freeze/exclusion_list.csv", exclusion_freeze=f"{RUN}/freeze/exclusion_freeze.json",
               exclusion_sha256=frz["exclusion_list_sha256"], holdout_salt="NCI60-Pretrain-v1|pretrain-holdout", holdout_modulus=20,
               schedule=dict(SCHEDULE, max_steps=int(max_steps)))
    for k in ("points", "graphs", "compounds"):
        cfg[f"{k}_sha256"] = sha256_file(ROOT / cfg[k])
    write(HERE / f"{name}.yaml", cfg)


def transfer(name: str, jobs: list):
    cfg = copy.deepcopy(yaml.safe_load((ROOT / "configs/model_improvement/sparse_mono_v1/stab.yaml").read_text(encoding="utf-8")))
    cfg["protocol"] = f"NCI60-Pretrain-v1-transfer-{name}"
    cfg["claim_scope"] = "development_evidence_only"
    cfg.pop("seeds"); cfg.pop("budgets"); cfg.pop("pipelines")
    cfg["pretrained"] = {}
    for g in ("g1", "g2"):
        p = ROOT / RUN / f"pretrain_{g}" / "pretrained_encoders.pt"
        if p.exists():
            cfg["pretrained"][g] = dict(path=f"{RUN}/pretrain_{g}/pretrained_encoders.pt", sha256=sha256_file(p))
    cfg["jobs"] = jobs
    cfg["control_reference"] = CONTROL_REFERENCE
    write(HERE / f"transfer_{name}.yaml", cfg)


if __name__ == "__main__":
    {"pretrain": lambda: pretrain(int(sys.argv[2]), *sys.argv[3:4]), "transfer": lambda: transfer(sys.argv[2], json.loads(sys.argv[3]))}[sys.argv[1]]()

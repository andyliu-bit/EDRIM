"""Engineering check: PretrainedNetPrior.fit is NetPrior.fit except for the encoder initialisation.

With a checkpoint holding exactly the encoder weights NetPrior.fit would draw for seed s, both fits must give
bit-identical predictions; with the real pretrained checkpoint they must differ. Writes
runs/model_improvement/nci60_pretrain_v1/checks/transfer_equivalence_<g>.json.
  python check_transfer_equivalence.py <transfer_config.yaml> <g1|g2> [<g1|g2> ...]
"""
import json, sys, time
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments/model_improvement/nci60_pretrain"))
import run_transfer as rt  # noqa: E402
import numpy as np, pandas as pd, torch, yaml  # noqa: E402

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")); archs = sys.argv[2:]
out = ROOT / "runs/model_improvement/nci60_pretrain_v1/checks"; out.mkdir(parents=True, exist_ok=True)
torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(cfg["threads"])
orig = pd.read_parquet
def guard(path, *a, **k):
    if Path(path).name.lower() in {"test.parquet", "points.parquet"}:
        raise RuntimeError("Forbidden")
    return orig(path, *a, **k)
t0 = time.monotonic()
with patch.object(pd, "read_parquet", guard):
    ctx = rt.sm.Context(cfg)
mono = ctx.mono; seed = 41
base_cfg = dict(ctx.cfg)
for g in archs:
    ctx.cfg = dict(base_cfg); rt._CKPT.clear()
    c = cfg["net_prior"][g]
    rt.sm.learned._seed_everything(seed)
    init = rt.sm.MonoPriorNet("gnn", ctx.banks.atom_feature_dim, ctx.banks.cell_feature_dim, hidden=c["hidden"], dropout=c["dropout"],
                              dose_mean=float(mono.dose.mean()), dose_sd=float(mono.dose.std()), mono_mean=float(mono.mono.mean()))
    ident = out / f"identity_init_{g}.pt"
    torch.save(dict(drug_encoder=init.drug_encoder.state_dict(), cell_encoder=init.cell_encoder.state_dict(),
                    meta=dict(hidden=c["hidden"], arch=g)), ident)
    q = mono.sample(2000, random_state=0)
    base = rt.sm.NetPrior("gnn", g).fit(mono, ctx, seed).predict(q.drug, q.cell_id, q.dose)
    ctx.cfg = dict(base_cfg, pretrained={g: dict(path=str(ident.relative_to(ROOT)), sha256=rt.sha256_file(ident))})
    same = rt.PretrainedNetPrior(g).fit(mono, ctx, seed).predict(q.drug, q.cell_id, q.dose)
    ctx.cfg = dict(base_cfg, pretrained=cfg["pretrained"]); rt._CKPT.clear()
    pt = rt.PretrainedNetPrior(g).fit(mono, ctx, seed).predict(q.drug, q.cell_id, q.dose)
    res = dict(arch=g, seed=seed, device=cfg["device"], pretrained_checkpoint=cfg["pretrained"][g],
               identity_init_max_abs_diff=float(np.max(np.abs(base - same))),
               pretrained_vs_random_max_abs_diff=float(np.max(np.abs(base - pt))),
               train_mono_rmse_random_init=float(np.sqrt(np.mean((base - q.mono.to_numpy()) ** 2))),
               train_mono_rmse_pretrained_init=float(np.sqrt(np.mean((pt - q.mono.to_numpy()) ** 2))), seconds=time.monotonic() - t0)
    res["passed"] = res["identity_init_max_abs_diff"] == 0.0 and res["pretrained_vs_random_max_abs_diff"] > 0
    (out / f"transfer_equivalence_{g}.json").write_text(json.dumps(res, indent=1), encoding="utf-8"); ident.unlink()
    print(json.dumps(res, indent=1), flush=True)

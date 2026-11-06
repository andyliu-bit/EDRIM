"""DeepSynBa on O'Neil (card E v2, DEVIATIONS D13), two methods:
  deepsynba_4x4  author's O'Neil usage: 4x4 combination-only matrices, author model code (author_n/ with n = 4 is
                 bit-identical to commit 43c16961, see test_grid_n.py)
  deepsynba_5x5  4x4 combination cells + interpolated same-batch monotherapy edges + control 100 at (0,0); the edges are
                 training/validation-loss targets only, never inputs; scoring uses only the 16 cells [1:, 1:]

  full run (one per method and seed 41-45; refuses if its prediction files exist):
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/deepsynba/run_deepsynba.py --method deepsynba_4x4 --seed 41
  smoke: set ONEIL_PRED_ROOT=<scratch>/preds and add --epochs 2 (logs then go to <scratch>/logs)

Mirrors the NCI repro v2 (bisurfgnn/experiments/audit_revision/deepsynba_repro_v2): config_v2.yaml grid point
(gene_channel 2048, emb_size 256), AdamW 1e-4 / wd 1e-5, LinearLR 1 -> 0.5 over 30 epochs (stepped per epoch), batch 128
shuffled with drop_last, grad-norm clip 5, fp32, author loss = MSE over the whole n x n matrix in %, 500 epochs.
Lightning (used by the author trainer.py and the NCI repro) is no longer installed, so the author TrainingModule's
training step is re-implemented below in plain PyTorch with the same semantics.
Selection: best validation surface-macro RMSE over the 16 combination cells (common.surface_macro_percent). The epoch the
author rule (min validation loss) would pick and the repro v2 G1/G2 dead-head checks are recorded in the meta JSON.
Inputs are built in memory from the train/validation point tables (no pickle, no test cache); test predictions come
from test inputs only (drug ids, cell, doses).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

common.install_guard()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

AUTHOR = common.OLD / "data/raw/external_benchmarks/deepsynba"          # commit 43c16961, read-only
CONFIG_V2 = common.OLD / "experiments/audit_revision/deepsynba_repro_v2/config_v2.yaml"
EPOCHS = 500
GRID = {"deepsynba_4x4": 4, "deepsynba_5x5": 5}
SURFACES = {"train": 10590, "validation": 3528, "test": 3624}          # card E v2
HEADS = ("h1_mean", "h2_mean", "alpha_mean")
INPUTS = ("d1", "d2", "gex", "x1", "x2")
BS = 128


def load_config(n):
    cfg = yaml.safe_load(CONFIG_V2.read_text(encoding="utf-8"))
    cfg["model_params"].update(gene_dim=968, dose_grid=n)              # D13: 968 genes
    return cfg


def build(role, n, emb, gex):
    """Author item tensors (NciAlmanacDataset.__getitem__ conventions) for every surface of one role.
    Rows = drug_a doses, cols = drug_b doses (drug_a < drug_b), ascending as row_i / col_j; 5x5 puts dose 0 at index 0."""
    pts = common.load_points(role)
    keys = np.sort(pts["matrix_key"].unique())
    if len(keys) != SURFACES[role] or len(pts) != 16 * len(keys):
        raise SystemExit(f"{role}: {len(keys)} surfaces / {len(pts)} points, card says {SURFACES[role]}")
    if not (pts["drug_a_id"] < pts["drug_b_id"]).all():
        raise SystemExit(f"{role}: drug order is not canonical")
    first = pts.drop_duplicates("matrix_key").set_index("matrix_key").loc[keys]
    s = pd.Index(keys).get_indexer(pts["matrix_key"])
    off = n - 4
    r, c = pts["row_i"].to_numpy() - 1 + off, pts["col_j"].to_numpy() - 1 + off
    x1, x2 = np.zeros((len(keys), n)), np.zeros((len(keys), n))
    x1[s, r], x2[s, c] = pts["dose_a_M"], pts["dose_b_M"]
    if not ((np.diff(x1, axis=1) > 0).all() and (np.diff(x2, axis=1) > 0).all()):
        raise SystemExit(f"{role}: doses do not rise with row_i / col_j")
    if not ((x1[s, r] == pts["dose_a_M"]).all() and (x2[s, c] == pts["dose_b_M"]).all()):
        raise SystemExit(f"{role}: dose not constant along a row / column")
    data = {
        "d1": torch.stack([emb[d] for d in first["drug_a_id"]]),
        "d2": torch.stack([emb[d] for d in first["drug_b_id"]]),
        "gex": torch.from_numpy(np.stack([np.expand_dims(gex[x].astype("float32"), axis=1) for x in first["cell_id"]])),
        "x1": torch.from_numpy(x1.astype("float32") * 1e6),            # M -> uM as the author dataset
        "x2": torch.from_numpy(x2.astype("float32") * 1e6),
        "points": pts[common.KEY + (["response"] if role != "test" else [])].copy(), "s": s, "r": r, "c": c,
    }
    if role != "test":
        y = np.full((len(keys), n, n), np.nan)
        y[s, r, c] = pts["response"].to_numpy() * 100                  # ratio -> %
        if n == 5:
            y[:, 0, 0] = 100.0
            y[s, r, 0] = pts["mono_response_a"].to_numpy() * 100
            y[s, 0, c] = pts["mono_response_b"].to_numpy() * 100
            # every point's edge must hold its own anchor (fails if an anchor were not constant along its row / column)
            assert (y[s, r, 0] == pts["mono_response_a"].to_numpy() * 100).all()
            assert (y[s, 0, c] == pts["mono_response_b"].to_numpy() * 100).all()
        assert np.isfinite(y).all() and (y[s, r, c] == pts["response"].to_numpy() * 100).all()
        data["y"] = torch.from_numpy(y.astype("float32"))
    assert all(data[k].dtype == torch.float32 for k in INPUTS) and len(keys) % BS != 1  # model squeezes the batch dim
    return data


def to_points(data, pred):
    """n x n % predictions -> one ratio-scale value per point (the 16 combination cells)."""
    return pred[data["s"], data["r"], data["c"]].astype(np.float64) / 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(GRID))
    ap.add_argument("--seed", required=True, type=int, choices=common.SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    method, seed, n = args.method, args.seed, GRID[args.method]
    if args.epochs != EPOCHS and "ONEIL_PRED_ROOT" not in os.environ:
        raise SystemExit("short (smoke) runs must set ONEIL_PRED_ROOT to a scratch folder")
    outs = [common.pred_path(method, seed, role) for role in ("validation", "test")]
    if any(p.exists() for p in outs):
        raise SystemExit(f"prediction file exists, refusing to overwrite: {[str(p) for p in outs if p.exists()]}")
    log_path = common.PRED.parent / "logs" / f"{method}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def say(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as h:
            h.write(line + "\n")

    sys.path.insert(0, str(HERE / "author_n"))
    from deepsynba import DeepSynBa  # copied author model, dose grid n parameterised
    from losses import mse_loss_new  # author loss (unchanged copy)

    t0 = time.time()
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    cfg = load_config(n)
    emb = torch.load(common.MOLFORMER, weights_only=True)
    g = pd.read_csv(common.GENE_CSV)
    if g.shape[0] != cfg["model_params"]["gene_dim"]:
        raise SystemExit(f"gene CSV has {g.shape[0]} genes")
    gex = {cell: g[cell].values for cell in g.columns[1:]}
    tr, va, te = (build(role, n, emb, gex) for role in ("train", "validation", "test"))
    for d in (tr, va):
        for k in INPUTS + ("y",):
            d[k] = d[k].to(dev)

    # lightning.seed_everything(seed) equivalent, then model init (same order as the NCI repro)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DeepSynBa(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    sch = torch.optim.lr_scheduler.LinearLR(opt, start_factor=cfg["scheduler"]["start_factor"],
                                            end_factor=cfg["scheduler"]["end_factor"],
                                            total_iters=cfg["scheduler"]["total_iters"])
    clip, steps = cfg["trainer_params"]["grad_norm_clip"], len(tr["d1"]) // BS
    assert cfg["trainer_params"]["batch_size"] == BS and cfg["trainer_params"]["precision"] == 32

    # repro v2 HeadMonitor: count validation samples whose h1 / h2 / alpha pre-activation is > 0
    alive = {}

    def hook(k):
        def f(mod, _inp, out):
            if not mod.training and k in alive:
                alive[k] += int((out.detach() > 0).sum())
        return f
    for k in HEADS:
        model.predictors[k].prediction_head[3].register_forward_hook(hook(k))

    @torch.no_grad()
    def predict(d):
        model.eval()
        return torch.cat([model(*(d[k][i:i + BS].to(dev) for k in INPUTS))["dose_response"]
                          for i in range(0, len(d["d1"]), BS)])

    say(f"{method} seed {seed}: n={n}, train/val/test {len(tr['d1'])}/{len(va['d1'])}/{len(te['d1'])} surfaces, "
        f"{steps} steps/epoch x {args.epochs} epochs, pred root {common.PRED}")
    hist, best, best_state, best_epoch = [], np.inf, None, None
    for epoch in range(args.epochs):
        te0 = time.time()
        model.train()
        perm = torch.randperm(len(tr["d1"])).to(dev)                    # DataLoader(shuffle=True, drop_last=True)
        losses = []
        for b in range(steps):
            idx = perm[b * BS:(b + 1) * BS]
            out = model(*(tr[k][idx] for k in INPUTS))
            loss = mse_loss_new(out["dose_response"], tr["y"][idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(loss.detach())
        lr = opt.param_groups[0]["lr"]
        sch.step()
        alive.update({k: 0 for k in HEADS})
        pred = predict(va)
        val_alive = {k: alive.pop(k) / len(va["d1"]) for k in HEADS}
        val_loss = float(((pred.double() - va["y"].double()) ** 2).mean())   # author validation loss (whole n x n)
        sm = common.surface_macro_percent(va["points"], to_points(va, pred.cpu().numpy()))
        if sm < best:
            best, best_epoch = sm, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, "train_loss": float(torch.stack(losses).mean()), "val_loss": val_loss,
               "val_rmse_all_cells": val_loss ** 0.5, "val_surface_macro_percent": sm, "lr": lr,
               **{f"val_alive_{k}": v for k, v in val_alive.items()}, "seconds": round(time.time() - te0, 2)}
        hist.append(row)
        say(f"epoch {epoch} train_loss {row['train_loss']:.3f} val_loss {val_loss:.3f} val_sm% {sm:.4f} "
            f"alive h1/h2/alpha {val_alive['h1_mean']:.3f}/{val_alive['h2_mean']:.3f}/{val_alive['alpha_mean']:.3f} "
            f"{row['seconds']}s" + (" *" if best_epoch == epoch else ""))

    model.load_state_dict(best_state)
    val_pred = to_points(va, predict(va).cpu().numpy())
    check = common.surface_macro_percent(va["points"], val_pred)
    assert abs(check - best) < 1e-3, (check, best)
    test_pred = to_points(te, predict(te).cpu().numpy())
    author_epoch = int(np.argmin([h["val_loss"] for h in hist]))
    g1 = all(h[f"val_alive_{k}"] > 0 for h in hist for k in HEADS)
    meta = {
        "method": method, "seed": seed, "grid_n": n, "epochs": args.epochs, "steps_per_epoch": steps,
        "total_steps": steps * args.epochs, "batch_size": BS,
        "surfaces": {r: len(d["d1"]) for r, d in (("train", tr), ("validation", va), ("test", te))},
        "selection": {"rule": "min validation surface-macro RMSE over the 16 combination cells (%)",
                      "epoch": best_epoch, "val_surface_macro_percent": best, "recomputed_after_reload": check},
        "author_rule": {"rule": "min validation loss (author ModelCheckpoint on val/dose_response_loss, whole n x n, %^2)",
                        "epoch": author_epoch, "val_loss": hist[author_epoch]["val_loss"],
                        "val_surface_macro_percent": hist[author_epoch]["val_surface_macro_percent"]},
        "dead_heads": {"G1_all_heads_alive_every_epoch": g1,
                       "G2_val_rmse_last_below_first": len(hist) >= 2 and hist[-1]["val_rmse_all_cells"] < hist[0]["val_rmse_all_cells"],
                       "min_val_alive": {k: min(h[f"val_alive_{k}"] for h in hist) for k in HEADS},
                       "selected_epoch_alive": {k: hist[best_epoch][f"val_alive_{k}"] for k in HEADS}},
        "runtime_s": round(time.time() - t0, 1),
        "median_epoch_seconds": float(np.median([h["seconds"] for h in hist])),
        "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1),
        "gpu_peak_reserved_mib": round(torch.cuda.max_memory_reserved(dev) / 2**20, 1),
        "config": cfg, "model_params_total": sum(p.numel() for p in model.parameters()),
        "inputs": {"molformer": str(common.MOLFORMER), "genes": str(common.GENE_CSV), "config_v2": str(CONFIG_V2),
                   "author_code": str(AUTHOR) + " @ 43c16961, copied to " + str(HERE / "author_n")},
        "training_loop": "plain PyTorch re-implementation of the author Lightning TrainingModule (Lightning not installed)",
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                     "numpy": np.__version__, "gpu": torch.cuda.get_device_name(dev)},
        "history": hist,
    }
    for role, d, p in (("validation", va, val_pred), ("test", te, test_pred)):
        frame = d["points"][common.KEY].copy()
        frame["prediction"] = p
        path = common.write_predictions(method, seed, role, frame, meta)
        say(f"wrote {path}")
    say(f"done: selected epoch {best_epoch} val_sm% {best:.4f}; author-rule epoch {author_epoch} "
        f"(val_sm% {hist[author_epoch]['val_surface_macro_percent']:.4f}); G1 {g1} G2 "
        f"{meta['dead_heads']['G2_val_rmse_last_below_first']}; {meta['runtime_s']}s, median epoch "
        f"{meta['median_epoch_seconds']}s, GPU peak alloc/reserved {meta['gpu_peak_allocated_mib']}/"
        f"{meta['gpu_peak_reserved_mib']} MiB")


if __name__ == "__main__":
    main()

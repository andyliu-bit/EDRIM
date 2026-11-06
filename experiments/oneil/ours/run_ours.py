"""Ours on O'Neil (launch card E v2, docs/ONEIL_COMPARISON_CARD_20261020.md; DEVIATIONS D13), method "ours".

The NCI final recipe, trained from scratch: run_parity.build_parity_model(arm "both") (ParityAdditiveCompactBiSurfGNN,
gate_common.MODEL_KWARGS) trained by run_long_both.train_long_both (60 epochs, cosine 3e-4 -> 3e-6, no early stop,
batches from public_split_gate/configs/gate.yaml, best validation surface-macro epoch). Both are imported, not copied.
Only the NCI inputs are replaced:
  * points: common.load_points + the gate_common.load_points ID columns; test is inputs only and gets a placeholder
    response 0.0 in memory (never scored; the test table is built only after training);
  * drug graphs: runs/oneil_v1/drug_features; cell graphs: runs/oneil_v1/cell_features/cell_graphs.pt;
  * MoLFormer: common.MOLFORMER, z-scored per dimension over train-role drugs (ddof 0) as load_deepsynba_features;
  * genes: build_cell_features.load_gene_rows, 968 dims, so the arm's cell_ext_dim is 968 instead of 976.
NCI-only bindings (REFERENCE epoch -1/0 bit-match, protocol/gate hash bindings, S1 mapping) are not used.

Full run (refuses if its prediction files exist; log tees to runs/oneil_v1/logs/ours_seed<S>.log):
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/ours/run_ours.py --seed 41
Smoke (needs ONEIL_PRED_ROOT pointing outside runs/oneil_v1/preds): --smoke tiny (0.5% train surfaces, 2 epochs)
or --smoke full (full data, 1 epoch, for timing). Validation and test predictions are always full size.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

common.install_guard()
ROOT = common.ROOT
sys.path.insert(0, str(ROOT / "experiments/model_improvement/long_schedule"))
sys.path.insert(0, str(HERE.parent / "drug_features"))
sys.path.insert(0, str(HERE.parent / "cell_features"))
import run_long_both as rlb  # noqa: E402  (imports run_parity first: CUBLAS_WORKSPACE_CONFIG before torch)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from build_cell_features import load_gene_rows  # noqa: E402
from build_drug_graphs import load_drug_graphs  # noqa: E402

rp, gate, run_gate, graph, learned = rlb.rp, rlb.gate, rlb.run_gate, rlb.rp.graph, rlb.learned
METHOD = "ours"                                  # card setting (60 epochs); --patience switches to "ours_e500"
SMOKE = {"tiny": 2, "full": 1}                                   # smoke kind -> epochs


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def load_frame(role):
    """common.load_points plus the ID columns gate_common.load_points adds for the compact point table."""
    frame = common.load_points(role)
    if role == "test":
        frame["response"] = 0.0                                  # placeholder, in memory only; never scored
    frame["source_combodrugseq"] = gate.ROLE_OFFSET[role] + np.arange(len(frame), dtype=np.int64)
    frame["surface_replicate_id"] = frame.matrix_key
    frame["prediction_group_id"] = frame.matrix_key
    frame["heldout_cluster_id"] = frame.pair_id
    frame["response_clipped_high"] = False
    return frame


def build_banks(frames):
    """GraphFeatureBanks over every drug and cell of the three roles (IDs only), O'Neil graph caches."""
    ids = lambda cols: sorted({v for f in frames.values() for c in cols for v in f[c]})  # noqa: E731
    drug_ids, cell_ids = ids(("drug_a_id", "drug_b_id")), ids(("cell_id",))
    mapping, _, drugs = load_drug_graphs(common.DRUG, drug_ids)
    cache = torch.load(common.CELL / "cell_graphs.pt", map_location="cpu", weights_only=True)
    cells = [graph._as_graph_item(cache[c], f"cell {c}") for c in cell_ids]
    biological = {c: b for f in frames.values() for c, b in zip(f.cell_id, f.biological_cell_id)}
    record = json.loads((common.CELL / "cell_features_record.json").read_text(encoding="utf-8"))
    drug_graphs = [drugs[d] for d in drug_ids]
    return graph.GraphFeatureBanks(
        fold=0, drug_ids=tuple(drug_ids), drug_structure_keys=tuple(mapping[d] for d in drug_ids),
        cell_ids=tuple(cell_ids), biological_cell_ids=tuple(biological[c] for c in cell_ids),
        drug_graphs=tuple(drug_graphs), cell_graphs=tuple(cells),
        drug_graph_hashes=tuple(graph._graph_hash(g) for g in drug_graphs),
        cell_graph_hashes=tuple(graph._graph_hash(g) for g in cells),
        context_state_hash=record["hallmark"]["state_hash"],
        source_assets={"drug_graphs": str(common.DRUG), "cell_graphs": str(common.CELL / "cell_graphs.pt")})


def external_rows(banks, train):
    """MoLFormer z-scored over train-role drugs (ddof 0, as load_deepsynba_features) and the 968 gene rows."""
    embeddings = torch.load(common.MOLFORMER, map_location="cpu", weights_only=True)
    raw = np.stack([embeddings[d].reshape(-1).numpy().astype(np.float64) for d in banks.drug_ids])
    train_drugs = set(train.drug_a_id) | set(train.drug_b_id)
    fit = [i for i, d in enumerate(banks.drug_ids) if d in train_drugs]
    mean, sd = raw[fit].mean(axis=0), raw[fit].std(axis=0, ddof=0)
    if raw.shape[1] != rp.DRUG_DIM or not np.isfinite(raw).all() or not (sd > 0).all():
        raise RuntimeError("MoLFormer rows must be finite 768-d with non-zero train SD")
    genes, cell = load_gene_rows(banks.cell_ids)
    if cell.shape != (len(banks.cell_ids), 968) or not bool(torch.isfinite(cell).all()):
        raise RuntimeError(f"Expected finite 968-gene rows, got {tuple(cell.shape)}")
    record = dict(molformer_fit_drugs=len(fit), molformer_bank_drugs=len(banks.drug_ids), genes=len(genes),
                  cells=len(banks.cell_ids))
    return SimpleNamespace(drug=torch.as_tensor(((raw - mean) / sd).astype(np.float32)), cell=cell), record


def random_rows(features, record):
    """--arm random (docs/ONEIL_ABLATION_CARD_20261104.md) = NCI B9: every drug and cell row replaced by a fixed
    standard-normal vector of the same shape, per-dimension standardised, with run_random's RNG seed and draw order."""
    sys.path.insert(0, str(ROOT / "experiments/model_improvement/random_control"))
    import run_random
    rng = np.random.default_rng(run_random.RANDOM_SEED)
    drug = run_random._standard(rng, tuple(features.drug.shape))
    cell = run_random._standard(rng, tuple(features.cell.shape))
    note = {"rng_seed": run_random.RANDOM_SEED, "note": "real rows replaced by N(0,1), per-dim standardised"}
    return SimpleNamespace(drug=drug, cell=cell), dict(record, random_control=note)


def patch_noanchor(banks, tables):
    """--arm noanchor (card 20261104) = NCI B11 via run_ablation.patch_builder: y = b + delta with r as an extra decoder
    input. It patches rp.build_parity_model, so the reload after training rebuilds the same architecture."""
    sys.path.insert(0, str(ROOT / "experiments/model_improvement/ablation"))
    import run_ablation
    run_ablation.patch_builder("B11", banks, tables)


def train_es(tables, banks, ext, cfg, seed, out, patience):
    """rlb.train_long_both plus early stopping (D13, 10-20 20:30): stop once `patience` epochs pass without a
    validation surface-macro improvement larger than min_delta; the best epoch's weights are restored as before."""
    device = torch.device(cfg["device"]); t = cfg["training"]; S = rlb.SCHEDULE
    model = rp.build_parity_model(banks, rlb.ARM, seed, cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if t["learning_rate"] != S["lr_start"]:
        raise RuntimeError("Gate learning rate differs from the registered schedule start")
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=S["max_epochs"], eta_min=S["lr_end"])
    tr, va = tables["train"], tables["validation"]
    start = time.monotonic()
    initial = rp.predict_ext(model, va, banks, ext, cfg)
    np.testing.assert_allclose(initial, np.minimum(va.mono_a.numpy(), va.mono_b.numpy()), atol=3e-6, rtol=3e-6)
    best = gate.surface_macro(va, initial); state, best_epoch = rlb.pilot.frozen_state(model), -1
    history = [{"epoch": -1, "surface_macro_rmse": best, "pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), initial)}]
    stopped_by, last = "max_epochs", 0.0
    for epoch in range(t["max_epochs"]):
        if time.monotonic() - start + last > S["wall_cap_seconds"]:
            stopped_by = "wall_cap"; break
        epoch_start = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        loss = rp.train_epoch_ext(model, optimizer, tr, banks, ext, cfg, seed, epoch, device)
        pred = rp.predict_ext(model, va, banks, ext, cfg)
        score = gate.surface_macro(va, pred)
        improved = score < best - t["min_delta"]
        if improved:
            best, state, best_epoch = score, rlb.pilot.frozen_state(model), epoch
        scheduler.step()
        last = time.monotonic() - epoch_start
        history.append({"epoch": epoch, "surface_macro_rmse": score, "pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), pred),
                        "train_batch_mean_loss": loss, "wall_seconds": last, "improved": improved, "learning_rate": lr})
        rlb.pilot.write_history(out, history, best_epoch=best_epoch, arm=rlb.ARM, device=str(device))
        print(f"LONG_BOTH seed={seed} epoch={epoch} lr={lr:.2e} val_surface_macro={score:.7f} "
              f"val_pooled={history[-1]['pooled_rmse_percent']:.4f} best={best:.7f} ({last:.0f}s)", flush=True)
        if epoch - best_epoch >= patience:
            stopped_by = "early_stopping"; break
    model.load_state_dict(state)
    prediction = rp.predict_ext(model, va, banks, ext, cfg)
    torch.save(state, out / "model.pt")
    return prediction, {"best_epoch": best_epoch, "stopped_by": stopped_by, "epochs_run": len(history) - 1,
                        "wall_seconds": time.monotonic() - start,
                        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=common.SEEDS, required=True)
    ap.add_argument("--smoke", choices=sorted(SMOKE), help="scratch-only engineering check; needs ONEIL_PRED_ROOT")
    ap.add_argument("--patience", type=int, help="D13 10-20 20:30 setting: 500 epochs + early stopping -> method ours_e500")
    ap.add_argument("--batch-surfaces", type=int, help="D14 H3a only (analysis runs): surfaces per batch, card value 128")
    ap.add_argument("--arm", choices=("random", "noanchor"), help="docs/ONEIL_ABLATION_CARD_20261104.md, validation only")
    a = ap.parse_args()
    if a.arm is not None:
        if a.patience is not None or a.batch_surfaces is not None:
            raise SystemExit("--arm runs use the card setting only (no --patience / --batch-surfaces)")
        arm_root = common.ANALYSIS / f"ablation_{a.arm}" / "preds"
        if not a.smoke and common.PRED.resolve() != arm_root.resolve():
            raise SystemExit(f"--arm {a.arm} writes to {arm_root}; set ONEIL_PRED_ROOT to it")
    global METHOD
    if a.patience is not None:
        METHOD = "ours_e500"
        rlb.SCHEDULE = dict(rlb.SCHEDULE, max_epochs=500, early_stopping=True, patience=a.patience)
    paths = [common.pred_path(METHOD, a.seed, r) for r in ("validation", "test")]
    # an empty ONEIL_PRED_ROOT makes common.PRED the cwd, so check the resolved folder, not the variable
    if a.smoke and (not os.environ.get("ONEIL_PRED_ROOT") or common.PRED.resolve().is_relative_to(ROOT)):
        raise SystemExit("--smoke needs ONEIL_PRED_ROOT set to a scratch folder outside the repository")
    if a.batch_surfaces is not None and not common.analysis_run():
        raise SystemExit("--batch-surfaces is a D14 analysis setting; set ONEIL_PRED_ROOT under runs/oneil_v1/analysis")
    if not a.smoke and not common.analysis_run() and common.PRED.resolve() != (common.OUT / "preds").resolve():
        raise SystemExit(f"Full runs write to runs/oneil_v1/preds only; unset ONEIL_PRED_ROOT (now {common.PRED})")
    if any(p.exists() for p in paths):
        raise SystemExit(f"Prediction file already exists, refusing to overwrite: {[str(p) for p in paths if p.exists()]}")
    out = paths[0].parent                                        # model.pt and training_progress.json beside the preds
    out.mkdir(parents=True, exist_ok=True)
    log_dir = common.PRED.parent / "logs"                        # runs/oneil_v1/logs, or <scratch>/logs for smoke
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / f"{METHOD}_seed{a.seed}{'_smoke_' + a.smoke if a.smoke else ''}.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)

    cfg = gate.load_config(run_gate.CONFIG)                      # configs/gate.yaml, as effective_config minus NCI checks
    cfg["training"]["max_epochs"] = SMOKE[a.smoke] if a.smoke else rlb.SCHEDULE["max_epochs"]
    if a.batch_surfaces is not None:
        cfg["training"]["batch_surfaces"] = a.batch_surfaces
    cfg["schedule"] = dict(rlb.SCHEDULE, registered="runs/model_improvement/bisurfgnn_v2_dev/DEVIATIONS.md "
                           + ("D13 (10-20 20:30, ours_e500)" if a.patience is not None else "D6"))
    cfg["oneil"] = dict(card="docs/ONEIL_COMPARISON_CARD_20261020.md (v2), D13", smoke=a.smoke, cell_ext_dim=968)
    if a.arm is not None:
        cfg["oneil"]["ablation"] = dict(card="docs/ONEIL_ABLATION_CARD_20261104.md", arm=a.arm)
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    rp.ARMS["both"] = {"drug_ext_dim": rp.DRUG_DIM, "cell_ext_dim": 968}   # the only model change: 968 DepMap genes
    device = torch.device(cfg["device"])
    start = time.monotonic()
    with threadpool_limits(limits=cfg["threads"]):
        torch.set_num_threads(cfg["threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frames = {r: load_frame(r) for r in common.ROLES}
        counts = {r: (frames[r].matrix_key.nunique(), frames[r].pair_id.nunique()) for r in common.ROLES}
        if counts != {"train": (10590, 349), "validation": (3528, 116), "test": (3624, 118)}:
            raise RuntimeError(f"Surface/pair counts differ from the card: {counts}")
        banks = build_banks(frames)
        features, feature_record = external_rows(banks, frames["train"])
        if a.arm == "random":
            features, feature_record = random_rows(features, feature_record)
        fit = {r: frames[r] for r in ("train", "validation")}
        if a.smoke == "tiny":
            fit["train"] = run_gate.tiny_frames({"train": fit["train"]}, cfg["timing"]["subset_seed"] + 2)["train"]
        tables = gate.build_tables(fit, banks)
        if a.arm == "noanchor":
            patch_noanchor(banks, tables)
        ext = rp.DeviceExternal(features, "both", device)
        va = tables["validation"]
        print(f"START ours seed={a.seed} smoke={a.smoke} train_points={len(tables['train'])} "
              f"validation_points={len(va)} drugs={len(banks.drug_ids)} cells={len(banks.cell_ids)}", flush=True)
        if a.patience is None:
            val_pred, detail = rlb.train_long_both(tables, banks, ext, cfg, a.seed, out, check=False,
                                                   constant_initial=a.arm == "noanchor")
        else:
            val_pred, detail = train_es(tables, banks, ext, cfg, a.seed, out, a.patience)

        # reload the selected checkpoint (run_parity's reload check), then predict test once
        model = rp.build_parity_model(banks, "both", a.seed, cfg["device"])
        model.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu"))
        np.testing.assert_array_equal(rp.predict_ext(model, va, banks, ext, cfg), val_pred)
        test_table = gate.build_tables({"test": frames["test"]}, banks)["test"]
        test_pred = rp.predict_ext(model, test_table, banks, ext, cfg)

    val_metric = common.surface_macro_percent(frames["validation"], val_pred)
    history = json.loads((out / "training_progress.json").read_text(encoding="utf-8"))["history"]
    selected = next(h for h in history if h["epoch"] == detail["best_epoch"])
    # same function and surfaces; the training loop scores against float32 table targets, common against float64
    if not np.isclose(val_metric, 100 * selected["surface_macro_rmse"], rtol=1e-6, atol=0):
        raise RuntimeError("Selection metric disagrees with common.surface_macro_percent")
    meta = dict(method=METHOD, seed=a.seed, smoke=a.smoke, selected_epoch=detail["best_epoch"],
                validation_surface_macro_percent=val_metric,
                validation_surface_macro_percent_by_epoch={h["epoch"]: 100 * h["surface_macro_rmse"] for h in history},
                history=history, **{k: v for k, v in detail.items() if k != "first_epochs_identical_to_phase1"},
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                total_wall_seconds=time.monotonic() - start, train_points=len(tables["train"]),
                features=feature_record, config=cfg, torch=torch.__version__, test_response_rows_read=0)
    for role, pred in (("validation", val_pred), ("test", test_pred)):
        frame = frames[role][common.KEY].copy()
        frame["prediction"] = pred
        print(f"WROTE {common.write_predictions(METHOD, a.seed, role, frame, meta)}", flush=True)
    print(f"DONE {METHOD} seed={a.seed} smoke={a.smoke} best_epoch={detail['best_epoch']} "
          f"val_surface_macro={val_metric:.4f}% wall={meta['total_wall_seconds']:.0f}s "
          f"peak_alloc={detail['peak_gpu_allocated_bytes']}", flush=True)


if __name__ == "__main__":
    main()

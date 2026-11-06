"""DD-PRiSM (measured-monotherapy adapted version) on O'Neil, method "ddprism" (launch card E v2, DEVIATIONS D13).

Reuses the NCI adapter (experiments/model_improvement/ddprism_measured_mono/ddprism_mm.py: author classes, loss,
density weights, schedule, monotherapy stage) with O'Neil inputs:
  * combination network: attention(drug_a, cell), attention(drug_b, cell), mono_response_a/b = Ours' anchors;
  * monotherapy stage: AACR single-agent workbook (supp_0_w2lh45.xlsx, no combination responses), 30 cells,
    mean of finite replicates per (batch, cell, drug, dose) = the anchor rule, then mean across batches per
    (drug, cell, dose); one value per NCI dose key (f"{M:.6g}") is asserted;
  * cells: 50 Hallmark sets of DepMap gene z-scores, the train-only transform of build_cell_features (fit on the
    30 cells; its state hash is asserted), i.e. the NCI adapter's Hallmark x CellMiner set with DepMap values;
  * drugs: runs/oneil_v1/drug_features/morgan512.csv; dd_concentration = log10(M) + 6.
Hyper-parameters: formal_config.json of the NCI adapter (smoke: its smoke_config.json). The author schedule (LR x0.1,
early stop) still runs on the validation author loss; the epoch is selected by common.surface_macro_percent on
validation, the author-rule epoch (val-loss argmin) is recorded. Test inputs carry no response.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/ddprism/run_ddprism.py --seed 41
  smoke: ONEIL_PRED_ROOT=<scratch>/preds ... run_ddprism.py --seed 41 --smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True            # the sibling repo is read-only: no __pycache__ there
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

common.install_guard()

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from decimal import Decimal  # noqa: E402

sys.path.insert(0, str(common.ROOT / "experiments/model_improvement/ddprism_measured_mono"))
sys.path.insert(0, str(HERE.parent / "cell_features"))
sys.path.insert(0, str(common.OLD / "experiments/audit_revision"))
import ddprism_mm as dm  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before CUDA starts)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import build_cell_features as bcf  # noqa: E402
import prepare_oneil_development_v2 as v2  # noqa: E402
from bisurfgnn.data.cellminer import (ExpressionMatrix, _gene_z_scores, fit_train_only_pathway_transform,  # noqa: E402
                                      load_cached_pathway_topology, load_cellminer_config, load_hallmark_gene_sets,
                                      normalize_cell_label, normalize_gene_symbol, sha256_file)

METHOD = "ddprism"
CONFIGS = common.ROOT / "experiments/model_improvement/ddprism_measured_mono"
LOGS = common.PRED.parent / "logs"         # runs/oneil_v1/logs, or <scratch>/logs when ONEIL_PRED_ROOT is set
ADAPTER = "DD-PRiSM-Hallmark-MeasuredMono on O'Neil (adapted; not the authors' original setting)"
DEVIATIONS = {
    "D1": "measured monotherapy = O'Neil same-batch interpolated anchors mono_response_a/b (exactly Ours' input) "
          "instead of the monotherapy model's predicted viability",
    "D2": "50 MSigDB Hallmark sets with DepMap 24Q4 log2(TPM+1) gene z-scores (train-only transform fit on the 30 "
          "cells, as Ours' pathway graphs) instead of 186 KEGG sets; NCI adapter used CellMiner values",
    "D3": "monotherapy model trained from scratch with the notebook-04 recipe on AACR single-agent records (30 cells; "
          "mean of finite replicates per batch, then mean across batches per (drug, cell, dose)); no NCI-60 "
          "pretraining or ALMANAC fine-tuning",
    "D4": dm.DEVIATIONS["D4"],
    "D5": "validation role drives LR schedule and early stopping (author loss); epoch selected by validation "
          "surface-macro RMSE (card E), the author-rule epoch (validation-loss argmin) is recorded",
    "D6": dm.DEVIATIONS["D6"],
    "D7": dm.DEVIATIONS["D7"],
}


# ---------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------
def monotherapy_records(drugs, cells):
    """AACR single-agent workbook -> one ratio per (drug, cell, dose) (batch means of the anchor rule, averaged)."""
    source = {r["role"]: r for r in json.loads(v2.SOURCE.read_text(encoding="utf-8"))["records"]}
    record = source["single_agent"]
    if "supp_0_w2lh45" not in record["local_path"]:
        raise RuntimeError(f"unexpected single-agent workbook: {record['local_path']}")
    pools, skipped = defaultdict(list), dict(rows=0, outside=0, no_finite=0)
    for _, row, strings in v2.workbook_rows(record):
        batch, cell, drug, dose = (v2.value(row.get(c), strings) for c in "ABCD")
        skipped["rows"] += 1
        cell = normalize_cell_label(cell)
        if cell not in cells or drug not in drugs:
            skipped["outside"] += 1
            continue
        reps = v2.finite_replicates(row, strings, ("E", "F", "G", "H", "I", "J"))
        if not reps:
            skipped["no_finite"] += 1
            continue
        pools[(batch, cell, drug, v2.positive_dose(dose))].extend(reps)
    per_batch = pd.DataFrame([dict(batch=b, cell=c, drug=d, dose=x, dose_M=float(x * Decimal("1e-6")),
                                   ratio=math.fsum(r) / len(r)) for (b, c, d, x), r in pools.items()])
    if (per_batch.dose <= 0).any():
        raise RuntimeError("zero or negative single-agent dose")
    per_batch["dose_key"] = [f"{x:.6g}" for x in per_batch.dose_M]       # NCI S1 lookup key rule
    records = per_batch.groupby(["drug", "cell", "dose_key"], sort=True).agg(
        dose_M=("dose_M", "first"), dose_variants=("dose", "nunique"), ratio=("ratio", "mean"),
        batches=("batch", "nunique"), batch_means=("ratio", "size")).reset_index()
    if (records.batches != records.batch_means).any():
        raise RuntimeError("two doses of one batch share an NCI dose key")
    records["percent"] = 100.0 * records.ratio      # ddprism_mm reads `percent`; one row per (drug, cell, dose key)
    info = dict(workbook=record["local_path"], workbook_rows=skipped["rows"], rows_outside_30_cells_or_38_drugs=
                skipped["outside"], rows_without_finite_replicate=skipped["no_finite"], batch_means=len(per_batch),
                records=len(records), records_from_2plus_batches=int((records.batches > 1).sum()),
                dose_keys_merging_decimal_variants=int((records.dose_variants > 1).sum()),
                drug_cell_pairs=int(records[["drug", "cell"]].drop_duplicates().shape[0]),
                ratio_range=[float(records.ratio.min()), float(records.ratio.max())])
    return records, info


def cell_gene_features(cells):
    """Hallmark-grouped DepMap gene z-scores: build_cell_features' expression matrix and train-only transform."""
    depmap_cells, labels, symbols, entrez, values = bcf.load_depmap()
    if list(depmap_cells) != list(cells):
        raise RuntimeError("DepMap cell order differs from the point tables' cells")
    cfg = load_cellminer_config(bcf.CELL_CONFIG)
    expression = ExpressionMatrix(
        cell_ids=tuple(depmap_cells), gene_ids=tuple(normalize_gene_symbol(s) for s in symbols),
        entrez_ids=tuple(entrez), values=torch.as_tensor(values, dtype=torch.float32),
        observed_mask=torch.ones(values.shape, dtype=torch.bool),
        source_release="DepMap 24Q4 Public (figshare v1) OmicsExpressionProteinCodingGenesTPMLogp1, 30 O'Neil cells",
        source_sha256=sha256_file(bcf.DEPMAP_NPZ), value_scale="log2(TPM+1)")
    catalog = load_hallmark_gene_sets(cfg)
    topology = load_cached_pathway_topology(cfg, expected_catalog_hash=catalog.catalog_hash)
    state = fit_train_only_pathway_transform(expression, catalog, topology, cells, bcf.SPLIT_LABEL,
                                             min_pathway_genes=cfg.min_pathway_genes, clip_value=cfg.clip_value,
                                             std_epsilon=cfg.std_epsilon)
    record = json.loads((bcf.OUT / "cell_features_record.json").read_text(encoding="utf-8"))["hallmark"]
    if state.state_hash != record["state_hash"]:
        raise RuntimeError("Hallmark transform differs from the one behind runs/oneil_v1/cell_features")
    z = _gene_z_scores(expression.values, expression.observed_mask, state)
    gene_sets, tensors = {}, []
    for p, name in enumerate(state.pathway_ids):
        sel = torch.where(state.membership[p])[0]
        gene_sets[name] = [state.gene_ids[i] for i in sel.tolist()]
        tensors.append(z[:, sel].float().contiguous())
    info = dict(pathways=len(gene_sets), genes_per_pathway=[len(v) for v in gene_sets.values()],
                unique_genes=len({g for v in gene_sets.values() for g in v}), fit_cells=len(state.fit_cell_ids),
                state_hash=state.state_hash, expression_hash=expression.content_hash)
    return gene_sets, tensors, info


def combination_data(points, drugs, cells, device, swap):
    """dm.CombinationData on O'Neil columns; anchors are Ours' mono_response_a/b (asserted bit-equal in float32)."""
    frame = points.assign(biological_cell_id=points.cell_id, mono_a_percent=100.0 * points.mono_response_a,
                          mono_b_percent=100.0 * points.mono_response_b,
                          response_percent=100.0 * points["response"] if "response" in points else np.nan)
    data = dm.CombinationData(frame, drugs, cells, device, swap)
    n = len(points)
    for got, col in ((data.v_a[:n], "mono_response_a"), (data.v_b[:n], "mono_response_b")):
        want = torch.tensor(points[col].to_numpy(np.float64), dtype=torch.float32, device=device).reshape(-1, 1)
        if not torch.equal(got, want):
            raise RuntimeError(f"combination input differs from {col}")
    return data


def predict(combo, attention, data, batch_size):
    combo.eval()
    with torch.no_grad():
        idx = torch.arange(data.n, device=data.y.device)
        return torch.cat([combo(data.features(attention, idx[lo:lo + batch_size]))[2]
                          for lo in range(0, data.n, batch_size)]).cpu().numpy().ravel().astype(np.float64)


# ---------------------------------------------------------------------------------------------------------------
# combination stage: ddprism_mm.train_combination with epoch selection by validation surface-macro RMSE
# ---------------------------------------------------------------------------------------------------------------
def train_combination(Combo, attention, train, val, val_points, cfg, seed, device, deadline, log):
    c = cfg["combo"]
    dm.seed_everything(seed)
    combo = Combo().to(device)
    dens = dm.DensityWeights(train.y64, c["gamma"])
    w_tr = torch.tensor(dens.weights(train.y.cpu().numpy().ravel()), device=device)
    w_va = torch.tensor(dens.weights(val.y.cpu().numpy().ravel()), device=device)
    opt = torch.optim.AdamW(combo.parameters(), lr=c["learning_rate"])
    sched = dm.AuthorSchedule(c["learning_rate"], c["threshold"], c["patience_lr"], c["patience_es"], c["lr_factor"])
    val_order = torch.randperm(val.n, generator=torch.Generator().manual_seed(seed * 7 + 1)).to(device)
    history, stopped_by = [], "max_epochs"
    best_macro, best_state, best_epoch, author_loss, author_epoch = math.inf, None, -1, math.inf, -1
    for epoch in range(c["max_epochs"]):
        if time.monotonic() > deadline:
            stopped_by = "wall_limit"; break
        t0 = time.perf_counter()
        combo.train()
        order = torch.randperm(train.n, generator=torch.Generator().manual_seed(seed * 100_003 + epoch)).to(device)
        losses = []
        for lo in range(0, train.n, c["batch_size"]):
            idx = order[lo:lo + c["batch_size"]]
            if len(idx) < 2:
                continue
            _, _, pred = combo(train.features(attention, idx))
            loss, _, _ = dm.author_loss(pred, train.y[idx], w_tr[idx], c["alpha"], c["beta"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite combination loss")
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.detach())
        train_loss = float(torch.stack(losses).mean())
        t_train = time.perf_counter() - t0
        val_loss, pred = dm.evaluate_combination(combo, attention, val, w_va, cfg, val_order)
        p = pred.cpu().numpy().ravel().astype(np.float64)
        macro = common.surface_macro_percent(val_points, p)
        pooled = common.pooled_rmse_percent(val_points["response"], p)
        if macro < best_macro:
            best_macro, best_state, best_epoch = macro, copy.deepcopy(combo.state_dict()), epoch
        if val_loss < author_loss:
            author_loss, author_epoch = val_loss, epoch
        lr_used = sched.lr
        lr_changed, stop = sched.step(val_loss)
        if lr_changed:
            opt.param_groups[0]["lr"] = sched.lr
        sec = time.perf_counter() - t0
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_surface_macro_percent=macro,
                            val_pooled_rmse_percent=pooled, lr=lr_used, train_seconds=round(t_train, 3),
                            seconds=round(sec, 3)))
        log(f"combo epoch {epoch} train_loss {train_loss:.5f} val_loss {val_loss:.5f} macro% {macro:.3f} "
            f"pooled% {pooled:.3f} lr {lr_used:g} {sec:.1f}s")
        if stop:
            stopped_by = "early_stopping"; break
    combo.load_state_dict(best_state)
    return combo, dict(selected_epoch=best_epoch, selected_val_surface_macro_percent=best_macro,
                       author_rule_epoch=author_epoch, author_rule_val_loss=author_loss, stopped_by=stopped_by,
                       epochs=len(history), density_grid_top=dens.grid_top, train_rows_with_swap=train.n,
                       history=history)


# ---------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="capped epochs (smoke_config.json); needs ONEIL_PRED_ROOT")
    args = parser.parse_args()
    if args.smoke and "ONEIL_PRED_ROOT" not in os.environ:
        raise SystemExit("--smoke needs ONEIL_PRED_ROOT so nothing is written into runs/oneil_v1/preds")
    for role in ("validation", "test"):
        if common.pred_path(METHOD, args.seed, role).exists():
            raise SystemExit(f"Refusing to overwrite {common.pred_path(METHOD, args.seed, role)}")
    LOGS.mkdir(parents=True, exist_ok=True)
    console = (LOGS / f"{METHOD}_seed{args.seed}.log").open("x", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        console.write(message + "\n"); console.flush()

    cfg_file = CONFIGS / ("smoke_config.json" if args.smoke else "formal_config.json")
    cfg = json.loads(cfg_file.read_text(encoding="utf-8")) | dict(seed=args.seed)
    log(f"{ADAPTER} seed {args.seed} config {cfg_file.name} pred_root {common.PRED}")
    t_start = time.perf_counter()
    deadline = time.monotonic() + 3600.0 * float(cfg["max_wall_hours"])
    device = dm.model_device(cfg)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    points = {role: common.load_points(role) for role in common.ROLES}
    drugs = sorted(set(points["train"].drug_a_id) | set(points["train"].drug_b_id))
    cells = sorted(set(points["train"].cell_id))
    for role in ("validation", "test"):
        f = points[role]
        if not (set(f.drug_a_id) | set(f.drug_b_id)) <= set(drugs) or not set(f.cell_id) <= set(cells):
            raise RuntimeError(f"{role} drugs/cells must all appear in train (pair hold-out)")
    if (len(drugs), len(cells)) != (38, 30):
        raise RuntimeError(f"expected 38 drugs x 30 cells, got {len(drugs)} x {len(cells)}")
    records, records_info = monotherapy_records(set(drugs), set(cells))
    gene_sets, cell_tensors, cell_info = cell_gene_features(cells)
    morgan = pd.read_csv(common.DRUG / "morgan512.csv").set_index("drug_id")
    fps = torch.tensor(morgan.loc[drugs].to_numpy(np.float32), device=device)
    cell_tensors = [t.to(device) for t in cell_tensors]
    log(f"inputs: {json.dumps(records_info)}; pathways {cell_info['pathways']} genes {cell_info['unique_genes']}; "
        f"fps {tuple(fps.shape)}")

    Mono, Combo = dm.author_classes(len(gene_sets))
    t0 = time.perf_counter()
    mono, mono_info = dm.train_monotherapy(Mono, gene_sets, cell_tensors, fps, records, cells, drugs, cfg, args.seed,
                                           device, deadline, log)
    mono_seconds = time.perf_counter() - t0
    attention = dm.attention_table(mono, cell_tensors, fps, len(drugs), len(cells), device)
    train = combination_data(points["train"], drugs, cells, device, swap=cfg["combo"]["swap_augment"])
    val = combination_data(points["validation"], drugs, cells, device, swap=False)
    test = combination_data(points["test"], drugs, cells, device, swap=False)
    t0 = time.perf_counter()
    combo, combo_info = train_combination(Combo, attention, train, val, points["validation"], cfg, args.seed, device,
                                          deadline, log)
    combo_seconds = time.perf_counter() - t0

    bs = cfg["combo"]["batch_size"]
    val_pred, test_pred = predict(combo, attention, val, bs), predict(combo, attention, test, bs)
    macro = common.surface_macro_percent(points["validation"], val_pred)
    if abs(macro - combo_info["selected_val_surface_macro_percent"]) > 1e-6:
        raise RuntimeError(f"selected state does not reproduce its validation metric ({macro} vs "
                           f"{combo_info['selected_val_surface_macro_percent']})")
    gpu = dict(device=str(device))
    if device.type == "cuda":
        gpu.update(name=torch.cuda.get_device_name(device),
                   peak_allocated_mb=round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1),
                   peak_reserved_mb=round(torch.cuda.max_memory_reserved(device) / 2 ** 20, 1))
    meta = dict(
        method=METHOD, adapter=ADAPTER, seed=args.seed, smoke=args.smoke, config_file=str(cfg_file), config=cfg,
        selection="validation surface-macro RMSE % (common.surface_macro_percent)",
        selected_epoch=combo_info["selected_epoch"], selected_val_surface_macro_percent=macro,
        selected_val_pooled_rmse_percent=common.pooled_rmse_percent(points["validation"]["response"], val_pred),
        author_rule_epoch=combo_info["author_rule_epoch"], stopped_by=combo_info["stopped_by"],
        combination=combo_info | dict(seconds=round(combo_seconds, 1)),
        mono=mono_info | dict(seconds=round(mono_seconds, 1)), monotherapy_records=records_info,
        cell_features=cell_info, drugs=len(drugs), cells=len(cells), test_responses_read=0, gpu=gpu,
        parameters=dict(mono=sum(p.numel() for p in mono.parameters()), combo=sum(p.numel() for p in combo.parameters())),
        wall_seconds=round(time.perf_counter() - t_start, 1), deviations=DEVIATIONS, author_code_note=dm.AUTHOR_NOTE,
        versions=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
                      pandas=pd.__version__))
    for role, pred in (("validation", val_pred), ("test", test_pred)):
        frame = points[role][common.KEY].assign(prediction=pred)
        log(f"wrote {common.write_predictions(METHOD, args.seed, role, frame, meta)}")
    log(json.dumps({k: meta[k] for k in ("selected_epoch", "selected_val_surface_macro_percent",
                                         "selected_val_pooled_rmse_percent", "author_rule_epoch", "stopped_by", "gpu",
                                         "wall_seconds")}))


if __name__ == "__main__":
    main()

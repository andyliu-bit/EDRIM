"""V2 gap diagnosis, step D1: DeepSynBa v2 inference on the VALIDATION split (split 2) only.

No training, no test data. Uses the authors' code (read only) and the saved checkpoints of the DeepSynBa
reproduction v2 (`C:/LHZ/research/bisurfgnn/runs/audit_revision/deepsynba_repro_v2_20261012/`, read only).

Inputs are served exactly as in the reproduction's validation loop:
  * `inputs/trainval_cache.npz` (splits 1/2 only; hash checked against the reproduction's protocol.json);
  * `repro_common.SubsetCacheDataset` (item dicts identical to the authors' NciAlmanacDataset);
  * batch size 128, no shuffling, `module.model(drug1_smile, drug2_smile, cell_gex, drug1_dose, drug2_dose)`.

Correctness check: the 16-entry validation RMSE recomputed here must match the per-epoch validation RMSE logged
during training (`full/monitor_val.jsonl`) for the checkpoint's epoch.

Run with the DeepSynBa venv in isolated mode, without writing bytecode into the read-only checkout:
  C:/LHZ/research/bisurfgnn/.venvs/deepsynba/Scripts/python.exe -I -B \
      experiments/model_improvement/v2_gap_diagnosis/d1_deepsynba_val_inference.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                  # this worktree
MAIN = Path("C:/LHZ/research/bisurfgnn")                # read-only main checkout
REPRO_CODE = MAIN / "experiments/audit_revision/deepsynba_repro_v2"
REPRO_RUN = MAIN / "runs/audit_revision/deepsynba_repro_v2_20261012"
OUT = ROOT / "runs/model_improvement/v2_gap_diagnosis/deepsynba_val"

CHECKPOINTS = {
    # tag: (run, checkpoint path, file holding the recorded sha256 or None). The epoch is read from the checkpoint
    # itself: the first attempt assumed last.ckpt = epoch 499 and stopped because its validation RMSE matched
    # epoch 484 instead (last.ckpt was last written at 00:33, when epoch=484.ckpt was saved).
    "full_epoch484_selected": ("full", REPRO_RUN / "full/checkpoints/epoch=484.ckpt", REPRO_RUN / "full/train_finished.json"),
    "full_last": ("full", REPRO_RUN / "full/checkpoints/last.ckpt", None),
    "pilot_epoch047": ("pilot", REPRO_RUN / "pilot/checkpoints/epoch=047.ckpt", REPRO_RUN / "pilot/train_finished.json"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        raise SystemExit("run with `python -I -B` (pickle rule; no bytecode in the read-only checkout)")
    sys.path.insert(0, str(REPRO_CODE))
    import repro_common as rc  # noqa: E402  (read only; puts the author repo on sys.path)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from trainer import TrainingModule  # author code, read only

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "d1_report.json").exists():
        raise SystemExit("D1 already completed")
    proto = json.loads((REPRO_RUN / "protocol.json").read_text(encoding="utf-8"))
    cache_hash = sha256(rc.CACHE)
    if cache_hash != proto["inputs_sha256"]["inputs/trainval_cache.npz"]:
        raise SystemExit("train/val cache differs from the reproduction's frozen hash")
    d = rc.load_cache()
    splits = set(np.unique(d["split"]).tolist())
    if splits != {1, 2}:
        raise SystemExit(f"cache must contain splits 1/2 only, found {splits}")
    rows = np.where(d["split"] == 2)[0]
    cell2gex, emb = rc.load_features()
    val_ds = rc.SubsetCacheDataset(d, rows, cell2gex, emb)
    logged = {}
    for name in ("full", "pilot"):
        for line in (REPRO_RUN / name / "monitor_val.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                logged[(name, r["epoch"])] = r["val_rmse"]

    keys = np.array([f"{a}_{b}_{c}" for a, b, c in zip(val_ds.drug1, val_ds.drug2, val_ds.cell)])
    report = {"step": "D1", "split_read": 2, "test_rows_read": 0, "validation_matrices": int(len(rows)),
              "cache_sha256": cache_hash, "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                                                       "numpy": np.__version__}, "checkpoints": {}}
    for tag, (run_name, path, record) in CHECKPOINTS.items():
        t0 = time.time()
        digest = sha256(path)
        if record is not None:
            expected = json.loads(record.read_text(encoding="utf-8"))["best_checkpoint_sha256"]
            if digest != expected:
                raise SystemExit(f"{tag}: checkpoint hash differs from the training record")
        meta = torch.load(str(path), map_location="cpu", weights_only=False)     # python -I; own training output
        epoch, global_step = int(meta["epoch"]), int(meta["global_step"])
        state_digest = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in meta["state_dict"].values())).hexdigest()
        del meta
        module = TrainingModule.load_from_checkpoint(str(path), strict=True, map_location="cuda").eval().to("cuda")
        cfg = module.config
        if (cfg["model_params"]["gene_channel"], cfg["model_params"]["emb_size"]) != (2048, 256):
            raise SystemExit("checkpoint is not the grid-point model")
        loader = DataLoader(val_ds, batch_size=cfg["trainer_params"]["batch_size"], shuffle=False, num_workers=0,
                            drop_last=False)
        preds, truth = [], []
        with torch.no_grad():
            for batch in loader:
                out = module.model(*(batch[k].to("cuda") for k in ("drug1_smile", "drug2_smile", "cell_gex",
                                                                   "drug1_dose", "drug2_dose")))
                preds.append(out["dose_response"].cpu().numpy().reshape(-1, 4, 4))
                truth.append(batch["dose_response"].numpy().reshape(-1, 4, 4))
        p, t = np.concatenate(preds), np.concatenate(truth)
        if not np.array_equal(t, d["matrix"][rows]):
            raise SystemExit("truth order differs from the cache rows")
        rmse16 = math.sqrt(float(np.mean((p.astype(np.float64) - t.astype(np.float64)) ** 2)))
        ref = logged[(run_name, epoch)]
        rel = abs(rmse16 - ref) / ref
        if rel > 1e-4:
            raise SystemExit(f"{tag}: recomputed val RMSE {rmse16:.6f} differs from the logged {ref:.6f} (epoch {epoch})")
        target = OUT / f"{tag}_val_predictions.npz"
        reused = target.exists()
        if reused:      # written by the first attempt: keep it, but require the same predictions
            old = np.load(target, allow_pickle=False)
            max_diff = float(np.max(np.abs(old["predictions"].astype(np.float64) - p.astype(np.float64))))
            if max_diff > 1e-4 or not np.array_equal(old["matrix_key"], keys):
                raise SystemExit(f"{tag}: existing prediction file differs (max {max_diff})")
        else:
            max_diff = None
            np.savez_compressed(target, predictions=p, truth=t, matrix_key=keys, drug1=np.array(val_ds.drug1),
                                drug2=np.array(val_ds.drug2), cell=np.array(val_ds.cell), x1=d["x1"][rows], x2=d["x2"][rows])
        report["checkpoints"][tag] = {"path": str(path), "epoch_in_checkpoint": epoch, "global_step": global_step,
                                      "file_sha256": digest, "state_dict_sha256": state_digest,
                                      "hash_checked_against_record": record is not None,
                                      "val_rmse16_recomputed": rmse16, "val_rmse16_logged_same_epoch": ref,
                                      "relative_difference": rel, "seconds": round(time.time() - t0, 1),
                                      "predictions_file": target.name, "reused_first_attempt_file": reused,
                                      "reused_max_abs_diff": max_diff}
        print(tag, json.dumps(report["checkpoints"][tag]), flush=True)
        del module
        torch.cuda.empty_cache()
    with (OUT / "d1_report.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()

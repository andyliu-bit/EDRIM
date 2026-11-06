"""Formal validation run of the DD-PRiSM measured-monotherapy adapter (docs/DDPRISM_MEASURED_MONO_ADAPTER.md).

Prepared, NOT launched by the agent that wrote it. One seed per call (default 41, the first of the gate seeds).
  * training: S1 split 1 only; selection and evaluation: S1 split 2 only (no split-3 file is ever opened);
  * hyperparameters: formal_config.json (authors' notebook-04 values where defined; see hyperparameter_sources);
  * standard eval mode for all evaluation (no test-time BatchNorm refresh; deviation D4);
  * refuses to run if the output directory already exists; requires --confirm to train.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/ddprism_measured_mono/launch_validation.py --seed 41 --dry-run
  .venv/Scripts/python.exe -I -B experiments/model_improvement/ddprism_measured_mono/launch_validation.py --seed 41 --confirm
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ddprism_mm as dm  # noqa: E402

RUNS = dm.ROOT / "runs/model_improvement/bisurfgnn_v2_dev/ddprism_measured_mono"


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(dm.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--dry-run", action="store_true", help="check inputs and bindings only; no training, no output")
    parser.add_argument("--confirm", action="store_true", help="required to train")
    args = parser.parse_args()
    if not sys.flags.isolated:
        raise SystemExit("run with python -I")
    out = RUNS / f"seed_{args.seed}"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite an existing run directory: {out}")
    cfg = json.loads((HERE / "formal_config.json").read_text(encoding="utf-8"))
    cfg["seed"] = int(args.seed)
    if args.dry_run:
        tables = dm.load_tables()
        drugs, cells = dm.entity_index(tables)
        records = dm.monotherapy_records(tables["train"])
        print(json.dumps(dict(planned_output=str(out), seed=args.seed, train_points=len(tables["train"]),
                              validation_points=len(tables["validation"]), drugs=len(drugs), cells=len(cells),
                              monotherapy_records=len(records), source_sha256=dm.source_hashes(),
                              files_read=sorted(set(dm.READ_LOG))), indent=1))
        return
    if not args.confirm:
        raise SystemExit("Training needs --confirm (the formal run is launched by the user, not by default)")
    out.mkdir(parents=True, exist_ok=False)
    console = (out / "console.log").open("x", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        console.write(message + "\n"); console.flush()

    dm.write_json(out / "config.json", dict(cfg, adapter=dm.ADAPTER_NAME, deviations=dm.DEVIATIONS,
                                            author_code_note=dm.AUTHOR_NOTE, source_sha256=dm.source_hashes(),
                                            git_head=git_head(), started_local_time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    result = dm.run_pipeline(cfg, out, save_predictions=True, log=log)
    result["finished_local_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dm.write_json(out / "metrics.json", result)
    dm.write_json(out / "input_access.json", dict(files_read=sorted(set(dm.READ_LOG)), split_read=[1, 2],
                                                  test_rows_read=0))
    log(json.dumps({k: result[k] for k in ("selected_epoch", "stopped_by", "validation", "gpu", "wall_seconds")},
                   indent=1))


if __name__ == "__main__":
    main()

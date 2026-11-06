"""Smoke run of the DD-PRiSM measured-monotherapy adapter: full S1 data (split 1 train, split 2 validation),
monotherapy stage capped at 3 epochs and combination stage at 2 epochs, wall guard 9 minutes. Measures time per
epoch and peak GPU memory. The numbers it prints are NOT a result. Refuses to overwrite its output directory.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/ddprism_measured_mono/run_smoke.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ddprism_mm as dm  # noqa: E402

OUT = dm.ROOT / "runs/model_improvement/bisurfgnn_v2_dev/ddprism_measured_mono_smoke"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if not sys.flags.isolated:
        raise SystemExit("run with python -I")
    cfg = json.loads((HERE / "smoke_config.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=False)
    console = (args.out / "console.log").open("x", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        console.write(message + "\n"); console.flush()

    dm.write_json(args.out / "config.json", dict(cfg, adapter=dm.ADAPTER_NAME, deviations=dm.DEVIATIONS,
                                                 author_code_note=dm.AUTHOR_NOTE, source_sha256=dm.source_hashes()))
    result = dm.run_pipeline(cfg, args.out, save_predictions=False, log=log)
    result["note"] = "SMOKE RUN with capped epochs: timing and memory only; the validation numbers are not a result"
    dm.write_json(args.out / "smoke_metrics.json", result)
    dm.write_json(args.out / "input_access.json", dict(files_read=sorted(set(dm.READ_LOG)), split_read=[1, 2],
                                                       test_rows_read=0))
    log(json.dumps({k: result[k] for k in ("mono", "combination", "gpu", "wall_seconds", "validation")}, indent=1))


if __name__ == "__main__":
    main()

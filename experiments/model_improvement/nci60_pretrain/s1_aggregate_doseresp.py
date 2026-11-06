"""NCI60-Pretrain-v1 step 1: stream DOSERESP.zip (``unzip -p``, never extracted) into a compact aggregate.

Run isolated (raw files are untrusted data):
  .venv/Scripts/python.exe -I experiments/model_improvement/nci60_pretrain/s1_aggregate_doseresp.py
Outputs (data/interim/nci60_pretrain_v1/): doseresp_aggregate_raw.parquet, doseresp_cell_labels.csv, s1_stats.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from bisurfgnn.pretrain.nci60 import append_log, sha256_file, stream_doseresp_aggregate  # noqa: E402

RAW = ROOT / "data/raw/nci60_dtp_202610"
OUT = ROOT / "data/interim/nci60_pretrain_v1"
RUN = ROOT / "runs/model_improvement/nci60_pretrain_v1"
UNZIP = "C:/software/Git/usr/bin/unzip.exe"


def main():
    OUT.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    target = OUT / "doseresp_aggregate_raw.parquet"
    if target.exists():
        raise FileExistsError(target)
    t0 = time.monotonic()
    sums = {}
    for line in (RAW / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        if line.strip():
            h, name = line.split()
            sums[name.lstrip("*")] = h.lower()
    zip_path = RAW / "DOSERESP.zip"
    got = sha256_file(zip_path)
    if got != sums["DOSERESP.zip"]:
        raise RuntimeError(f"DOSERESP.zip SHA-256 mismatch {got}")
    agg, combos, st = stream_doseresp_aggregate(zip_path, UNZIP, cwd=OUT)
    agg.to_parquet(target, index=False)
    combos.to_csv(OUT / "doseresp_cell_labels.csv", index=False)
    st["doseresp_zip_sha256"] = got
    st["outputs"] = {p.name: dict(sha256=sha256_file(p), bytes=p.stat().st_size)
                     for p in (target, OUT / "doseresp_cell_labels.csv")}
    st["total_seconds"] = time.monotonic() - t0
    (OUT / "s1_stats.json").write_text(json.dumps(st, indent=1, default=float), encoding="utf-8")
    msg = (f"STEP1 DOSERESP stream: rows {st['rows']:,}; aggregate keys {st['aggregate_keys']:,}; "
           f"curves(EXPID,NSC,cell) {st['curves_expid_nsc_cell_all_prefixes']:,}; distinct EXPID {st['distinct_expid']:,}; "
           f"invalid PTC rows {st['ptc_invalid_rows']:,}; units {[(k, v['rows']) for k, v in st['by_unit'].items()]}; "
           f"zip sha256 {got}; aggregate sha256 {st['outputs'][target.name]['sha256']}; {st['total_seconds']:.0f}s")
    append_log(RUN / "STEP_LOG.md", msg); print(msg)


if __name__ == "__main__":
    main()

"""BiSurfGNN v2 phase 1: two-seed screening of the three input-parity arms against the gate additive baselines.

Frozen rule (docs/BISURFGNN_V2_DEV_PROTOCOL.md section 3, two-seed screening): for arm a and seed s,

    Delta_s = RMSE_a(s) / RMSE_gate(s) - 1,

with RMSE = validation (split 2) combination-cell pooled RMSE in % at the selected epoch (metrics.json
"validation_pooled_rmse_percent") and the gate additive run of the same seed as baseline. The arm advances to five
seeds only if mean(Delta_41, Delta_42) <= -1% and Delta_41 < 0 and Delta_42 < 0. Surface-macro RMSE is reported as a
descriptive secondary quantity only.

Baselines (PublicSplitGate-v1, validation only):
  seed 41  runs/model_improvement/public_split_gate_v1/s3_additive_r2/bisurf_additive_seed_41/metrics.json
  seed 42  runs/model_improvement/public_split_gate_v1/s4_bisurf_additive_a/bisurf_additive_seed_42/metrics.json

Reads only the metrics/summary/launch JSON files of these runs and of runs/model_improvement/bisurfgnn_v2_dev/phase1/
<arm>_seed_<seed>/; nothing under s5/ and no test file. Writes phase1/phase1_summary.json exactly once, only when all
six runs passed, and refuses to overwrite it. --dry-run prints the same JSON (incomplete arms marked) and writes nothing.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/input_parity/summarize_phase1.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / "runs/model_improvement/public_split_gate_v1"
V2 = ROOT / "runs/model_improvement/bisurfgnn_v2_dev"
PHASE1 = V2 / "phase1"
SUMMARY = PHASE1 / "phase1_summary.json"
PROTOCOL = ROOT / "docs/BISURFGNN_V2_DEV_PROTOCOL.md"
PROTOCOL_SHA256 = "155ae4f03ceb8275f68d10f9830f5bcd193c41f70564cc138bc175840788a333"
ARMS = ("molformer", "genes976", "both")
SEEDS = (41, 42)
BASELINES = {41: GATE / "s3_additive_r2/bisurf_additive_seed_41",
             42: GATE / "s4_bisurf_additive_a/bisurf_additive_seed_42"}
METRIC = "validation_pooled_rmse_percent"
SECONDARY = "validation_surface_macro_rmse"
MEAN_THRESHOLD = -0.01                      # mean paired relative change must be <= -1%
PROTECTED = ("public_split_gate_v1/s5", "test_evaluated", "test_predictions", "final_predictions", "test.parquet",
             "points_test", "jaaks", "intensity", "az_sanger", "reserve")
RULE = ("Delta_s = RMSE_arm(s) / RMSE_gate_additive(s) - 1 on validation combination-cell pooled RMSE (%); "
        "advance to 5 seeds iff mean(Delta_41, Delta_42) <= -1% and both Delta_s < 0 "
        "(docs/BISURFGNN_V2_DEV_PROTOCOL.md section 3)")


def guard(path: Path) -> Path:
    text = Path(path).resolve().as_posix().lower()
    if any(token in text for token in PROTECTED):
        raise SystemExit(f"refusing to open a protected path: {text}")
    return Path(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(guard(path).read_bytes()).hexdigest()


def read_json(path: Path, inputs: dict) -> dict:
    inputs[Path(path).resolve().relative_to(ROOT).as_posix()] = sha256(path)
    return json.loads(guard(path).read_text(encoding="utf-8"))


def screen(deltas: dict[int, float]) -> dict:
    """The frozen two-seed screening rule applied to the paired relative changes of one arm."""
    if sorted(deltas) != list(SEEDS):
        raise ValueError(f"screening needs exactly the seeds {SEEDS}")
    mean = sum(deltas.values()) / len(deltas)
    mean_ok = mean <= MEAN_THRESHOLD
    every_negative = all(d < 0 for d in deltas.values())
    return dict(mean_relative_change=mean, mean_le_minus_1pct=mean_ok, every_seed_negative=every_negative,
                decision="advance_to_5_seeds" if mean_ok and every_negative else "stop")


def baseline(seed: int, inputs: dict) -> dict:
    run = BASELINES[seed]
    metrics = read_json(run / "metrics.json", inputs)
    launch = read_json(run.parent / "launch.json", inputs)
    status = read_json(run.parent / "summary.json", inputs)
    if metrics.get("model") != "bisurf_additive" or metrics.get("seed") != seed or status.get("status") != "passed" \
            or launch.get("form") != "additive" or seed not in launch.get("seeds", []):
        raise SystemExit(f"unexpected gate baseline record for seed {seed}: {run}")
    return dict(run=run.relative_to(ROOT).as_posix(), rmse=metrics[METRIC], surface_macro=metrics[SECONDARY],
                best_epoch=metrics["best_epoch"], stopped_by=metrics["stopped_by"], epochs_run=metrics["epochs_run"],
                config=launch["config"])


def arm_run(arm: str, seed: int, inputs: dict, configs: dict) -> dict | None:
    run = PHASE1 / f"{arm}_seed_{seed}"
    if not (run / "summary.json").exists():
        return None
    summary = read_json(run / "summary.json", inputs)
    launch = read_json(run / "launch.json", inputs)
    metrics = read_json(run / "metrics.json", inputs)
    for label, record in (("summary", summary), ("launch", launch), ("metrics", metrics)):
        if record.get("arm") != arm or record.get("seed") != seed or record.get("mode") != "train":
            raise SystemExit(f"{run}: {label} record does not describe a formal {arm} seed-{seed} run")
    if summary.get("status") != "passed":
        raise SystemExit(f"{run}: run did not pass")
    if launch["config"] != configs[seed]:
        raise SystemExit(f"{run}: configuration differs from the gate baseline of seed {seed}")
    return dict(run=run.relative_to(ROOT).as_posix(), rmse=metrics[METRIC], surface_macro=metrics[SECONDARY],
                best_epoch=metrics["best_epoch"], stopped_by=metrics["stopped_by"], epochs_run=metrics["epochs_run"],
                parameters=metrics["parameters"], wall_hours=round(metrics["wall_seconds"] / 3600, 3),
                peak_gpu_allocated_gb=(round(metrics["peak_gpu_allocated_bytes"] / 1e9, 3)
                                       if metrics.get("peak_gpu_allocated_bytes") else None))


def build_summary() -> dict:
    inputs: dict[str, str] = {}
    protocol = sha256(PROTOCOL)
    frozen = read_json(V2 / "PROTOCOL_FREEZE.json", inputs)["sha256"]
    if not protocol == frozen == PROTOCOL_SHA256:
        raise SystemExit("docs/BISURFGNN_V2_DEV_PROTOCOL.md differs from its freeze record")
    inputs[PROTOCOL.relative_to(ROOT).as_posix()] = protocol
    base = {seed: baseline(seed, inputs) for seed in SEEDS}
    configs = {seed: base[seed].pop("config") for seed in SEEDS}
    arms, missing = {}, []
    for arm in ARMS:
        runs = {seed: arm_run(arm, seed, inputs, configs) for seed in SEEDS}
        absent = [seed for seed, r in runs.items() if r is None]
        missing += [f"{arm}_seed_{seed}" for seed in absent]
        entry = {"runs": {str(s): r for s, r in runs.items()}}
        if absent:
            entry.update(decision="incomplete", missing_seeds=absent)
        else:
            deltas = {s: runs[s]["rmse"] / base[s]["rmse"] - 1 for s in SEEDS}
            entry["relative_change"] = {str(s): d for s, d in deltas.items()}
            entry.update(screen(deltas))
            entry["secondary_surface_macro_relative_change"] = {
                str(s): runs[s]["surface_macro"] / base[s]["surface_macro"] - 1 for s in SEEDS}
        arms[arm] = entry
    return dict(
        protocol=dict(path=PROTOCOL.relative_to(ROOT).as_posix(), sha256=protocol, stage="phase 1 two-seed screening"),
        rule=RULE, metric=METRIC, mean_threshold=MEAN_THRESHOLD, seeds=list(SEEDS),
        baselines={str(s): b for s, b in base.items()}, arms=arms,
        advancing_arms=[a for a in ARMS if arms[a].get("decision") == "advance_to_5_seeds"],
        complete=not missing, missing_runs=missing, inputs_sha256=inputs,
        generated_local_time=datetime.now().astimezone().isoformat(timespec="seconds"),
        split="validation (split 2) only", test_rows_read=0,
        note="Screening result on validation data; not a test-set result and not a final model claim.",
    )


def table(summary: dict) -> str:
    lines = ["arm        seed  rmse_arm  rmse_gate  delta_%   decision"]
    for arm, entry in summary["arms"].items():
        for seed in map(str, SEEDS):
            run = entry["runs"].get(seed)
            gate = summary["baselines"][seed]["rmse"]
            if run is None:
                lines.append(f"{arm:<10} {seed:>4}  {'-':>8}  {gate:9.4f}  {'-':>7}   missing")
                continue
            delta = 100 * (run["rmse"] / gate - 1)
            lines.append(f"{arm:<10} {seed:>4}  {run['rmse']:8.4f}  {gate:9.4f}  {delta:+7.3f}")
        mean = entry.get("mean_relative_change")
        lines.append(f"{arm:<10} mean  {'':>8}  {'':>9}  {(100 * mean if mean is not None else float('nan')):+7.3f}   "
                     f"{entry['decision']}")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print the summary (incomplete arms allowed); write nothing")
    args = parser.parse_args()
    if not args.dry_run and SUMMARY.exists():
        print(f"refusing to overwrite {SUMMARY}", file=sys.stderr)
        return 3
    summary = build_summary()
    print(table(summary))
    if args.dry_run:
        print(json.dumps(summary, indent=1))
        return 0
    if not summary["complete"]:
        print(f"not written: runs missing or unfinished: {summary['missing_runs']}", file=sys.stderr)
        return 2
    with SUMMARY.open("x", encoding="utf-8", newline="\n") as handle:       # exclusive create: never overwrites
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {SUMMARY.relative_to(ROOT).as_posix()} (sha256 {sha256(SUMMARY)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

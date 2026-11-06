"""Bounded, resumable development queue; decisions never access test responses."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "runs/model_improvement/remaining_suite_v1"
CONFIG = ROOT / "configs/model_improvement/remaining_suite_v1.yaml"
BASE = "bisurf_mono_v2"
FACTORIAL = [BASE, "bisurf_pooled_concat_v2", "bisurf_pooled_gate_v2",
             "bisurf_bixt_concat_v2", "bisurf_cross_attention_concat_v2",
             "bisurf_cross_attention_gate_v2"]
BEGIN = "<!-- AUTO_REMAINING_SUITE_BEGIN -->"
END = "<!-- AUTO_REMAINING_SUITE_END -->"


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_regression(path):
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    if not suites or any(int(s.attrib[k]) != 0 for s in suites for k in ("failures", "errors", "skipped")):
        raise RuntimeError("Regression report contains failures, errors or skipped checks")
    count = sum(int(s.attrib["tests"]) for s in suites)
    if count < 295:
        raise RuntimeError("Regression report is incomplete")
    return count


def verify_stage(output, cfg):
    summary = read_json(output / "summary.json")
    launch = read_json(output / "launch.json")
    if summary.get("status") != "passed" or launch["config"] != cfg:
        raise RuntimeError("Existing stage is unsuccessful or its config differs")
    expected = len(cfg["models"]) * (len(cfg["seeds"]) + 1)
    if len(summary["results"]) != expected or any(r["status"] != "passed" for r in summary["results"]):
        raise RuntimeError("Stage has missing or failed jobs")
    repeats = summary["repeat_checks"]
    if {r["model"] for r in repeats} != set(cfg["models"]) or any(r["max_abs_difference"] > 1e-7 for r in repeats):
        raise RuntimeError("Stage repetition checks are incomplete or inconsistent")
    if read_json(output / "input_access.json")["test_response_rows_read"] != 0:
        raise RuntimeError("Forbidden test response access")
    audit = read_json(output / "sample_split_audit.json")
    if not all(audit["checks"].values()):
        raise RuntimeError("Stage split audit failed")
    ids = pd.read_csv(output / "validation_ids.csv")
    paths = list(output.glob("*/predictions.csv"))
    if len(paths) != expected:
        raise RuntimeError("Prediction files are incomplete")
    for path in paths:
        frame = pd.read_csv(path)
        if frame.source_row.tolist() != ids.source_combodrugseq.tolist() or frame.surface_id.tolist() != ids.surface_replicate_id.tolist():
            raise RuntimeError("Prediction rows differ from frozen validation IDs")
        if not np.isfinite(frame[["target", "prediction"]].to_numpy()).all():
            raise RuntimeError("Nonfinite predictions")
        if not (path.parent / "model.pt").exists() or not (path.parent / "largest_errors.csv").exists():
            raise RuntimeError("Checkpoint or failure examples are missing")
        metrics = read_json(path.parent / "metrics.json")
        if len(metrics["history"]) != cfg["training"]["epochs"] + 1:
            raise RuntimeError("Training history is incomplete")
    return summary


def choose_candidate(paths):
    candidates = []
    for path in paths:
        for result in read_json(path / "development_decision.json")["candidates"]:
            if result["advance_to_multifold"] and all(result["checks"].values()):
                candidates.append({**result, "source_stage": str(path)})
    return sorted(candidates, key=lambda r: (-r["relative_rmse_reduction_percent"], r["model"]))[0] if candidates else None


def paired_cluster_interval(frame, repetitions=2000):
    """Resample global scaffold clusters with shared weights across folds."""
    folds, clusters = sorted(frame.fold.unique()), sorted(frame.cluster.unique())
    sums = np.zeros((len(folds), len(clusters)))
    counts = np.zeros_like(sums)
    for row in frame.itertuples():
        i, j = folds.index(row.fold), clusters.index(row.cluster)
        sums[i, j] += row.sum_delta
        counts[i, j] += row.n_surfaces
    estimate = float(np.mean(sums.sum(1) / counts.sum(1)))
    weights = np.random.default_rng(20261003).multinomial(len(clusters), np.full(len(clusters), 1/len(clusters)), size=repetitions)
    denominator = weights @ counts.T
    keep = (denominator > 0).all(1)
    if keep.sum() < repetitions * .9:
        raise RuntimeError("Too few valid global-cluster bootstrap draws")
    bootstrap = ((weights[keep] @ sums.T) / denominator[keep]).mean(1)
    low, high = np.quantile(bootstrap, [.025, .975])
    return dict(mean_rmse_reduction=estimate, ci95_low=float(low), ci95_high=float(high),
                bootstrap_repetitions=repetitions, valid_draws=int(keep.sum()), clusters=len(clusters),
                scope="fixed_trained_models_development_scaffold_cluster_bootstrap")


class Queue:
    def __init__(self):
        self.cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        SUITE.mkdir(parents=True, exist_ok=True)
        self.state = read_json(SUITE / "queue_progress.json") if (SUITE / "queue_progress.json").exists() else {
            "status": "preparing", "started": now(), "steps": {}, "test_response_rows_read": 0}
        self.state["pid"] = os.getpid()
        current = datetime.now(timezone(timedelta(hours=9)))
        initial_age = (current - datetime.fromisoformat(self.state["started"])).total_seconds()
        self.initial_wait_deadline = time.monotonic() + max(0., self.cfg["gpu_initial_wait_seconds"] - initial_age)
        work_started = self.state.get("gpu_work_started")
        work_age = (current - datetime.fromisoformat(work_started)).total_seconds() if work_started else None
        self.deadline = time.monotonic() + max(0., self.cfg["suite_total_wall_seconds"] - work_age) if work_age is not None else None
        self.sources = [record(p) for p in [CONFIG, ROOT / self.cfg["protocol_document"],
            SUITE / "diagnostic_summary.json"] + list((ROOT / "src/bisurfgnn").rglob("*.py"))
            + list((ROOT / "experiments/model_improvement").glob("*.py"))]
        binding = SUITE / "suite_binding.json"
        if binding.exists():
            if read_json(binding)["sources"] != self.sources:
                raise RuntimeError("Suite source or config drift; refusing silent resume")
        else:
            atomic_json(binding, {"sources": self.sources, "frozen_at": now()})
        self.regression_count = validate_regression(SUITE / "regression.xml")

    def save(self, status=None, **extra):
        if status:
            self.state["status"] = status
        self.state.update(extra, updated=now())
        atomic_json(SUITE / "queue_progress.json", self.state)

    def note(self, text):
        guide = ROOT / "docs/MODEL_IMPROVEMENT_PLAN.md"
        content = guide.read_text(encoding="utf-8")
        block = f"{BEGIN}\n\n### 剩余模型实验自动队列 v1（程序维护）\n\n更新时间：{now()}。\n\n{text}\n\n实时记录：`runs/model_improvement/remaining_suite_v1/queue_progress.json`。实验协议：`docs/REMAINING_SUITE_PROTOCOL.md`。\n\n{END}"
        if BEGIN in content:
            if END not in content:
                raise RuntimeError("Progress marker is damaged")
            before, remainder = content.split(BEGIN, 1)
            _, after = remainder.split(END, 1)
            content = before + block + after
        else:
            content += "\n\n" + block + "\n"
        lines = content.splitlines()
        status_name = {"waiting_gpu": "等待 GPU", "running": "训练中", "complete": "已完成",
                       "needs_attention": "运行异常待检查"}.get(self.state["status"], self.state["status"])
        for index, line in enumerate(lines):
            if line.startswith("建立日期："):
                lines[index] = f"建立日期：2026-10-01（日本时间）。最近更新：{now()[:10]}。状态：对比模型、方案 A/B 开发轮已完成；剩余模型矩阵自动队列{status_name}，见第 8 节。"
                break
        content = "\n".join(lines) + "\n"
        guide.write_text(content, encoding="utf-8")
        print(text, flush=True)

    def wait_gpu(self, stage):
        self.save("waiting_gpu", current_stage=stage)
        stable = 0
        self.note(f"阶段 `{stage}` 等待 GPU：需连续三次显存可用至少 2500 MiB、利用率不高于 85%。不结束其他训练。已通过完整回归 {self.regression_count} 项。")
        while stable < 3:
            if time.monotonic() > (self.deadline or self.initial_wait_deadline):
                raise TimeoutError("Frozen queue/waiting budget exhausted")
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                    text=True, capture_output=True, check=True, timeout=20)
            utilization, used, total = [int(value.strip()) for value in result.stdout.strip().splitlines()[0].split(",")]
            free = total - used
            self.save(gpu_utilization_percent=utilization, gpu_free_mib=free, gpu_memory_used_mib=used,
                      consecutive_ready_observations=stable)
            stable = stable + 1 if free >= self.cfg["gpu_minimum_free_mib"] and utilization <= self.cfg["gpu_max_utilization_percent"] else 0
            if stable < 3:
                time.sleep(15)
        if self.deadline is None:
            self.deadline = time.monotonic() + self.cfg["suite_total_wall_seconds"]
            self.save(gpu_work_started=now())

    def stage(self, name, models, *, preflight=False, fold=0):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(models=models, fold=fold, report_prefix=None if preflight else name,
                   stage="preflight" if preflight else "development", max_wall_seconds=600 if preflight else 3600)
        if preflight:
            cfg["seeds"] = [41]
            cfg["training"]["epochs"] = 2
        configs = SUITE / "configs"
        configs.mkdir(exist_ok=True)
        config_path = configs / f"{name}.yaml"
        if config_path.exists() and yaml.safe_load(config_path.read_text(encoding="utf-8")) != cfg:
            raise RuntimeError("Stage configuration drift")
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        output = ROOT / "runs/model_improvement" / name
        if output.exists():
            summary = verify_stage(output, cfg)
            self.state["steps"][name] = dict(status="passed", resumed=True, wall_seconds=summary["wall_seconds"], jobs=len(summary["results"]))
            self.save()
            return output
        self.wait_gpu(name)
        if self.deadline - time.monotonic() < cfg["max_wall_seconds"] + 60:
            raise TimeoutError("Insufficient frozen total budget for next stage")
        log = SUITE / "logs" / name
        log.mkdir(parents=True, exist_ok=False)
        env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONIOENCODING": "utf-8"}
        args = [sys.executable, "-u", str(ROOT / "experiments/model_improvement/run_remaining_variant_experiment.py"),
                "--config", str(config_path), "--output-dir", str(output)]
        with (log / "stdout.log").open("w", encoding="utf-8") as stdout, (log / "stderr.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            atomic_json(log / "process.json", {"pid": process.pid, "started": now(), "args": args})
            self.save("running", current_stage=name, child_pid=process.pid)
            self.note(f"正在执行 `{name}`，fold {fold}，{len(models)} 个版本，{len(cfg['seeds'])} 个种子、{cfg['training']['epochs']} 轮；每个版本另做一次完整重跑。后台子进程 {process.pid}。测试响应禁读。")
            while process.poll() is None:
                if time.monotonic() > self.deadline:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=stdout, stderr=stderr, check=False)
                    else:
                        process.terminate()
                    raise TimeoutError("Frozen suite deadline; terminated only this queue's child tree")
                progress = output / "progress.json"
                if progress.exists():
                    self.save(current_stage_progress=read_json(progress))
                time.sleep(15)
            if process.returncode != 0:
                raise RuntimeError(f"Stage {name} failed with exit {process.returncode}; see {log / 'stderr.log'}")
        summary = verify_stage(output, cfg)
        self.state["steps"][name] = dict(status="passed", wall_seconds=summary["wall_seconds"], jobs=len(summary["results"]))
        self.save("between_stages", child_pid=None)
        return output

    def finish(self, stages, winner, extra=None):
        for item in self.sources:
            if record(item["path"])["sha256"] != item["sha256"]:
                raise RuntimeError("Queue source/config changed while running")
        result = {"status": "complete", "stages": self.state["steps"], "selected_candidate": winner,
                  "dose_interaction": {"status": "skipped_diagnostic_gate", "reason": "fixed dose-bin RMSE ratio 1.000664 < 1.5"},
                  "test_response_rows_read": 0, "independent_confirmation": False, **(extra or {})}
        atomic_json(SUITE / "queue_summary.json", result)
        rows = []
        for stage in stages:
            frame = pd.read_csv(stage / "results.csv")
            frame = frame[frame.run == "primary"].copy()
            frame["stage"] = stage.name
            rows.append(frame)
        combined = pd.concat(rows, ignore_index=True)
        combined.to_csv(ROOT / "reports/tables/remaining_suite_v1_results.csv", index=False)
        table = combined.groupby(["stage", "model"]).surface_macro_rmse.agg(["count", "mean", "std"])
        table.to_csv(ROOT / "reports/tables/remaining_suite_v1_summary.csv")
        self.save("complete", selected_candidate=winner, finished=now())
        message = "剩余有限矩阵已按协议执行结束：输出参数化与六组消融均完成，所有重复、ID 与有限性核对通过；剂量早交互因诊断条件不满足跳过。"
        message += "没有候选通过至少 4/5 种子及至少 1% 改善门槛，保留负结果，不启动多折。" if winner is None else f"开发候选 `{winner['model']}` 已补完五折比较，结果见 queue_summary.json 与汇总表；尚未独立确认。"
        self.note(message + "\n\n```text\n" + table.to_string() + "\n```\n\n总表：`reports/tables/remaining_suite_v1_summary.csv`。源码和数据未推送。")

    def run(self):
        diagnosis = read_json(SUITE / "diagnostic_summary.json")
        if diagnosis["dose_experiment_eligible"]:
            raise RuntimeError("Unexpected eligible dose experiment; frozen current suite requires the recorded negative gate")
        self.stage("residual_preflight_v1", self.cfg["models"], preflight=True)
        self.stage("factorial_preflight_v1", FACTORIAL, preflight=True)
        output = self.stage("residual_output_v1", self.cfg["models"])
        factorial = self.stage("factorial_exchange_v1", FACTORIAL)
        # The identical current model must reproduce across both stage configs.
        for seed in self.cfg["seeds"]:
            a = pd.read_csv(output / f"{BASE}_seed_{seed}_primary/predictions.csv")
            b = pd.read_csv(factorial / f"{BASE}_seed_{seed}_primary/predictions.csv")
            np.testing.assert_allclose(a.prediction, b.prediction, atol=0, rtol=0)
        stages = [output, factorial]
        winner = choose_candidate(stages)
        atomic_json(SUITE / "selection.json", {"winner": winner, "selection_scope": "observed_fold0_development"})
        if winner is None:
            self.finish(stages, None)
            return
        chosen_stage = Path(winner["source_stage"])
        fold_runs = [chosen_stage]
        for fold in range(1, 5):
            fold_runs.append(self.stage(f"remaining_multifold_f{fold}_v1", [BASE, winner["model"]], fold=fold))
        rows, effects = [], []
        for fold, run in enumerate(fold_runs):
            results = pd.read_csv(run / "results.csv"); results = results[results.run == "primary"]
            means = results.groupby("model").surface_macro_rmse.mean()
            effects.append(dict(fold=fold, base=float(means[BASE]), candidate=float(means[winner["model"]])))
            units = pd.read_csv(run / "validation_units.csv").drop_duplicates("surface_id").set_index("surface_id")
            differences = []
            for seed in self.cfg["seeds"]:
                base = pd.read_csv(run / f"{BASE}_seed_{seed}_primary/predictions.csv")
                candidate = pd.read_csv(run / f"{winner['model']}_seed_{seed}_primary/predictions.csv")
                assert base.source_row.tolist() == candidate.source_row.tolist()
                np.testing.assert_allclose(base.target, candidate.target, atol=0, rtol=0)
                base["delta_sq"] = (base.prediction-base.target)**2
                candidate["delta_sq"] = (candidate.prediction-candidate.target)**2
                differences.append(np.sqrt(base.groupby("surface_id").delta_sq.mean())-
                                   np.sqrt(candidate.groupby("surface_id").delta_sq.mean()))
            delta = pd.concat(differences, axis=1).mean(axis=1)
            part = pd.DataFrame(dict(cluster=units.loc[delta.index, "cluster_id"], delta=delta))
            grouped = part.groupby("cluster").delta.agg(["sum", "count"])
            rows.extend(dict(fold=fold, cluster=cluster, sum_delta=float(r["sum"]), n_surfaces=int(r["count"])) for cluster, r in grouped.iterrows())
        effects = pd.DataFrame(effects)
        effects.to_csv(ROOT / "reports/tables/remaining_suite_v1_multifold.csv", index=False)
        interval = paired_cluster_interval(pd.DataFrame(rows))
        improvement = float((effects.base.mean()-effects.candidate.mean())/effects.base.mean())
        multifold = dict(fold_effects=effects.to_dict("records"), improved_folds=int((effects.candidate<effects.base).sum()),
            relative_rmse_reduction_percent=100*improvement, cluster_interval=interval,
            development_gate_passed=int((effects.candidate<effects.base).sum())>=4 and improvement>=.01,
            limitations="observed development selection; bootstrap does not cover all shared cell/partner dependencies or retraining uncertainty")
        self.finish(stages + fold_runs[1:], winner, {"multifold": multifold})


def main():
    queue = None
    try:
        queue = Queue()
        queue.run()
    except Exception as error:
        if queue is not None:
            queue.save("needs_attention", error=repr(error))
            queue.note(f"连续队列尚未完成：{error!r}。已通过的阶段保留；不把运行故障或预算耗尽当作负性能结果。")
        else:
            atomic_json(SUITE / "queue_failure.json", {"status": "needs_attention", "error": repr(error), "at": now()})
        raise


if __name__ == "__main__":
    main()

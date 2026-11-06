"""Automatically verify short runs, then execute the registered five-seed round."""
from __future__ import annotations
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fingerprint_queue_helpers", Path(__file__).with_name("run_remaining_suite.py"))
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)
SUITE = ROOT / "runs/model_improvement/fingerprint_suite_v1"
CONFIG = ROOT / "configs/model_improvement/fingerprint_fusion_v1.yaml"
BEGIN, END = "<!-- AUTO_FINGERPRINT_SUITE_BEGIN -->", "<!-- AUTO_FINGERPRINT_SUITE_END -->"


class Queue:
    def __init__(self):
        self.cfg = util.yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        SUITE.mkdir(parents=True, exist_ok=True)
        if (SUITE / "queue_progress.json").exists():
            raise RuntimeError("This queue is single-launch; inspect existing state instead of overwriting")
        self.state = dict(status="preparing", started=util.now(), pid=os.getpid(), steps={}, test_response_rows_read=0)
        self.regression_count = util.validate_regression(SUITE / "regression.xml")
        self.sources = [util.record(p) for p in [CONFIG, ROOT / self.cfg["protocol_document"]]
            + list((ROOT / "src/bisurfgnn").rglob("*.py"))
            + list((ROOT / "experiments/model_improvement").glob("*.py"))]
        util.atomic_json(SUITE / "suite_binding.json", dict(sources=self.sources, frozen_at=util.now(), regression_tests=self.regression_count))
        self.deadline = None
        self.wait_deadline = time.monotonic() + self.cfg["gpu_initial_wait_seconds"]
        self.child = None

    def save(self, status=None, **extra):
        if status:
            self.state["status"] = status
        self.state.update(extra, updated=util.now())
        util.atomic_json(SUITE / "queue_progress.json", self.state)

    def note(self, message):
        path = ROOT / "docs/MODEL_IMPROVEMENT_PLAN.md"
        content = path.read_text(encoding="utf-8")
        block = f"{BEGIN}\n\n### 分子指纹融合实验 v1（程序维护）\n\n更新时间：{util.now()}。\n\n{message}\n\n配置：`configs/model_improvement/fingerprint_fusion_v1.yaml`。协议：[FINGERPRINT_FUSION_PROTOCOL.md](FINGERPRINT_FUSION_PROTOCOL.md)。实时状态：`runs/model_improvement/fingerprint_suite_v1/queue_progress.json`。\n\n{END}"
        if BEGIN in content:
            before, rest = content.split(BEGIN, 1)
            _, after = rest.split(END, 1)
            content = before + block + after
        else:
            content += "\n\n" + block + "\n"
        path.write_text(content, encoding="utf-8")
        print(message, flush=True)

    def wait_gpu(self, stage):
        self.save("waiting_gpu", current_stage=stage)
        stable = 0
        while stable < 3:
            if time.monotonic() > (self.deadline or self.wait_deadline):
                raise TimeoutError("Frozen GPU waiting/queue budget exceeded")
            sample = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True, timeout=10,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            free, use = map(int, sample.stdout.strip().splitlines()[0].split(","))
            stable = stable + 1 if free >= self.cfg["gpu_minimum_free_mib"] and use <= self.cfg["gpu_max_utilization_percent"] else 0
            self.save(gpu=dict(free_mib=free, utilization_percent=use, stable_samples=stable))
            if stable < 3:
                time.sleep(15)
        if self.deadline is None:
            self.deadline = time.monotonic() + self.cfg["suite_total_wall_seconds"]
            self.save(gpu_work_started=util.now())

    def stage(self, name, cfg):
        self.wait_gpu(name)
        for record in self.sources:
            if util.record(record["path"]) != record:
                raise RuntimeError("Source/config drift during queue")
        configs = SUITE / "configs"
        configs.mkdir(exist_ok=True)
        config_path = configs / f"{name}.yaml"
        config_path.write_text(util.yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        output = ROOT / "runs/model_improvement" / name
        if output.exists():
            raise RuntimeError("Output exists; refusing overwrite")
        logs = SUITE / "logs" / name
        logs.mkdir(parents=True)
        command = [sys.executable, str(Path(__file__).with_name("run_fingerprint_experiment.py")),
                   "--config", str(config_path), "--output-dir", str(output)]
        with (logs / "stdout.log").open("w", encoding="utf-8") as stdout, (logs / "stderr.log").open("w", encoding="utf-8") as stderr:
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.save("running", child_pid=self.child.pid, current_stage=name, current_stage_progress=None)
            self.note(f"阶段 `{name}` 已进入训练；完成短轮后自动开始五种子正式轮。完整回归 {self.regression_count} 项通过。")
            while self.child.poll() is None:
                if time.monotonic() > self.deadline:
                    self.child.terminate()
                    self.child.wait(timeout=30)
                    raise TimeoutError("Frozen queue budget exceeded; only this queue child stopped")
                progress = output / "progress.json"
                if progress.exists():
                    self.save(current_stage_progress=util.read_json(progress))
                time.sleep(15)
            code = self.child.returncode
        self.child = None
        self.save(child_pid=None)
        if code != 0:
            raise RuntimeError(f"{name} failed, exit={code}; inspect {logs}")
        summary = util.verify_stage(output, cfg)
        self.state["steps"][name] = dict(status="passed", jobs=len(summary["results"]), wall_seconds=summary["wall_seconds"])
        self.save(current_stage_progress=util.read_json(output / "progress.json"))
        return output

    def run(self):
        pre = copy.deepcopy(self.cfg)
        pre.update(stage="preflight", seeds=[41], report_prefix=None, max_wall_seconds=600)
        pre["training"]["epochs"] = 2
        self.stage("fingerprint_preflight_v1", pre)
        output = self.stage("fingerprint_fusion_v1", self.cfg)
        decision = util.read_json(output / "development_decision.json")
        eligible = [r for r in decision["candidates"] if r["model"] in {"bisurf_fp_only_v2", "bisurf_fp_fused_v2"} and r["advance_to_multifold"]]
        winner = sorted(eligible, key=lambda r: -r["relative_rmse_reduction_percent"])[0] if eligible else None
        summary = dict(status="complete", steps=self.state["steps"], winner=winner,
                       next_step="freeze_multifold_development" if winner else "record_negative_result_then_diagnose_input_paths",
                       test_response_rows_read=0, completed=util.now())
        util.atomic_json(SUITE / "queue_summary.json", summary)
        self.save("complete", winner=winner, child_pid=None, current_stage=None, current_stage_progress=None)
        scores = util.pd.read_csv(output / "seed_statistics.csv")
        text = "短验证8任务、五种子正式轮24任务完成；重复、ID、有限预测、检查点与划分审计通过。"
        text += "有候选通过门槛，下一步冻结多折开发。" if winner else "没有分子特征候选通过门槛，保留本轮负结果。"
        self.note(text + "\n\n```text\n" + scores.to_string(index=False) + "\n```\n\n结果：`reports/tables/fingerprint_fusion_v1_summary.csv`。区间只表示固定开发样本的训练种子变异。")


if __name__ == "__main__":
    queue = None
    try:
        queue = Queue()
        queue.run()
    except Exception as error:
        if queue is not None:
            queue.save("needs_attention", error=repr(error))
            queue.note(f"实验停止，待检查：{error!r}。尚不能宣称完成；保留已有日志。")
        raise

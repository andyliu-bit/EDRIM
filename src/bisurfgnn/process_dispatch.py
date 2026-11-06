"""Frozen serial subprocess dispatch with receipts, fail-closed restart and no retry.

This is a scheduling primitive, not permission to run an unreviewed full matrix.
Completed receipts are checked, not blindly trusted. A worker without a receipt
after a scheduler crash requires manual review, never automatic rerun.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from .learned_benchmark import _canonical_hash, _file_record, _write_json_exclusive
from .resumable_training import exclusive_job


def _verify(records):
    for record in records:
        if _file_record(Path(record["path"])) != record:
            raise ValueError("Frozen dispatch file changed")


def dispatch(directory, jobs, source_records, environment, *, timeout_seconds=180):
    root = Path(directory).resolve()
    if not jobs or not source_records or len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Unique jobs and source bindings are required")
    for job in jobs:
        if not job["command"] or not Path(job["receipt"]).resolve().is_relative_to(root):
            raise ValueError("Worker receipt must be inside dispatch directory")
    plan = dict(version="serial_dispatch_v1", jobs=jobs, files=source_records,
                environment=environment, timeout_seconds=timeout_seconds)
    plan_hash = _canonical_hash(plan)
    with exclusive_job(root):
        _verify(source_records)
        plan_path = root/"dispatch_plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
                raise ValueError("Frozen dispatch plan changed")
        else:
            _write_json_exclusive(plan_path, plan)
        results = []
        for index, job in enumerate(jobs):
            result_path = root/f"dispatch_{index:04d}_result.json"
            started_path = root/f"dispatch_{index:04d}_started.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                digest = result.pop("manifest_sha256")
                if digest != _canonical_hash(result) or result["plan_hash"] != plan_hash or result["job_id"] != job["id"]:
                    raise ValueError("Dispatch result integrity mismatch")
                if result["returncode"] != 0 or result.get("validation_error"):
                    raise ValueError("Failed job retained; no automatic retry")
                if result["receipt"]["path"] != str(Path(job["receipt"]).resolve()):
                    raise ValueError("Worker receipt path changed")
                _verify([result["receipt"], *result["artifacts"]])
                results.append({**result, "manifest_sha256": digest})
                continue
            if started_path.exists() or Path(job["receipt"]).exists():
                raise ValueError("Ambiguous previously started job requires inspection")
            _verify(source_records)
            _write_json_exclusive(started_path, dict(plan_hash=plan_hash, job_id=job["id"]))
            env = os.environ.copy()
            env.update(environment)
            process = subprocess.Popen(job["command"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            timed_out = False
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                # Only the explicitly launched direct worker, never arbitrary PIDs.
                process.kill()
                stdout, stderr = process.communicate()
                timed_out = True
            result = dict(plan_hash=plan_hash, job_id=job["id"], pid=process.pid,
                returncode=process.returncode, timed_out=timed_out, stdout=stdout, stderr=stderr,
                receipt=None, artifacts=[], validation_error=None)
            if process.returncode == 0 and not timed_out:
                try:
                    receipt_path = Path(job["receipt"]).resolve()
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    digest = receipt.pop("manifest_sha256")
                    if digest != _canonical_hash(receipt):
                        raise ValueError("Worker receipt integrity mismatch")
                    artifacts = [receipt[key] for key in job["artifact_keys"]]
                    if any(not Path(r["path"]).resolve().is_relative_to(root) for r in artifacts):
                        raise ValueError("Worker artifact escapes dispatch directory")
                    _verify(artifacts)
                    result.update(receipt=_file_record(receipt_path), artifacts=artifacts)
                except (ValueError, KeyError, OSError) as exc:
                    result["validation_error"] = str(exc)
            result["manifest_sha256"] = _canonical_hash(result)
            _write_json_exclusive(result_path, result)
            if process.returncode != 0 or timed_out or result["validation_error"]:
                raise RuntimeError(f"Worker {job['id']} failed; see {result_path}")
            _verify(source_records)
            results.append(result)
        return results

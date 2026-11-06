"""Explicit observer overlay over an unchanged, frozen formal experiment.

Original receipts remain original. Additional execution provenance binds observer
sources separately; no original hash or checkpoint contract is rewritten.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import run_formal as f
from training_progress import Reporter, observe_training


def observer_sources():
    return [f.learned._file_record(Path(__file__).resolve()),
            f.learned._file_record(Path(__file__).with_name('training_progress.py').resolve())]


def worker(plan_path, job_id, overlay_path):
    overlay = f.read_json(overlay_path); f.check_seal(overlay); f.verify(overlay['sources'])
    plan = f.read_json(plan_path); f.check_seal(plan)
    if overlay['formal_plan_hash'] != plan['manifest_sha256']:
        raise ValueError('Observer/formal plan mismatch')
    root = Path(plan_path).resolve().parent
    job_dir = root/'jobs'/job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    f.seal(job_dir/'observer_invocation.json', dict(overlay=f.learned._file_record(Path(overlay_path).resolve()),
        sources=overlay['sources'], job_id=job_id, pid=os.getpid(), started=time.time(),
        note='Observer adds wall-time overhead outside measured batch region; old training sources unchanged.'))
    reporter = Reporter(job_dir/'progress.json', job_id)
    reporter.emit('loading')
    try:
        with observe_training(f, reporter):
            result = f.worker(plan_path, job_id)
        f.verify(overlay['sources'])
        reporter.emit('finished')
        return result
    except BaseException as exc:
        reporter.emit('failed', error=str(exc))
        raise


def launch(plan_path, readiness, overlay_path):
    plan_path = Path(plan_path).resolve(); root = plan_path.parent
    plan = f.read_json(plan_path); ready = f.read_json(readiness)
    overlay = f.read_json(overlay_path)
    for obj in (plan, ready, overlay): f.check_seal(obj)
    if ready['status'] != 'passed' or overlay['formal_plan_hash'] != plan['manifest_sha256']:
        raise ValueError('Readiness/overlay mismatch')
    f.verify(plan['sources']+ready['sources']+ready['evidence']+overlay['sources']+overlay['acceptance'])
    if plan['sources'] != ready['sources']: raise ValueError('Readiness source mismatch')
    original_dispatch = f.read_json(root/'dispatch_plan.json')
    original_hash = f.learned._canonical_hash(original_dispatch)
    # Results sealed by a superseded observer overlay stay valid; a replacement overlay
    # must name every overlay hash it inherits, so unknown lineage still fails loudly.
    lineage = {original_hash, overlay['manifest_sha256'], *overlay.get('inherited_overlays', ())}
    with f.exclusive_job(root):
        for index, job in enumerate(plan['inventory']['jobs']):
            dest = root/f'dispatch_{index:04d}_result.json'
            start = root/f'dispatch_{index:04d}_started.json'
            if dest.exists():
                result = f.read_json(dest); f.check_seal(result)
                if result['plan_hash'] not in lineage:
                    raise ValueError('Unknown dispatcher lineage')
                if result['job_id'] != job['id'] or result['returncode'] != 0 or result.get('validation_error'):
                    raise ValueError('Failed or mismatched task requires review')
                f.verify([result['receipt'], *result['artifacts']])
                receipt = f.read_json(result['receipt']['path'])
                f.verify_receipts({'jobs':[job]}, [receipt], plan['manifest_sha256'])
                continue
            if start.exists() or (root/'jobs'/job['id']/'receipt.json').exists():
                raise ValueError('Started/uncommitted task requires manual review')
            f.verify(plan['sources']+overlay['sources'])
            f.learned._write_json_exclusive(start, dict(job_id=job['id'],plan_hash=overlay['manifest_sha256']))
            command = [sys.executable,'-u',str(Path(__file__).resolve()),'worker','--plan',str(plan_path),
                       '--job-id',job['id'],'--overlay',str(Path(overlay_path).resolve())]
            print(f'START {index+1}/{len(plan["inventory"]["jobs"])} {job["id"]}', flush=True)
            stdout_path = root/f'dispatch_{index:04d}.stdout.log'
            stderr_path = root/f'dispatch_{index:04d}.stderr.log'
            timed_out = False
            with stdout_path.open('x', encoding='utf-8') as stdout, stderr_path.open('x', encoding='utf-8') as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                try:
                    code = process.wait(timeout=plan['config']['worker_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    process.kill(); code = process.wait(); timed_out = True
            result = dict(plan_hash=overlay['manifest_sha256'], job_id=job['id'], pid=process.pid,
                returncode=code, timed_out=timed_out, stdout=stdout_path.read_text(encoding='utf-8'),
                stderr=stderr_path.read_text(encoding='utf-8'), receipt=None, artifacts=[], validation_error=None,
                command=command, observer=f.learned._file_record(Path(overlay_path).resolve()))
            if code == 0 and not timed_out:
                try:
                    path = root/'jobs'/job['id']/'receipt.json'; receipt = f.read_json(path)
                    f.verify_receipts({'jobs':[job]},[receipt],plan['manifest_sha256'])
                    result.update(receipt=f.learned._file_record(path),
                                  artifacts=[receipt[k] for k in ('prediction','errors','checkpoint')])
                except (ValueError, KeyError, OSError) as exc:
                    result['validation_error'] = str(exc)
            f.seal(dest, result)
            if code != 0 or timed_out or result['validation_error']:
                raise RuntimeError(f'Task failed; inspect {dest}')
            print(f'DONE {index+1}/{len(plan["inventory"]["jobs"])} {job["id"]}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['worker','launch'])
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--overlay', type=Path, required=True)
    p.add_argument('--readiness', type=Path); p.add_argument('--job-id')
    a = p.parse_args()
    if a.action == 'worker': worker(a.plan, a.job_id, a.overlay)
    else: launch(a.plan, a.readiness, a.overlay)

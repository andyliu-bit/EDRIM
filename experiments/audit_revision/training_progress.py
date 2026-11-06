"""Versioned, read-only training observer; never changes batches or RNG state.

Status publication is best effort. A reader holding the status file, a full disk,
or a non-finite metric must never interrupt the training it only observes, so every
publication failure is counted and dropped instead of raised.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time


def _publishable(value):
    # NaN/Inf have no JSON representation; report them as null rather than failing.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Reporter:
    def __init__(self, path, job, interval=5.0, attempts=5, backoff=0.05):
        self.path = Path(path)
        self.job = job
        self.interval = interval
        self.attempts = attempts
        self.backoff = backoff
        self.last = -float('inf')
        self.start = time.perf_counter()
        self.dropped = 0

    def _publish(self, row):
        """Publish one status row; return whether it reached the status file."""
        temporary = self.path.with_suffix('.tmp')
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps({k: _publishable(v) for k, v in row.items()}, allow_nan=False)
            temporary.write_text(payload, encoding='utf-8')
            for attempt in range(self.attempts):
                try:
                    # Windows denies the replace while a reader holds the target open.
                    os.replace(temporary, self.path)
                    return True
                except OSError:
                    if attempt == self.attempts-1:
                        raise
                    time.sleep(self.backoff*(attempt+1))
        except Exception as exc:
            self.dropped += 1
            if self.dropped in (1, 10, 100, 1000):
                try:
                    print(f'{self.job} status_publication_dropped count={self.dropped} '
                          f'error={exc}', file=sys.stderr, flush=True)
                except Exception:
                    pass  # Even the diagnostic sink may be closed or full.
        return False

    def emit(self, phase, *, force=True, **fields):
        try:
            self._emit(phase, force=force, **fields)
        except Exception:
            # Isolate formatting/serialization and closed output streams as well.
            # Do not catch BaseException: an explicit interrupt must remain effective.
            self.dropped += 1

    def _emit(self, phase, *, force=True, **fields):
        now = time.perf_counter()
        if not force and now-self.last < self.interval:
            return
        self.last = now
        row = dict(job=self.job, phase=phase, pid=os.getpid(), timestamp=time.time(),
                   attempt_seconds=now-self.start, **fields)
        self._publish(row)
        done, total = fields.get('batch', 0), fields.get('batches', 0)
        bar = ''
        if total:
            n = min(24, int(24*done/total))
            bar = f" [{'#'*n}{'-'*(24-n)}] batch {done}/{total}"
        detail = ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}'
                          for k, v in fields.items() if k not in ('batch', 'batches'))
        try:
            print(f'{self.job} {phase}{bar} {detail}', flush=True)
        except OSError:
            self.dropped += 1


class ObservedBatches(list):
    def __init__(self, values, reporter, epoch):
        super().__init__(values)
        self.reporter, self.epoch = reporter, epoch

    def __iter__(self):
        start = time.perf_counter()
        self.reporter.emit('train', epoch=self.epoch+1, batch=0, batches=len(self))
        for i, batch in enumerate(super().__iter__(), 1):
            yield batch
            # Runs between complete batches, outside the worker's measured batch time.
            elapsed = time.perf_counter()-start
            self.reporter.emit('train', force=i == len(self), epoch=self.epoch+1,
                batch=i, batches=len(self), epoch_wall_seconds=elapsed,
                epoch_eta_seconds=elapsed/i*(len(self)-i))


@contextmanager
def observe_training(formal, reporter):
    original_train = formal.train_resumable
    original_batches = formal.graph._surface_batch_indices

    def observed_train(model, optimizer, directory, contract, train_epoch, evaluate, **kwargs):
        current_epoch = [-1]

        def epoch(model, optimizer, index, *args):
            current_epoch[0] = index
            formal.graph._surface_batch_indices = lambda *a, **kw: ObservedBatches(
                original_batches(*a, **kw), reporter, index)
            try:
                metrics = train_epoch(model, optimizer, index, *args)
            finally:
                formal.graph._surface_batch_indices = original_batches
            reporter.emit('epoch_train_done', epoch=index+1, loss=metrics['surface_mse'],
                training_seconds=metrics['training_seconds'], updates=metrics['updates'])
            return metrics

        def validation(model):
            reporter.emit('validation', epoch=current_epoch[0]+1)
            start = time.perf_counter()
            score = evaluate(model)
            reporter.emit('validation_done', epoch=current_epoch[0]+1,
                validation_rmse=float(score), validation_seconds=time.perf_counter()-start)
            return score

        return original_train(model, optimizer, directory, contract, epoch, validation, **kwargs)

    formal.train_resumable = observed_train
    try:
        yield
    finally:
        formal.train_resumable = original_train
        formal.graph._surface_batch_indices = original_batches

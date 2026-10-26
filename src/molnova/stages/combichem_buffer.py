"""Bounded worker result buffers with durable, idempotent restart journals."""
import json
import os
from pathlib import Path
import time

from molnova import database
from molnova.chemistry.combichem3d.logging_utils import logger


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ResultBuffer:
    def __init__(self, db, batch, cfg):
        self.db, self.batch = db, batch
        self.limit = cfg.get('result-buffer-size', 16)
        self.interval = cfg.get('result-buffer-seconds', 2.)
        self.path = Path(db).parent / 'result-buffer' / f"batch-{batch['id']}.jsonl"
        self.records = []
        self.started = time.monotonic()

    def __enter__(self):
        if self.path.exists():
            raise RuntimeError('Unrecovered result journal; restart under the stage lock')
        return self

    def append(self, ordinal, report, payload):
        # Each journal is owned by one claimed batch. Fsync before exposing the
        # result as buffered, so SIGKILL/failed DB commits retain completed work.
        created = not self.path.exists()
        self.path.parent.mkdir(exist_ok=True)
        row = dict(batch=self.batch, ordinal=ordinal, report=report, payload=payload)
        encoded = (json.dumps(row) + '\n').encode('utf-8')
        with self.path.open('ab') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if created:
            _sync_directory(self.path.parent)
            _sync_directory(self.path.parent.parent)
        self.records.append((ordinal, report, payload))
        if len(self.records) >= self.limit or time.monotonic() - self.started >= self.interval:
            self.flush()

    def flush(self):
        if not self.records:
            return
        database.growth_save_attempts(self.db, self.batch, self.records)
        # If killed after commit but before unlink, replay deduplicates ordinals.
        self.path.unlink()
        _sync_directory(self.path.parent)
        logger.debug('result buffer committed batch=%s count=%s', self.batch['id'], len(self.records))
        self.records.clear()
        self.started = time.monotonic()

    def __exit__(self, *exc):
        self.flush()


def recover(db):
    """Recovery worker runs alone, before session claims can be rearmed."""
    directory = Path(db).parent / 'result-buffer'
    for path in sorted(directory.glob('batch-*.jsonl')):
        rows = []
        with path.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    logger.warning('Incomplete result journal tail retained for recalculation: %s', path.name)
                    break
                rows.append(json.loads(line))
        if rows:
            batch = rows[0]['batch']
            if path.name != f"batch-{batch['id']}.jsonl" or any(r['batch'] != batch for r in rows):
                raise ValueError(f'Inconsistent result journal: {path}')
            ordinals = [r['ordinal'] for r in rows]
            if ordinals != list(range(ordinals[0], ordinals[0] + len(rows))):
                raise ValueError(f'Non-contiguous result journal: {path}')
            database.growth_save_attempts(db, batch,
                [(r['ordinal'], r['report'], r['payload']) for r in rows])
            database.growth_finish_batch(db, batch)
            logger.info('recovered result buffer batch=%s count=%s', batch['id'], len(rows))
        path.unlink()
        _sync_directory(directory)

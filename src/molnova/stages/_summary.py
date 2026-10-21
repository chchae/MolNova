"""Read-only completion summaries; native timings are distinct from throughput."""
import json
import math
from pathlib import Path
import re
import statistics

from molnova.paths import iteration_directory, stage_in_iteration
from molnova import database


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _native_times(output, iteration, stage, successful):
    """One duration per original compound, from verified completed run records."""
    directory = iteration_directory(Path(output), iteration)
    times = {}
    if stage == 'mmgbsa':
        root = stage_in_iteration(directory, 'mmgbsa')
        for path in sorted(root.glob('queue-*.completed.json')):
            record = _read_json(path)
            if not isinstance(record, dict) or record.get('mode') != 'compound_queue_v1':
                continue
            run_dir = record.get('run_dir', '')
            entries = record.get('entries')
            if (not isinstance(run_dir, str) or not re.fullmatch(r'queue-\d+', run_dir)
                    or not isinstance(entries, dict)):
                continue
            for cid in successful - times.keys():
                if entries.get(str(cid)) != 'done':
                    continue
                try:
                    log = (root / run_dir / f'CMPID_{cid}' / 'mmgbsa_input.log').read_text()
                except OSError:
                    continue
                matches = re.findall(
                    r'Total Time Elapsed \(hours:minutes:seconds\)\s+(\d+):(\d+):(\d+)', log)
                if len(matches) == 1:
                    h, m, s = map(int, matches[0])
                    times[cid] = h * 3600 + m * 60 + s
    elif stage == 'amber-gbsa':
        # Older remote workers did not record durations; never invent them.
        snapshot = _read_json(stage_in_iteration(directory, 'amber_gbsa') / 'remote_snapshot.json')
        if isinstance(snapshot, dict) and isinstance(snapshot.get('results'), list):
            for result in snapshot.get('results', []):
                if not isinstance(result, dict):
                    continue
                cid, duration = result.get('id'), result.get('elapsed_s')
                if (isinstance(cid, int) and not isinstance(cid, bool) and
                        cid in successful and 'error' not in result and
                        isinstance(duration, (int, float)) and not isinstance(duration, bool)
                        and math.isfinite(duration) and duration >= 0):
                    times[cid] = duration
    return list(times.values())


def completion_summary(args, iteration, stage, elapsed):
    successful, failed = database.stage_summary_results(args.db_path, iteration, stage)
    count = len(successful) + len(failed)
    fields = [f'compounds={count} (ok={len(successful)}, failed={len(failed)})']
    fields.append(f'wall/compound={elapsed / count:.2f}s' if count else 'wall/compound=n/a')
    if stage in {'mmgbsa', 'amber-gbsa'}:
        values = _native_times(args.output, iteration, stage, successful)
        if values:
            fields.append(f'native job n={len(values)}/{len(successful)} '
                          f'avg/median/min/max={statistics.mean(values):.1f}/'
                          f'{statistics.median(values):.1f}/{min(values):.1f}/{max(values):.1f}s')
        else:
            fields.append('native job time=n/a')
    return '; '.join(fields)

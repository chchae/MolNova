"""Read-only chemical-space reports; no calculation, selection or workflow writes."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from threadpoolctl import threadpool_limits

from molnova import _core, database
from molnova.config import distribution_project

METRICS = {
    'rdk_mw': 'Molecular weight', 'rdk_clogp': 'cLogP', 'rdk_tpsa': 'TPSA',
    'rdk_qed': 'QED', 'rdk_sa_score': 'SA', 'docking_score': 'Glide',
    'amber_gbsa_score': 'Amber GBSA', 'primg_gbsa_score': 'primg_gbsa', 'fep_score': 'FEP',
}
MAP_LIMIT = 2000
DIVERSITY_LIMIT = 200
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(value)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


@contextmanager
def report_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.distribution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def sample(rows, limit):
    """Stable, score-independent sample; never assume ID order is random."""
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{row['id']}:{row['smiles']}".encode()).digest())[:limit]


def balanced_sample(groups, limit):
    ordered = [sample(rows, limit) for _, rows in sorted(groups.items())]
    result = []
    for index in range(max(map(len, ordered), default=0)):
        for rows in ordered:
            if index < len(rows):
                result.append(rows[index])
                if len(result) == limit:
                    return result
    return result


def metric_values(row):
    return {name: (row.get(name) if finite(row.get(name)) else
                   row.get('sa_score') if name == 'rdk_sa_score' and finite(row.get('sa_score')) else None)
            for name in METRICS}


def statistics(values, population, bounds):
    values = np.asarray([v for v in values if finite(v)], dtype=float)
    if not len(values):
        return {'n': 0, 'missing': population, 'min': None, 'q1': None, 'median': None,
                'q3': None, 'max': None, 'mean': None, 'hist': [0] * 20}
    low, high = bounds
    counts, _ = np.histogram(values, bins=20, range=(low, high))
    return dict(zip(('min', 'q1', 'median', 'q3', 'max'), map(float, np.quantile(values, [0, .25, .5, .75, 1]))),
                n=len(values), missing=population - len(values), mean=float(values.mean()),
                hist=counts.tolist())


def fingerprint(row):
    mol = Chem.MolFromSmiles(row['smiles'])
    return FP.GetFingerprint(mol) if mol is not None and mol.GetNumAtoms() else None


def fingerprint_array(fps):
    data = np.zeros((len(fps), 2048), dtype=np.float32)
    for index, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, data[index])
    return data


def coordinates(rows, directory, through):
    """Freeze the first non-degenerate generated population's PCA axes."""
    valid = [(row, fingerprint(row)) for row in rows]
    invalid = [row['id'] for row, fp in valid if fp is None]
    valid = [(row, fp) for row, fp in valid if fp is not None]
    path = directory / 'chemical_space_pca.json'
    model = json.loads(path.read_text()) if path.exists() else None
    if not valid:
        return [], invalid, {'method': 'Morgan r2/2048 + PCA', 'status': 'No valid molecules'}
    matrix = fingerprint_array([fp for _, fp in valid])
    if model is None:
        if len(matrix) < 2 or np.all(matrix == matrix[0]):
            xy = np.zeros((len(matrix), 2))
            return [(row, float(x), float(y)) for (row, _), (x, y) in zip(valid, xy)], invalid, {
                'method': 'Morgan r2/2048 + PCA', 'status': 'Insufficient structural variation; axes not frozen'}
        from sklearn.decomposition import PCA
        with threadpool_limits(limits=1):
            pca = PCA(n_components=2, svd_solver='randomized', random_state=42).fit(matrix)
        model = {'version': 1, 'method': 'Morgan r2/2048 + PCA', 'trained_through': through,
                 'training_count': len(matrix), 'mean': pca.mean_.tolist(),
                 'components': pca.components_.tolist(),
                 'explained_variance_ratio': pca.explained_variance_ratio_.tolist()}
        # Reference-only previews must not pin axes to a handful of seed structures.
        if any(row['iteration'] > 0 for row, _ in valid):
            atomic_text(path, json_text(model))
    mean = np.asarray(model['mean'], dtype=float)
    components = np.asarray(model['components'], dtype=float)
    if mean.shape != (2048,) or components.shape != (2, 2048) or not np.isfinite(components).all() or not np.isfinite(mean).all():
        raise ValueError('Invalid saved chemical-space PCA model; report retained')
    with threadpool_limits(limits=1):
        xy = (matrix - mean) @ components.T
    metadata = {key: value for key, value in model.items() if key not in {'mean', 'components'}}
    return [(row, float(x), float(y)) for (row, _), (x, y) in zip(valid, xy)], invalid, metadata


def molecule_image(smiles, directory):
    name = hashlib.sha256(smiles.encode()).hexdigest()[:24] + '.svg'
    path = directory / 'molecules' / name
    if not path.exists():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not mol.GetNumAtoms():
            return None
        draw = rdMolDraw2D.MolDraw2DSVG(320, 200)
        rdMolDraw2D.PrepareAndDrawMolecule(draw, mol)
        draw.FinishDrawing()
        atomic_text(path, draw.GetDrawingText())
    return f'molecules/{name}'


def diversity(rows, previous_elites):
    candidates = [(row, fingerprint(row)) for row in sample(rows, DIVERSITY_LIMIT)]
    fps = [fp for _, fp in candidates if fp is not None]
    pairs = [score for index, fp in enumerate(fps) for score in DataStructs.BulkTanimotoSimilarity(fp, fps[:index])]
    elites = [fingerprint(row) for row in previous_elites]
    elites = [fp for fp in elites if fp is not None]
    nearest = [max(DataStructs.BulkTanimotoSimilarity(fp, elites)) for fp in fps] if elites else []
    return {'sample_n': len(fps), 'pair_n': len(pairs),
            'internal_diversity': 1 - float(np.mean(pairs)) if pairs else None,
            'previous_elite_n': len(elites),
            'previous_elite_similarity_median': float(np.median(nearest)) if nearest else None}


def substituents(rows, snapshot, directory):
    """Use the saved scaffold's attachment order; never infer a new scaffold."""
    scaffold = snapshot.get('scaffold_smiles') if snapshot else None
    if not scaffold:
        return {'status': 'No saved generation scaffold', 'sample_n': 0, 'matched_n': 0, 'sites': []}
    selected = sample(rows, DIVERSITY_LIMIT)
    try:
        _, _, labels, _, _ = _core._scaffold_attachment_layout(scaffold)
    except (RuntimeError, ValueError):
        return {'status': 'Saved scaffold has no usable attachment layout', 'sample_n': len(selected), 'matched_n': 0, 'sites': []}
    counts = [Counter() for _ in labels]
    matched = 0
    for row in selected:
        result = _core.decompose_elite_for_libinvent(row['smiles'], scaffold)
        if result is None:
            continue
        parts = result[1].split('|')
        if len(parts) != len(counts):
            continue
        matched += 1
        for counter, part in zip(counts, parts):
            counter[part] += 1
    return {'status': 'Sampled attachment frequencies; compare only identical saved scaffolds',
            'scaffold': scaffold, 'scaffold_key': hashlib.sha256(scaffold.encode()).hexdigest()[:12],
            'image': molecule_image(scaffold, directory), 'sample_n': len(selected), 'matched_n': matched,
            'sites': [{'label': label, 'unique_n': len(counter), 'top': [
                {'smiles': smiles, 'count': count, 'fraction': count / matched,
                 'image': molecule_image(smiles, directory)}
                for smiles, count in sorted(counter.items(), key=lambda entry: (-entry[1], entry[0]))[:6]]}
                for label, counter in zip(labels, counts)]}


def build_report(rows, scaffolds, directory, through, elite_count=10):
    groups = defaultdict(list)
    for row in rows:
        row.update(metric_values(row))
        groups[row['iteration']].append(row)
    bounds = {}
    for metric in METRICS:
        values = [row[metric] for row in rows if finite(row[metric])]
        low, high = (min(values), max(values)) if values else (0, 1)
        if low == high:
            low, high = low - .5, high + .5
        bounds[metric] = [low, high]
    elites = {iteration: sorted((row for row in group if finite(row['primg_gbsa_score'])),
                               key=lambda row: (row['primg_gbsa_score'], row['id']))[:elite_count]
              for iteration, group in groups.items()}
    summaries = []
    for iteration, group in sorted(groups.items()):
        cohorts = {'all': group, 'selected': [row for row in group if row['druglike_selected'] == 1],
                   'primg_gbsa_scored': [row for row in group if finite(row['primg_gbsa_score'])]}
        summaries.append({'iteration': iteration, 'n': len(group),
            'states': dict(Counter(row['state'] for row in group)),
            'selection_unknown': sum(row['druglike_selected'] is None for row in group),
            'elite_ids': [row['id'] for row in elites[iteration]],
            'cohorts': {name: {'n': len(population), 'metrics': {
                metric: statistics([row[metric] for row in population], len(population), bounds[metric])
                for metric in METRICS}} for name, population in cohorts.items()},
            'diversity': diversity(group, elites.get(iteration - 1, [])) if iteration else diversity(group, []),
            'substituents': substituents(group, scaffolds.get(iteration), directory)})
    sampled = balanced_sample(groups, MAP_LIMIT)
    xy, invalid, projection = coordinates(sampled, directory, through)
    elite_ids = {row['id'] for entries in elites.values() for row in entries}
    points = [{'id': row['id'], 'name': row['name'], 'smiles': row['smiles'],
               'iteration': row['iteration'], 'state': row['state'], 'x': x, 'y': y,
               'selected': row['druglike_selected'], 'elite': row['id'] in elite_ids,
               'image': molecule_image(row['smiles'], directory),
               'values': {metric: row[metric] for metric in METRICS}}
              for row, x, y in xy]
    return {'through_iteration': through, 'population_n': len(rows), 'map_n': len(points),
            'map_limit': MAP_LIMIT, 'invalid_sample_ids': invalid, 'projection': projection,
            'metrics': METRICS, 'bounds': bounds, 'iterations': summaries, 'points': points}


def log_summary(report):
    lines = [f"Population={report['population_n']}; map={report['map_n']} (balanced sample cap={MAP_LIMIT}); "
             'Morgan r2/2048 + fixed PCA; sampled Tanimoto diversity; lower energies rank better.']
    for item in report['iterations']:
        iteration = item['iteration']
        d = item['diversity']
        fmt = lambda value: f'{value:.3f}' if finite(value) else 'n/a'
        lines.append(f"iter{iteration:02d}: n={item['n']}, druglike_pass={item['cohorts']['selected']['n']}, "
                     f"selection_unknown={item['selection_unknown']}, diversity={fmt(d['internal_diversity'])} "
                     f"(sample={d['sample_n']}, pairs={d['pair_n']}), "
                     f"previous_primg_gbsa_elite_similarity={fmt(d['previous_elite_similarity_median'])} "
                     f"(elites={d['previous_elite_n']})")
        for metric, label in METRICS.items():
            stats = item['cohorts']['all']['metrics'][metric]
            bars = ''.join(' .:-=+*#%@'[min(9, int(9 * n / max(stats['hist'], default=1)))]
                           for n in stats['hist']) if any(stats['hist']) else 'n/a'
            lines.append(f"  {label}: n={stats['n']}/{item['n']}, missing={stats['missing']}, "
                         f"median={fmt(stats['median'])}, range=[{fmt(stats['min'])}, {fmt(stats['max'])}], "
                         f"hist_bounds=[{fmt(report['bounds'][metric][0])}, {fmt(report['bounds'][metric][1])}], hist={bars}")
        sub = item['substituents']
        lines.append(f"  Substituents: matched={sub['matched_n']}/{sub['sample_n']}; "
                     f"scaffold={sub.get('scaffold_key', 'n/a')}; " + ', '.join(
                         f"R{site['label']} unique={site['unique_n']}" for site in sub['sites']))
    return '\n'.join(lines)


def write_report(args, iteration=None, event='manual'):
    directory = args.output / 'analysis'
    with report_lock(directory):
        rows, scaffolds = database.distribution_snapshot(args.db_path, iteration)
        rows = [{key: None if isinstance(value, float) and not math.isfinite(value) else value
                 for key, value in row.items()} for row in rows]
        through = iteration if iteration is not None else max((row['iteration'] for row in rows), default=0)
        from molnova.distribution_html import TEMPLATE, render
        template_hash = hashlib.sha256(TEMPLATE.encode()).hexdigest()
        signature = hashlib.sha256(json_text([1, template_hash, rows, scaffolds, through, event]).encode()).hexdigest()
        marker = directory / '.distribution_checkpoint.json'
        published = ('index.html', 'summary.json', f'iter{through:02d}.html',
                     f'iter{through:02d}.json', 'distribution.log')
        if (marker.exists() and all((directory / name).is_file() for name in published)
                and json.loads(marker.read_text()).get('signature') == signature):
            print(f"[distribution] unchanged; report={directory / 'index.html'}", flush=True)
            return directory / 'index.html'
        report = build_report(rows, scaffolds, directory, through, getattr(args, 'primg_gbsa_elite_count', 10))
        report.update(project=args.project, event=event, signature=signature,
                      generated_at=datetime.now(timezone.utc).isoformat())
        html = render(report)
        encoded = json_text(report)
        for name in ('index.html', f'iter{through:02d}.html'):
            atomic_text(directory / name, html)
        for name in ('summary.json', f'iter{through:02d}.json'):
            atomic_text(directory / name, encoded)
        summary = log_summary(report)
        text = f"\n[{report['generated_at']}] through iter{through:02d}; {event}\n{summary}\nReport: {directory / 'index.html'}\n"
        with (directory / 'distribution.log').open('a') as log:
            log.write(text)
        atomic_text(marker, json_text({'signature': signature}))
        print('[distribution] ' + text.strip(), flush=True)
        return directory / 'index.html'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only chemical-space, substituent and distribution report')
    parser.add_argument('project_toml', type=Path)
    parser.add_argument('--iteration', type=int, help='Include references and iterations through N')
    parser.add_argument('--event', default='manual')
    ns = parser.parse_args(argv)
    if ns.iteration is not None and ns.iteration < 0:
        parser.error('--iteration must be nonnegative')
    from molnova.stages._logging import work_started
    work_started('[distribution] updating cumulative population report.')
    write_report(distribution_project(ns.project_toml), ns.iteration, ns.event)


if __name__ == '__main__':
    main()

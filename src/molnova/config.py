"""Project configuration API."""
from pathlib import Path
import math
from types import SimpleNamespace
from typing import Any

from molnova.paths import stage_directory
from molnova import _core


def combichem_3d_settings(filename):
    """Dedicated local pose-growth project; never initializes a legacy project."""
    import os
    import tomllib
    path = Path(filename).expanduser().resolve()
    with path.open('rb') as stream:
        raw = tomllib.load(stream)
    project_mode = 'combichem-3d' in raw or any(key in raw for key in ('dock-grid','reference-pose','amber-receptor','prime-gbsa-receptor'))
    inherited = {}
    if project_mode:
        section = raw.get('combichem-3d',{})
        if not isinstance(section,dict):
            raise ValueError('combichem-3d must be a TOML table')
        base_output = (path.parent/Path(raw.get('out-dir','output')).expanduser()).resolve()
        project = str(raw.get('project',path.stem))
        if not project or Path(project).name!=project:
            raise ValueError('Invalid project name')
        inherited = dict(raw)
        aliases = {'receptor':('dock-grid','receptor','prime-gbsa-receptor','amber-receptor'),
                   'parents':('reference-pose','parents','parent-pose','parent-poses'),
                   'reference-smiles':('reference-smiles',), 'history':('history','parent-history')}
        # Section values override shared keys, without invoking legacy project
        # initialization or changing its live database/selections.
        merged = {key:raw[key] for key in ('sidechain-library','max-iteration','target-count','reference-mcs') if key in raw}
        for target,keys in aliases.items():
            for key in keys:
                if key in raw:
                    merged[target]=raw[key]
                    break
        if 'receptor' not in merged:
            merged['receptor']=str(stage_directory(base_output,0,'prime_gbsa')/'receptor.maegz')
        if 'parents' not in merged:
            parent_root = stage_directory(base_output,1,'glide')
            merged['parents']=str(parent_root/('best_poses.sdf' if (parent_root/'best_poses.sdf').is_file() else 'best_poses.maegz'))
        merged['out-dir']=str(base_output/'combichem-3d')
        merged.update(section)
        raw = merged
    defaults = {
        'max-iteration': 1, 'sidechains-per-site': 10, 'conformers': 8,
        'sidechain-min-heavy-atoms': 5, 'sidechain-max-heavy-atoms': 10,
        'sidechain-selection': 'random', 'sidechain-include-h': False,
        'torsion-trials': 8, 'optimization-iterations': 250, 'minimum-valid-poses': 1,
        'core-displacement': .05, 'overlap-tolerance': .4,
        'maximum-receptor-clashes': 0, 'maximum-internal-clashes': 0,
        'one-four-policy': 'exclude', 'one-four-scale': .8,
        'include-water': False, 'include-cofactors': True, 'include-hydrogens': False,
        'force-field': 'auto', 'core-constraint': 'fixed', 'restraint-force': 1000.,
        'max-seconds': 1800., 'stop-mode': 'target', 'target-count': 10,
        'processes': min(16, os.cpu_count() or 1), 'seed': 20261025,
        'batch-size': 16, 'grace-seconds': 5., 'embed-timeout': 5,
        'prune-intermediates': False, 'filters': {}, 'site-candidates': {}, 'reference-mcs': None,
        'schrodinger': os.environ.get('SCHRODINGER','/home1/schrodinger/2026-3'),
        'infer-history': project_mode, 'inferred-parent-iteration': 1,
        'log-level': 'INFO', 'log-interval': 10.,
        'log-structures': 3,
        'growth-mode': 'frame-cycles', 'elite-count': 20, 'elite-rank-by': 'qed',
        'next-attachment-count': 3, 'attachment-distance': 'bonds', 'mcs-timeout': 60,
        'initial-attachment-max-cycles': {},
        'execution-host': None, 'remote-env-script': None, 'worker-hosts': {}, 'worker-ssh-known-hosts': None,
    }
    if project_mode:
        for key in defaults:
            if key in inherited and key not in raw:
                raw[key]=inherited[key]
    paths = ('receptor', 'parents', 'history', 'sidechain-library', 'reference-smiles', 'out-dir')
    unknown = set(raw) - set(defaults) - set(paths)
    if unknown:
        raise ValueError('Unknown combichem-3d settings: ' + ', '.join(sorted(unknown)))
    cfg = defaults | raw
    limits=cfg['initial-attachment-max-cycles']
    if isinstance(limits,str):
        parsed={}
        for entry in limits.split(','):
            pair=entry.strip().split(':')
            if len(pair)!=2 or not all(p.strip().isdigit() for p in pair):
                raise ValueError('initial-attachment-max-cycles must be a table or label:count list')
            label,value=(p.strip() for p in pair)
            if str(int(label)) in parsed:
                raise ValueError('Duplicate initial attachment cycle limit')
            parsed[str(int(label))]=int(value)
        limits=parsed
    if not isinstance(limits,dict):
        raise ValueError('initial-attachment-max-cycles must map initial labels to positive integers')
    normalized={}
    for label,value in limits.items():
        if (not isinstance(label,str) or not label.isdigit() or isinstance(value,bool)
                or not isinstance(value,int) or value<1 or str(int(label)) in normalized):
            raise ValueError('initial-attachment-max-cycles needs distinct nonnegative labels and positive integer limits')
        normalized[str(int(label))]=value
    cfg['initial-attachment-max-cycles']=normalized
    if normalized and cfg['growth-mode']!='frame-cycles':
        raise ValueError('initial-attachment-max-cycles requires growth-mode="frame-cycles"')
    if cfg['growth-mode']=='frame-cycles' and 'stop-mode' not in raw:
        cfg['stop-mode']='cycles'
    if 'history' not in cfg and (cfg['infer-history'] or cfg['growth-mode']=='frame-cycles'):
        output=(path.parent/Path(cfg['out-dir']).expanduser()).resolve()
        cfg['history']=str(output/'input-cache/inferred-history.json')
        cfg['history-source']='inferred'
    source_db = None
    reference_poses = None
    if project_mode and 'reference-smiles' not in cfg:
        source_db = base_output/f'{project}.sqlite'
        legacy = path.parent/f'{project}.sqlite'
        if not source_db.is_file() and legacy.is_file():
            source_db=legacy
        if not source_db.is_file():
            source_db=None
            value=inherited.get('reference-pose')
            if isinstance(value,str):
                reference_poses=(path.parent/Path(value).expanduser()).resolve()
        # Generated only in the dedicated output, after read-only source access.
        output=(path.parent/Path(cfg['out-dir']).expanduser()).resolve()
        cfg['reference-smiles']=str(output/'input-cache/references.smi')
    for key in paths:
        value = cfg.get(key)
        if not isinstance(value, str) or not value.strip() or '://' in value:
            extra = '; add [combichem-3d].history with explicit UID/core/fragment provenance' if key=='history' else ''
            raise ValueError(f'{key} must be a nonempty local path{extra}')
        cfg[key] = str((path.parent / Path(value).expanduser()).resolve())
    positive_ints = ('max-iteration', 'sidechains-per-site', 'conformers', 'optimization-iterations',
                     'minimum-valid-poses', 'target-count', 'processes', 'batch-size', 'embed-timeout', 'inferred-parent-iteration',
                     'sidechain-min-heavy-atoms', 'sidechain-max-heavy-atoms','elite-count','next-attachment-count','mcs-timeout')
    for key in (*positive_ints, 'torsion-trials', 'seed', 'maximum-receptor-clashes', 'maximum-internal-clashes', 'log-structures'):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key in positive_ints else 0):
            raise ValueError(f'{key} must be an integer >= {1 if key in positive_ints else 0}')
    if cfg['processes'] > 384 or cfg['seed'] > 2147483646:
        raise ValueError('processes must be <=384 and seed <=2147483646')
    for key in ('core-displacement', 'overlap-tolerance', 'one-four-scale', 'restraint-force', 'max-seconds', 'grace-seconds', 'log-interval'):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be a finite nonnegative number')
    if cfg['max-seconds'] <= 0 or cfg['restraint-force'] <= 0 or not 0 < cfg['one-four-scale'] <= 1:
        raise ValueError('max-seconds/restraint-force must be positive and one-four-scale in (0,1]')
    if cfg['log-interval'] <= 0 or cfg['log-level'] not in ('INFO','DEBUG','TRACE'):
        raise ValueError('log-interval must be positive and log-level must be INFO, DEBUG or TRACE')
    if cfg['log-structures']>20:
        raise ValueError('log-structures must be between 0 and 20')
    if cfg['next-attachment-count']>4:
        raise ValueError('next-attachment-count must be <=4')
    if cfg['elite-rank-by'] not in (None,'qed','similarity'):
        raise ValueError('elite-rank-by must be qed or similarity; external ranking requires an explicit imported-score protocol')
    if cfg['attachment-distance']!='bonds':
        raise ValueError('attachment-distance must be bonds (minimum shortest-path distance to previous attachment atoms)')
    if cfg['sidechain-min-heavy-atoms']>cfg['sidechain-max-heavy-atoms']:
        raise ValueError('sidechain-min-heavy-atoms must not exceed sidechain-max-heavy-atoms')
    for key in ('include-water', 'include-cofactors', 'include-hydrogens', 'prune-intermediates', 'infer-history', 'sidechain-include-h'):
        if not isinstance(cfg[key], bool):
            raise ValueError(f'{key} must be a boolean')
    for key, allowed in {'one-four-policy': ('exclude', 'include', 'scaled'),
                         'growth-mode': ('frame-cycles','frontier'),
                         'sidechain-selection': ('random','ordered'),
                         'force-field': ('auto', 'MMFF', 'UFF'),
                         'core-constraint': ('fixed', 'restraint'), 'stop-mode': ('target', 'time','cycles')}.items():
        if cfg[key] not in allowed:
            raise ValueError(f'{key} must be one of {allowed}')
    properties = {'mw', 'clogp', 'tpsa', 'qed', 'similarity', 'hbd', 'hba', 'rotatable_bonds', 'heavy_atoms'}
    if not isinstance(cfg['filters'], dict) or set(cfg['filters']) - properties:
        raise ValueError('filters must contain only supported property ranges')
    for key, limits in cfg['filters'].items():
        if (not isinstance(limits, list) or len(limits) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in limits)
                or limits[0] > limits[1]):
            raise ValueError(f'filters.{key} must be [finite minimum, finite maximum]')
    if not isinstance(cfg['site-candidates'], dict) or any(
            not isinstance(v, list) or not v or any(not isinstance(x, str) for x in v)
            for v in cfg['site-candidates'].values()):
        raise ValueError('site-candidates must map atom UIDs to nonempty fragment SMILES lists')
    host = cfg['execution-host']
    script = cfg['remote-env-script']
    if host is not None:
        import re
        if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*', host):
            raise ValueError('execution-host must be an SSH hostname or alias')
        if not isinstance(script, str) or not script.startswith('/') or '\n' in script:
            raise ValueError('remote-env-script must be an absolute remote environment script path')
    elif script is not None:
        raise ValueError('remote-env-script requires execution-host')
    hosts = cfg['worker-hosts']
    if not isinstance(hosts, dict):
        raise ValueError('worker-hosts must map SSH hostnames to worker counts')
    if hosts:
        import re
        if not host or not isinstance(script, str):
            raise ValueError('worker-hosts requires execution-host and remote-env-script')
        for name, slots in hosts.items():
            if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)
                    or isinstance(slots, bool) or not isinstance(slots, int) or not 1 <= slots <= 192):
                raise ValueError('worker-hosts requires hostnames and 1..192 workers per host')
        if len(hosts) > 2 or sum(hosts.values()) > 384:
            raise ValueError('worker-hosts supports at most two hosts and 384 workers total')
        if host not in hosts:
            raise ValueError('execution-host must be present in worker-hosts')
        cfg['processes'] = sum(hosts.values())
    known_hosts = cfg['worker-ssh-known-hosts']
    if known_hosts is not None:
        if not isinstance(known_hosts, str) or not known_hosts.strip():
            raise ValueError('worker-ssh-known-hosts must be a local shared file path')
        cfg['worker-ssh-known-hosts'] = str((path.parent / Path(known_hosts).expanduser()).resolve())
    else:
        cfg.pop('worker-ssh-known-hosts')
    if not hosts:
        cfg.pop('worker-hosts')
    cfg['config-path'] = str(path)
    if not isinstance(cfg['schrodinger'],str) or not cfg['schrodinger'].strip():
        raise ValueError('schrodinger must be a local installation path')
    cfg['schrodinger']=str((path.parent/Path(cfg['schrodinger']).expanduser()).resolve())
    if project_mode:
        cfg['project-input-toml'] = str(path)
        if source_db:
            cfg['reference-source-db'] = str(source_db)
        elif reference_poses:
            cfg['reference-source-poses'] = str(reference_poses)
        elif 'reference-smiles' not in raw:
            raise ValueError('No reference SMILES source: supply reference-smiles or an existing project DB/reference-pose SDF')
    return cfg


def distribution_settings(config):
    enabled = config.get('distribution-report', True)
    if not isinstance(enabled, bool):
        raise ValueError('distribution-report must be a boolean')
    return {'distribution_report_enabled': enabled}


def md_gbsa_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the optional, separate Desmond/Prime protocol and remote paths."""
    import getpass
    enabled = config.get('md-gbsa', False)
    if not isinstance(enabled, bool):
        raise ValueError('md-gbsa must be a boolean')
    if enabled and config.get('prime-gbsa', True) is False:
        raise ValueError('md-gbsa=true requires prime-gbsa=true')
    result = {'md_gbsa_enabled': enabled}
    for suffix, default, minimum, maximum in [
        ('snapshots', 20, 1, None), ('replicates', 1, 1, 100),
        ('gpus', 20, 1, 20), ('cpus', 384, 1, 384), ('seed', 20261021, 1, 2147483647),
    ]:
        key = 'md-gbsa-' + suffix
        value = config.get(key, default)
        if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
                or (maximum is not None and value > maximum)):
            requirement = 'a positive integer' if maximum is None else f'an integer between {minimum} and {maximum}'
            raise ValueError(f'{key} must be {requirement}')
        result['md_gbsa_' + suffix] = value
    for suffix, default, upper in [('production-ns', 5.0, None), ('equilibration-ns', 1.0, None),
                                   ('input-ratio', .5, 1.0), ('analysis-start-fraction', .2, 1.0)]:
        key = 'md-gbsa-' + suffix
        value = config.get(key, default)
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value <= 0 or (upper is not None and value > upper)
                or (suffix == 'analysis-start-fraction' and value == 1)):
            raise ValueError(f'{key} must be a finite positive number' + (' below 1' if suffix == 'analysis-start-fraction' else ''))
        result['md_gbsa_' + suffix.replace('-', '_')] = float(value)
    count = config.get('md-gbsa-input-count')
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
        raise ValueError('md-gbsa-input-count must be a positive integer')
    result['md_gbsa_input_count'] = count
    for suffix, default in [('ssh-host', 'slurm'), ('gpu-host', 'slurm-gpu'), ('cpu-host', 'slurm-cpu')]:
        value = config.get('md-gbsa-' + suffix, default)
        if (not isinstance(value, str) or not value or value.startswith('-')
                or any(c.isspace() for c in value) or ':' in value):
            raise ValueError(f'Invalid md-gbsa-{suffix}')
        result['md_gbsa_' + suffix.replace('-', '_')] = value
    for suffix, default in [('schrodinger', '/share/opt/schrodinger/2026-3'),
                            ('remote-work-dir', f'/homes1/{getpass.getuser()}/molnova-md-gbsa-runs')]:
        value = config.get('md-gbsa-' + suffix, default)
        if not isinstance(value, str) or not value.startswith('/') or any(c in value for c in '\n\r'):
            raise ValueError(f'md-gbsa-{suffix} must be an absolute remote path')
        result['md_gbsa_' + suffix.replace('-', '_')] = value
    return result


def distribution_project(path):
    """Resolve report inputs without initializing a project or external programs."""
    import tomllib
    path = Path(path).expanduser().resolve()
    with path.open('rb') as stream:
        config = tomllib.load(stream)
    output = (path.parent / Path(config['out-dir']).expanduser()).resolve()
    project = str(config.get('project', path.stem)).strip()
    if not project or Path(project).name != project:
        raise ValueError('Invalid project name')
    db_path = output / f'{project}.sqlite'
    if not db_path.is_file():
        legacy = path.parent / f'{project}.sqlite'
        if not legacy.is_file():
            raise FileNotFoundError(f'Project database not found: {db_path}')
        db_path = legacy  # Read in place; reporting never migrates a database.
    return SimpleNamespace(output=output, db_path=db_path, project=project,
                           prime_gbsa_elite_count=int(config.get('prime-gbsa-elite-count', 10)))


def prime_gbsa_flexdist(config: dict[str, Any]) -> float:
    """Zero preserves the frozen-receptor default; positive values are Angstroms."""
    value = config.get('prime-gbsa-flexdist', 0.0)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError('prime-gbsa-flexdist must be a finite non-negative number in Angstroms')
    return float(value)


def generation_settings(config: dict[str, Any], base: Path) -> dict[str, Any]:
    """Select the generator without requiring a REINVENT runtime for combichem."""
    mcs_timeout = config.get('mcs-timeout', 60)
    if isinstance(mcs_timeout, bool) or not isinstance(mcs_timeout, int) or mcs_timeout < 1:
        raise ValueError('mcs-timeout must be a positive integer in seconds')
    value = config.get('sidechain-library')
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError('sidechain-library must be a nonempty local path')
    library = (base / Path(value).expanduser()).resolve() if value is not None else None
    batch = config.get('sample-size', 3000)
    if library is not None and (isinstance(batch, bool) or not isinstance(batch, int) or batch < 1):
        raise ValueError('sample-size must be a positive integer for combichem')
    maximum = config.get('sidechain-max-heavy-atoms')
    if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1):
        raise ValueError('sidechain-max-heavy-atoms must be a positive integer')
    return {'sidechain_library': library,
            'mcs_timeout': mcs_timeout,
            'sidechain_max_heavy_atoms': maximum,
            'generation_method': 'combichem' if library is not None else 'reinvent4',
            'sample_size': int(batch)}


def log_level(config: dict[str, Any]) -> int:
    """Console verbosity; accept the hyphenated spelling for flat TOML consistency."""
    if ('log_level' in config and 'log-level' in config
            and config['log_level'] != config['log-level']):
        raise ValueError('log_level and log-level must agree')
    for key in ('log_level', 'log-level'):
        value = config.get(key, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be an integer >= 1')
    return config.get('log_level', config.get('log-level', 1))


def amber_settings(config: dict[str, Any], base: Path) -> dict[str, Any]:
    use_slurm = config.get('use-slurm', False)
    if not isinstance(use_slurm, bool):
        raise ValueError('use-slurm must be a boolean')
    for key in ('amber_gbsa', 'amber-gbsa', 'amber-gbsa-enabled', 'prime-gbsa'):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f'{key} must be a boolean')
    switches = [config[key] for key in ('amber_gbsa', 'amber-gbsa', 'amber-gbsa-enabled') if key in config]
    if switches and any(value != switches[0] for value in switches):
        raise ValueError('amber_gbsa, amber-gbsa and amber-gbsa-enabled must agree')
    enabled = switches[0] if switches else False
    prime_count = config.get('prime-gbsa-input-count', 50)
    if isinstance(prime_count, bool) or not isinstance(prime_count, int) or prime_count < 1:
        raise ValueError('prime-gbsa-input-count must be a positive integer')
    settings = {'amber_gbsa_enabled': enabled, 'prime_gbsa_enabled': config.get('prime-gbsa', True),
                'prime_gbsa_input_count': prime_count,
                'use_slurm': use_slurm}
    time_limit = config.get('amber-slurm-time-limit', 600)
    if isinstance(time_limit, bool) or not isinstance(time_limit, int) or time_limit < 1:
        raise ValueError('amber-slurm-time-limit must be a positive integer in seconds')
    hours, remainder = divmod(time_limit, 3600)
    minutes, seconds = divmod(remainder, 60)
    settings['amber_slurm_time_limit'] = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    for key, default in [('amber-cpus', 128)]:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
        settings[key.replace('-', '_')] = value
    for key in ('amber-input-count', 'amber-keep-count'):
        if key in config:
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
    if ('amber-input-count' in config and 'amber-keep-count' in config
            and config['amber-input-count'] != config['amber-keep-count']):
        raise ValueError('amber-input-count and legacy amber-keep-count must agree')
    # Retain the internal attribute for existing workers and API callers.
    settings['amber_keep_count'] = config.get('amber-input-count', config.get('amber-keep-count', 500))
    if enabled and settings['amber_keep_count'] < prime_count:
        raise ValueError('amber-input-count must be >= prime-gbsa-input-count')
    host = str(config.get('amber-ssh-host', 'tensor'))
    if not host or host.startswith('-') or any(c.isspace() for c in host):
        raise ValueError('Invalid amber-ssh-host')
    settings['amber_remote'] = {'host': host}
    for key, default in [
        ('amber-remote-env', '/home1/amber26/envs/AmberTools26'),
        ('amber-conda-init', '/home1/chchae/anaconda3/etc/profile.d/conda.sh'),
        ('amber-remote-work-dir', '/home1/chchae/molnova-amber-runs'),
    ]:
        value = str(config.get(key, default))
        if not value.startswith('/') or '\n' in value:
            raise ValueError(f'{key} must be an absolute remote path')
        settings['amber_remote'][key.removeprefix('amber-').replace('-', '_')] = value
    ssh_config = config.get('amber-ssh-config')
    if ssh_config:
        settings['amber_remote']['ssh_config'] = str((base / Path(ssh_config).expanduser()).resolve())
    if use_slurm:
        import getpass
        cpus = config.get('amber-slurm-cpus', 384)
        if isinstance(cpus, bool) or not isinstance(cpus, int) or not 1 <= cpus <= 384:
            raise ValueError('amber-slurm-cpus must be an integer between 1 and 384')
        settings['amber_cpus'] = cpus
        defaults = {
            'host': 'slurm', 'partition': 'cpu',
            'env': '/share/opt/amber26/envs/AmberTools26',
            'work-dir': f'/homes1/{getpass.getuser()}/molnova-amber-slurm-runs',
        }
        # The shared Amber installation has amber.sh but no conda installation.
        settings['amber_remote'].pop('conda_init', None)
        if 'amber-slurm-conda-init' in config:
            defaults['conda-init'] = config['amber-slurm-conda-init']
        keys = {'host': 'host', 'partition': 'partition', 'env': 'remote_env',
                'conda-init': 'conda_init', 'work-dir': 'remote_work_dir'}
        for suffix, default in defaults.items():
            key = f'amber-slurm-{suffix}'
            value = config.get(key, default)
            if not isinstance(value, str) or not value or '\n' in value or '\r' in value:
                raise ValueError(f'Invalid {key}')
            if suffix in ('host', 'partition'):
                if value.startswith('-') or any(c.isspace() for c in value):
                    raise ValueError(f'Invalid {key}')
            elif not value.startswith('/'):
                raise ValueError(f'{key} must be an absolute remote path')
            settings['amber_remote'][keys[suffix]] = value
    if 'amber-receptor' in config:
        receptor = Path(config['amber-receptor']).expanduser()
    else:
        output = (base / Path(config['out-dir']).expanduser()).resolve()
        receptor = stage_directory(output, 0, 'prime_gbsa') / 'receptor.maegz'
    settings['amber_receptor'] = (base / receptor).resolve()
    charge = config.get('amber-charge-method', 'bcc')
    if charge not in ('bcc', 'abcg2'):
        raise ValueError('amber-charge-method must be bcc or abcg2')
    settings['amber_charge_method'] = charge
    return settings


def debug_settings(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Scale population limits once at load time; preserve runtime/scientific knobs."""
    enabled = config.get('debug', False)
    if not isinstance(enabled, bool):
        raise ValueError('debug must be a boolean')
    settings['debug'] = enabled
    if enabled:
        for key in ('sample_size', 'target_count', 'amber_keep_count',
                    'prime_gbsa_input_count', 'prime_gbsa_elite_count', 'fep_input_count'):
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key.replace("_", "-")} must be a positive integer')
            settings[key] = max(1, value // 10)
    return settings


def glide_job_timeout(config: dict[str, Any]) -> int:
    value = config.get('glide-job-timeout', 1800)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('glide-job-timeout must be a non-negative integer (seconds)')
    return value


def schrodinger_cpu_settings(config: dict[str, Any]) -> dict[str, int | None]:
    settings = {}
    for stage in ("ligprep", "glide", "prime_gbsa"):
        key = f"{stage.replace('_', '-')}-cpus"
        value = config.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            if len(str(config["host"]).split()) != 1:
                raise ValueError(f"{key} requires a single compute host")
        settings[key.replace("-", "_")] = value
    return settings


def schrodinger_job_options(host: str, cpus: int | None = None,
                            *, split_jobs: bool = False) -> list[str]:
    """Use host slots for concurrency and explicit splitting when requested.

    Glide's JobDJ driver can otherwise combine small reference groups into
    very few jobs despite a large HOST slot count. Explicit NJOBS selects
    that driver's batch splitting (and disables automatic MQ driver selection
    in 2026-3); it does not change the docking backend or scientific input.
    Bare host entries retain the vendor's configured concurrency defaults.
    """
    if cpus is not None:
        if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
            raise ValueError("Schrodinger cpus must be a positive integer")
        if len(host.split()) != 1:
            raise ValueError("CPU override requires a single compute host")
        host = f"{host.split(':', 1)[0]}:{cpus}"
    options = ["-HOST", host]
    slots = [entry.rpartition(":")[2] for entry in host.split()]
    if split_jobs and slots and all(s.isdigit() and int(s) > 0 for s in slots):
        options += ["-NJOBS", str(sum(int(s) for s in slots))]
    return options


def prime_gbsa_job_options(host: str, ligand_count: int) -> list[str]:
    """Queue one ligand per subjob; host slots bound concurrent execution."""
    if isinstance(ligand_count, bool) or not isinstance(ligand_count, int) or ligand_count < 1:
        raise ValueError("prime_gbsa ligand_count must be a positive integer")
    names = list(dict.fromkeys(entry.split(':', 1)[0] for entry in host.split()))
    if not names:
        raise ValueError("prime_gbsa requires a configured host")
    # HOST limits are per host, not a shared limit. Divide seven slots across
    # hosts rather than accidentally allowing seven on every host.
    slots = [0] * len(names)
    for index in range(7):
        slots[index % len(names)] += 1
    hosts = " ".join(f"{name}:{count}" for name, count in zip(names, slots) if count)
    return ["-HOST", hosts, "-NJOBS", str(ligand_count)]


def glide_group_resources(host, cpus, counts, reserved=()):
    """Share explicit host slots by prepared-ligand count, reserving live jobs.

    A missing/uncertain saved host request consumes the full budget until
    recovery establishes completion. Bare hosts keep vendor resource defaults.
    Return only groups that can launch; others wait for a later worker pass.
    """
    host = schrodinger_job_options(host, cpus)[1]
    for count in counts.values():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('Glide group ligand counts must be positive integers')

    def slots(request):
        result = {}
        for entry in (request or '').split():
            name, colon, value = entry.rpartition(':')
            if not colon or not value.isdigit() or int(value) < 1:
                return None
            result[name] = result.get(name, 0) + int(value)
        return result or None

    available = slots(host)
    if available is None:
        return {ref: host for ref in counts}  # No explicit budget was configured.
    for request in reserved:
        used = slots(request)
        if used is None or any(name not in available for name in used):
            return {}
        for name, value in used.items():
            available[name] -= value
    # Preserve legacy requests even when they exceed a newly reduced budget.
    if any(value < 0 for value in available.values()):
        return {}

    budget = min(sum(available.values()), sum(counts.values()))
    allocation = {ref: 0 for ref in counts}
    # Give nonempty groups one slot first. If cores are scarce, larger groups
    # launch first; deferred groups receive slots after native jobs terminate.
    for ref in sorted(counts, key=lambda ref: (-counts[ref], ref))[:budget]:
        allocation[ref] = 1
    remaining = budget - sum(allocation.values())
    while remaining:
        active = [ref for ref in counts if allocation[ref] < counts[ref]]
        total = sum(counts[ref] for ref in active)
        shares = {ref: divmod(remaining * counts[ref], total) for ref in active}
        for ref in active:
            addition = min(shares[ref][0], counts[ref] - allocation[ref])
            allocation[ref] += addition
            remaining -= addition
        for ref in sorted(active, key=lambda ref: (-shares[ref][1], ref)):
            if remaining and allocation[ref] < counts[ref]:
                allocation[ref] += 1
                remaining -= 1

    requests = {}
    for ref in sorted(allocation):
        need = allocation[ref]
        entries = []
        for name in available:
            take = min(need, available[name])
            if take:
                entries.append(f'{name}:{take}')
                available[name] -= take
                need -= take
        if entries:
            requests[ref] = ' '.join(entries)
    return requests


def aizynth_process_count(config: dict[str, Any]) -> int:
    value = config.get("aizynth-nproc", 8)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("aizynth-nproc must be a positive integer")
    return value


def sa_score_process_count(config: dict[str, Any]) -> int:
    value = config.get('sa-score-cpus', 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('sa-score-cpus must be a positive integer')
    return value


def sa_accepted_ratio(config: dict[str, Any]) -> float:
    value = config.get('sa-accepted-ratio', 1.0)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 < value <= 1):
        raise ValueError('sa-accepted-ratio must be a number greater than 0 and at most 1')
    return float(value)


def fep_settings(config: dict[str, Any]) -> dict[str, Any]:
    enabled = config.get('fep-enabled', False)
    count = config.get('fep-input-count', 20)
    if not isinstance(enabled, bool):
        raise ValueError('fep-enabled must be a boolean')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('fep-input-count must be a positive integer')
    return {'fep_enabled': enabled, 'fep_input_count': count}


def elite_mcs(config: dict[str, Any]) -> str | None:
    value = config.get('elite-mcs')
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError('elite-mcs must be a nonempty SMARTS string')
    return value.strip()


def elite_mcs_fixed(config: dict[str, Any]) -> bool:
    value = config.get('elite-mcs-fixed', False)
    if not isinstance(value, bool):
        raise ValueError('elite-mcs-fixed must be a boolean')
    return value


def prime_gbsa_license_retry_settings(config: dict[str, Any]) -> dict[str, int]:
    settings = {}
    for key, default, minimum in (("prime-gbsa-license-retry-seconds", 300, 1),
                                  ("prime-gbsa-license-retries", 3, 0)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        settings["prime_gbsa_" + key.removeprefix("prime-gbsa-").replace("-", "_")] = value
    return settings


def remote_reinvent_settings(config: dict[str, Any]) -> dict[str, str] | None:
    """Remote paths belong to the SSH host, never to the local TOML directory."""
    host = str(config.get("reinvent-ssh-host", "")).strip()
    if not host:
        return None
    if host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("reinvent-ssh-host must be an SSH hostname or alias")
    settings = {
        "host": host,
        "env": str(config.get("reinvent-remote-env", "")),
        "work_dir": str(config.get("reinvent-remote-work-dir", "")),
    }
    for key in ("env", "work_dir"):
        if not settings[key].startswith("/"):
            raise ValueError(f"REINVENT remote {key} must be an absolute remote path")
    if not str(config["libinvent-prior"]).startswith("/"):
        raise ValueError("Remote libinvent-prior must be an absolute remote path")
    return settings


def load_project_toml(path: str | Path) -> dict[str, Any]:
    return _core.load_project_toml(path)


def remote_aizynth_settings(config: dict[str, Any]) -> dict[str, str] | None:
    """AiZynthFinder environment, configuration and work paths belong to SSH host."""
    host = str(config.get("aizynth-ssh-host", "")).strip()
    if not host:
        return None
    if host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("aizynth-ssh-host must be an SSH hostname or alias")
    settings = {
        "host": host,
        "env": str(config.get("aizynth-remote-env", "")),
        "work_dir": str(config.get("aizynth-remote-work-dir", "")),
        "config": str(config.get("aizynth-config", "")),
    }
    for key in ("env", "work_dir", "config"):
        if not settings[key].startswith("/"):
            raise ValueError(f"AiZynthFinder remote {key} must be an absolute remote path")
    return settings


def configure_project(
    path: str | Path, schrodinger: str | Path | None = None
) -> SimpleNamespace:
    return _core.configure_project(path, Path(schrodinger) if schrodinger else None)

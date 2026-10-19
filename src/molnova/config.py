"""Project configuration API."""
from pathlib import Path
import math
from types import SimpleNamespace
from typing import Any

from molnova import _core


def prime_flexdist(config: dict[str, Any]) -> float:
    """Zero preserves the frozen-receptor default; positive values are Angstroms."""
    value = config.get('prime-flexdist', 0.0)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError('prime-flexdist must be a finite non-negative number in Angstroms')
    return float(value)


def generation_settings(config: dict[str, Any], base: Path) -> dict[str, Any]:
    """Select the generator without requiring a REINVENT runtime for combichem."""
    value = config.get('sidechain-library')
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError('sidechain-library must be a nonempty local path')
    library = (base / Path(value).expanduser()).resolve() if value is not None else None
    batch = config.get('sample-size', 3000)
    if library is not None and (isinstance(batch, bool) or not isinstance(batch, int) or batch < 1):
        raise ValueError('sample-size must be a positive integer for combichem')
    return {'sidechain_library': library,
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
    for key in ('amber_gbsa', 'amber-gbsa-enabled'):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f'{key} must be a boolean')
    if ('amber_gbsa' in config and 'amber-gbsa-enabled' in config
            and config['amber_gbsa'] != config['amber-gbsa-enabled']):
        raise ValueError('amber_gbsa and amber-gbsa-enabled must agree')
    enabled = config.get('amber_gbsa', config.get('amber-gbsa-enabled', False))
    prime_count = config.get('gbsa-input-count', 50 if enabled else 200)
    if isinstance(prime_count, bool) or not isinstance(prime_count, int) or prime_count < 1:
        raise ValueError('gbsa-input-count must be a positive integer')
    settings = {'amber_gbsa_enabled': enabled, 'gbsa_input_count': prime_count}
    # amber-keep-count now limits the Glide-ranked population before Amber.
    for key, default in [('amber-cpus', 128), ('amber-keep-count', 200)]:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
        settings[key.replace('-', '_')] = value
    if enabled and settings['amber_keep_count'] < prime_count:
        raise ValueError('amber-keep-count must be >= gbsa-input-count')
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
    receptor = Path(config.get('amber-receptor', Path(config['out-dir']) / 'iter0/mmgbsa/receptor.maegz')).expanduser()
    settings['amber_receptor'] = (base / receptor).resolve()
    charge = config.get('amber-charge-method', 'bcc')
    if charge not in ('bcc', 'abcg2'):
        raise ValueError('amber-charge-method must be bcc or abcg2')
    settings['amber_charge_method'] = charge
    return settings


def glide_job_timeout(config: dict[str, Any]) -> int:
    value = config.get('glide-job-timeout', 1800)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('glide-job-timeout must be a non-negative integer (seconds)')
    return value


def schrodinger_cpu_settings(config: dict[str, Any]) -> dict[str, int | None]:
    settings = {}
    for stage in ("ligprep", "glide", "mmgbsa"):
        key = f"{stage}-cpus"
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


def mmgbsa_job_options(host: str, ligand_count: int) -> list[str]:
    """Queue one ligand per subjob; host slots bound concurrent execution."""
    if isinstance(ligand_count, bool) or not isinstance(ligand_count, int) or ligand_count < 1:
        raise ValueError("MM-GBSA ligand_count must be a positive integer")
    hosts = " ".join(f"{entry.split(':', 1)[0]}:7" for entry in host.split())
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
    value = config.get('sa-accepted-ratio', 0.75)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 < value <= 1):
        raise ValueError('sa-accepted-ratio must be a number greater than 0 and at most 1')
    return float(value)


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


def gbsa_license_retry_settings(config: dict[str, Any]) -> dict[str, int]:
    settings = {}
    for key, default, minimum in (("gbsa-license-retry-seconds", 300, 1),
                                  ("gbsa-license-retries", 3, 0)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        settings[key.replace("-", "_")] = value
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

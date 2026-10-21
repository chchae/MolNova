"""Output layout shared by stages, supervision and durable job recovery."""
from pathlib import Path
import re


STAGE_DIRECTORIES = {
    'scaffold': '01-scaffold',
    'reinvent': '02-generate',
    'combichem': '02-generate',
    'druglike': '03-druglike',
    'synthetic_feasibility': '04-synthetic_feasibility',
    'ligprep': '05-ligprep',
    'glide': '06-glide',
    'amber_gbsa': '07-amber_gbsa',
    'prime_gbsa': '08-prime_gbsa',
    'md_gbsa': '09-md_gbsa',
    'fep': '09-fep',
}


def iteration_directory(output, iteration):
    """Use padded names for new work; retain existing legacy launch paths."""
    output = Path(output)
    current = output / f'iter{iteration:02d}'
    legacy = output / f'iter{iteration}'
    if legacy != current and (legacy.exists() or legacy.is_symlink()):
        if current.exists() or current.is_symlink():
            raise RuntimeError(f'Ambiguous iteration directories; reconcile {legacy} and {current}')
        return legacy
    return current


def stage_in_iteration(directory, stage):
    """Keep an existing iteration's layout stable, including not-yet-run stages."""
    directory = Path(directory)
    numbered = directory / STAGE_DIRECTORIES[stage]
    legacy = directory / stage
    # Iterations >=10 have identical old/new names. Their existing children
    # identify the layout; do not move saved submissions or scientific inputs.
    old_layout = bool(re.fullmatch(r'iter[0-9]', directory.name)) or any(
        (directory / name).exists() or (directory / name).is_symlink()
        for name in STAGE_DIRECTORIES
    )
    if old_layout:
        if any((directory / name).exists() or (directory / name).is_symlink()
               for name in set(STAGE_DIRECTORIES.values())):
            raise RuntimeError(f'Ambiguous stage directories; reconcile layouts in {directory}')
        return legacy
    if stage == 'fep':
        following_md = directory / '10-fep'
        if following_md.exists() and numbered.exists():
            raise RuntimeError(f'Ambiguous FEP directories; reconcile layouts in {directory}')
        if following_md.exists() or (not numbered.exists() and (directory / '09-md_gbsa').exists()):
            return following_md
    return numbered


def stage_directory(output, iteration, stage):
    return stage_in_iteration(iteration_directory(output, iteration), stage)


def stage_manifests(output, stage, filename):
    """Find both layouts so saved jobs always reserve their project resources."""
    output = Path(output)
    return sorted({*output.glob(f'iter*/{stage}/{filename}'),
                   *output.glob(f'iter*/{STAGE_DIRECTORIES[stage]}/{filename}')})

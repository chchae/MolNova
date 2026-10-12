"""Conservative, fixed-column normalization of prepared Amber receptor PDBs.

Standalone standard-library module, also shipped beside the remote worker.
Only atom-name and residue-name columns change; no atoms or boundaries are removed.
"""
import math
from pathlib import Path


NORMALIZATION_VERSION = 'receptor-pdb-v1'
# ff14SB amino(nt/ct)12 uses ACE H1/H2/H3 and NME C/H1/H2/H3.
# Rename template atoms, preserving the library's types, charges and bonds.
CAP_TEMPLATE_NAMES = (
    'set ACE.1.H1 name HH31\nset ACE.1.H2 name HH32\nset ACE.1.H3 name HH33\n'
    'set NME.1.C name CH3\nset NME.1.H1 name HH31\n'
    'set NME.1.H2 name HH32\nset NME.1.H3 name HH33\n'
)
_AMINO_ACIDS = set('ALA ARG ASN ASP CYS CYX CYM GLN GLU GLY HID HIE HIP HIS ILE LEU LYS LYN MET PHE PRO SER THR TRP TYR VAL'.split())


def normalize_receptor_pdb(source, destination, log_path):
    try:
        return _normalize_receptor_pdb(source, destination, log_path)
    except Exception as exc:
        with Path(log_path).open('a') as log:
            log.write(f'\nReceptor normalization failed: {type(exc).__name__}: {exc}\n')
        raise


def _normalize_receptor_pdb(source, destination, log_path):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Receptor normalization requires a separate output PDB')
    # Preserve line endings and all columns outside the two names verbatim.
    lines = source.read_bytes().decode('ascii').splitlines(keepends=True)
    residues = {}
    segment = 0
    models = 0
    for index, line in enumerate(lines):
        record = line[:6].strip()
        if record == 'MODEL':
            models += 1
            if models > 1:
                raise ValueError('Multiple receptor PDB models are unsupported')
        if record in ('TER', 'ENDMDL'):
            segment += 1
        if record not in ('ATOM', 'HETATM'):
            continue
        if len(line.rstrip('\r\n')) < 54:
            raise ValueError(f'Truncated PDB atom record at line {index + 1}')
        key = (segment, line[21:22], line[22:26], line[26:27])
        residue = residues.setdefault(key, {'name': line[17:20], 'atoms': {}})
        label = f'{residue["name"].strip()} {key[1].strip() or "_"}{key[2].strip()}{key[3].strip()}'
        if residue['name'] != line[17:20]:
            raise ValueError(f'Conflicting residue names for {label}')
        name = line[12:16].strip()
        if line[16:17].strip() or name in residue['atoms']:
            raise ValueError(f'Alternate/duplicate atom {name} in {label}; prepare one conformer first')
        xyz = tuple(float(line[start:start+8]) for start in (30, 38, 46))
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError(f'Non-finite coordinates in {label} atom {name}')
        residue['atoms'][name] = (index, xyz)
    if not residues:
        raise ValueError('No receptor atoms in PDB')

    changes = []
    names = {}
    ordered = list(residues)
    for position, (key, residue) in enumerate(residues.items()):
        original = residue['name'].strip()
        atoms = residue['atoms']
        label = f'{original} {key[1].strip() or "_"}{key[2].strip()}{key[3].strip()}'
        new_residue = original
        mapping = {}
        if original in ('NMA', 'NME'):
            new_residue = 'NME'
            mapping = {'CA': 'CH3', '1HA': 'HH31', '2HA': 'HH32', '3HA': 'HH33',
                       'HA1': 'HH31', 'HA2': 'HH32', 'HA3': 'HH33'}
            expected = {'N', 'H', 'CH3', 'HH31', 'HH32', 'HH33'}
        elif original == 'ACE':
            mapping = {'1H': 'HH31', '2H': 'HH32', '3H': 'HH33'}
            expected = {'C', 'O', 'CH3', 'HH31', 'HH32', 'HH33'}
        else:
            expected = None
        if expected is not None:
            normalized = [mapping.get(atom, atom) for atom in atoms]
            if len(set(normalized)) != len(normalized) or set(normalized) != expected:
                raise ValueError(f'Unexpected cap structure in {label}: atoms={list(atoms)}; expected {sorted(expected)} after normalization')
            cap = {mapping.get(atom, atom): data for atom, data in atoms.items()}
            edges = [('CH3', h, 0.7, 1.3) for h in ('HH31', 'HH32', 'HH33')]
            edges += [('N', 'H', 0.7, 1.3), ('N', 'CH3', 1.1, 1.8)] if new_residue == 'NME' else [('CH3', 'C', 1.1, 1.8), ('C', 'O', 1.0, 1.5)]
            for atom, (index, _) in cap.items():
                element = lines[index][76:78].strip() if len(lines[index].rstrip('\r\n')) >= 78 else ''
                if element and element != atom[0]:
                    raise ValueError(f'Unexpected element {element} for {label} atom {atom}')
            for a, b, lower, upper in edges:
                distance = math.dist(cap[a][1], cap[b][1])
                if not lower <= distance <= upper:
                    raise ValueError(f'Unexpected cap geometry in {label}: {a}-{b} distance {distance:.3f} A')
        if original in ('HIS', 'HID', 'HIE', 'HIP'):
            hydrogens = {name for name in ('HD1', 'HE2') if name in atoms}
            protonation = {frozenset({'HD1'}): 'HID', frozenset({'HE2'}): 'HIE',
                           frozenset({'HD1', 'HE2'}): 'HIP'}.get(frozenset(hydrogens))
            if protonation is None:
                raise ValueError(f'{label} has neither HD1 nor HE2; explicit protonation/preparation is required')
            if original != 'HIS' and original != protonation:
                raise ValueError(f'{label} ring hydrogens conflict with explicit {original} (indicate {protonation})')
            new_residue = protonation
        names[key] = new_residue
        if original != new_residue:
            changes.append(f'{label}: residue {original} -> {new_residue}')
        for atom, (index, _) in atoms.items():
            new_atom = mapping.get(atom, atom)
            if new_atom != atom:
                changes.append(f'{label}: atom {atom} -> {new_atom}')
                field = new_atom.ljust(4) if len(new_atom) == 4 else (' ' + new_atom).ljust(4)
                lines[index] = lines[index][:12] + field + lines[index][16:]
            lines[index] = lines[index][:17] + new_residue.rjust(3) + lines[index][20:]
        if original == 'ACE' and position + 1 < len(ordered):
            following_key = ordered[position + 1]
            following = residues[following_key]
            # Same TER segment and chain, adjacent in source, and a peptide C-N
            # distance of 1.0–1.8 A establish connection without guessing from numbering.
            if (following_key[:2] == key[:2] and following['name'].strip() in _AMINO_ACIDS
                    and 'N' in following['atoms']
                    and 1.0 <= math.dist(atoms['C'][1], following['atoms']['N'][1]) <= 1.8
                    and 'H1' in following['atoms']):
                if 'H' in following['atoms']:
                    raise ValueError(f'ACE-connected residue {following_key} contains both H and H1')
                index = following['atoms']['H1'][0]
                lines[index] = lines[index][:12] + ' H  ' + lines[index][16:]
                changes.append(f'{following["name"].strip()} {following_key[1]}{following_key[2].strip()}{following_key[3].strip()}: atom H1 -> H (ACE-connected)')

    segment = 0
    for index, line in enumerate(lines):
        record = line[:6].strip()
        if record == 'TER':
            key = (segment, line[21:22], line[22:26], line[26:27])
            if key in names:
                lines[index] = line[:17] + names[key].rjust(3) + line[20:]
            segment += 1
        elif record == 'ENDMDL':
            segment += 1
    pending = destination.with_name(destination.name + '.pending')
    pending.write_bytes(''.join(lines).encode('ascii'))
    pending.replace(destination)
    with Path(log_path).open('a') as log:
        log.write(f'\nReceptor normalization {NORMALIZATION_VERSION}: {source.name} -> {destination.name}\n')
        log.write('\n'.join(changes) + '\n' if changes else 'No name changes required.\n')
    return changes

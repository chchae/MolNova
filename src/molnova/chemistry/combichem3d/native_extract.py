"""Read-only structure export, run using the installed Schrödinger Python.

No embedding, minimization, docking, hydrogen addition or ancestry inference.
"""
import argparse
import json
from pathlib import Path
import zipfile


def receptor_member(filename):
    with zipfile.ZipFile(filename) as archive:
        names=[r.filename for r in archive.infolist() if not r.is_dir() and r.filename.endswith('_recep.mae')]
        if len(names)!=1:
            raise ValueError('Grid ZIP must contain exactly one *_recep.mae receptor; found '+str(len(names)))
        return names[0],archive.read(names[0]).decode('utf-8')


def record(st, source_record):
    return dict(source_record=source_record,title=st.title,
                atomic_numbers=[a.atomic_number for a in st.atom],
                formal_charges=[a.formal_charge for a in st.atom],
                xyz=st.getXYZ().tolist(),residues=[a.pdbres.strip() for a in st.atom],
                source_atom_indices=[a.index for a in st.atom],bond_count=sum(1 for _ in st.bond))


def main(argv=None):
    from schrodinger import structure
    import schrodinger
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parents')
    parser.add_argument('--grid')
    parser.add_argument('--output',required=True)
    ns=parser.parse_args(argv)
    output=Path(ns.output)
    output.mkdir(parents=True,exist_ok=True)
    audit=dict(schrodinger_version=str(getattr(schrodinger,'__version__','unreported')))
    if ns.parents:
        records,skipped=[],[]
        with structure.StructureReader(ns.parents) as reader, structure.StructureWriter(str(output/'parents.sdf')) as writer:
            for index,st in enumerate(reader,1):
                # Official Glide pose-viewer receptor marker, not an atom-count
                # heuristic. Never silently discard an unmarked large ligand.
                if st.property.get('b_glide_receptor',False):
                    skipped.append(dict(source_record=index,title=st.title,reason='b_glide_receptor'))
                    continue
                records.append(record(st,index))
                writer.append(st)
        if not records:
            raise ValueError('reference-pose contains no ligand structures')
        audit['parents']=records
        audit['excluded_receptor_records']=skipped
    if ns.grid:
        member,text=receptor_member(ns.grid)
        with structure.StructureReader.fromString(text) as reader:
            structures=list(reader)
        if len(structures)!=1:
            raise ValueError('Grid receptor member must contain exactly one structure; none will be silently dropped')
        st=structures[0]
        st.write(str(output/'receptor.pdb'))
        audit['receptor']=record(st,1)
        audit['grid_member']=member
    (output/'atom_source_map.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':
    main()
